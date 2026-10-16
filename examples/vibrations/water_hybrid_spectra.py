"""M2: water optimization, AD Hessian and spectra for PBE0/B3LYP/WB97XD.

Edit XC below, or call run(xc). Each completed stage saves its own JSON;
optimize(xc), harmonic(xc), and spectra(xc) are also ordinary Python functions
for separate bounded jobs. All electronic work uses MolecularEnergy. The
original WB97XD calculation includes long-range exchange and original CHG.
"""
import hashlib
import json
from pathlib import Path
import platform
import runpy
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

from gradscf import dft, gto
from gradscf_tools.molecular_response import MolecularEnergy
from gradscf_tools.vibrations import normal_modes, AMU_TO_ELECTRON_MASS


XC = "pbe0"
METHODS = ("pbe0", "b3lyp", "wb97xd")
SYMBOLS = ["O", "H", "H"]
INITIAL = np.array([[0., 0., 0.], [0., 1.43, 1.11], [0., -1.43, 1.11]])
MASSES = {"H2O": [15.99491461957, 1.00782503223, 1.00782503223],
          "D2O": [15.99491461957, 2.01410177812, 2.01410177812]}
# Reuse the same mode JVPs, intensity conversions and FD checks as HF/PBE.
calculate_spectrum = runpy.run_path(str(Path(__file__).with_name(
    "water_ir_raman.py")))["calculate_spectrum"]


def folder(xc):
    if xc not in METHODS:
        raise ValueError(f"Choose one of {METHODS}.")
    return Path("reproducibility/ad_showcase") / f"M2_water_{xc}_spectra"


def molecular_energy(xc, coordinates):
    mol = gto.M(atom=list(zip(SYMBOLS, coordinates)), basis="6-31g*", unit="Bohr", cart=True)
    mf = dft.RKS(mol, xc=xc, grids_level=3, conv_tol=1e-12, conv_tol_density=1e-11,
                 conv_tol_grad=1e-10, max_cycle=200, integral_backend="native").run()
    return MolecularEnergy(mf)


def metadata(xc):
    names = ("src/gradscf_tools/molecular_response.py", "src/gradscf_tools/vibrations.py",
             "src/gradscf/scf/orbital_optimization.py", "src/gradscf/scf/uks.py",
             "src/gradscf/scf/rsh.py", "src/gradscf/scf/inputs/grid.py",
             "src/gradscf/dft/dispersion.py", "src/gradscf/integrals/grids/ao.py")
    return dict(method="RKS", xc=xc.upper(), basis="6-31g*", cart=True, symbols=SYMBOLS,
                grid_level=3, grid_response=True, integral_backend="native", grid_ao_backend="jax",
                dispersion="original CHG" if xc == "wb97xd" else "none",
                b3lyp_convention="VWN-RPA" if xc == "b3lyp" else None,
                orbital_gradient_tolerance=1e-9, implicit_tolerance=1e-10,
                seed_energy_tolerance=1e-12, seed_density_tolerance=1e-11,
                seed_orbital_tolerance=1e-10, dtype="float64", hostname=platform.node(),
                jax_version=jax.__version__, numpy_version=np.__version__,
                python_version=platform.python_version(), devices=[str(x) for x in jax.devices()],
                runtime_source_hashes={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                      for name in names})


def save(xc, name, data):
    output = folder(xc)
    output.mkdir(parents=True, exist_ok=True)
    (output / name).write_text(json.dumps(data, indent=2)+"\n")


def optimize(xc):
    energy = molecular_energy(xc, INITIAL)
    calculate = jax.jit(jax.value_and_grad(energy))
    history = []

    def objective(flat):
        coordinates = flat.reshape(3, 3)
        value, gradient = calculate(coordinates)
        value, gradient = float(value), np.asarray(gradient)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise RuntimeError(f"{xc}: electronic energy/gradient failed.")
        history.append(dict(energy_hartree=value, coordinates_bohr=coordinates.tolist(),
                            max_gradient=float(np.max(np.abs(gradient)))))
        print(xc, "optimization", len(history), value, history[-1]["max_gradient"], flush=True)
        return value, gradient.ravel()

    start = perf_counter()
    minimum = minimize(objective, INITIAL.ravel(), jac=True, method="BFGS",
                       options={"gtol": 1e-7, "maxiter": 80})
    coordinates = minimum.x.reshape(3, 3)
    value, gradient = calculate(coordinates)
    gradient = np.asarray(gradient)
    if not np.isfinite(gradient).all() or np.max(np.abs(gradient)) > 1e-7:
        raise RuntimeError(f"{xc}: geometry did not converge: {minimum.message}")
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-8, rtol=0)
    np.testing.assert_allclose(np.cross(coordinates, gradient).sum(axis=0), 0., atol=1e-8, rtol=0)
    bonds = coordinates[1:]-coordinates[0]
    lengths = np.linalg.norm(bonds, axis=1)
    data = metadata(xc)
    data.update(coordinates_bohr=coordinates.tolist(), energy_hartree=float(value),
                gradient_hartree_per_bohr=gradient.tolist(), bond_lengths_bohr=lengths.tolist(),
                angle_degrees=float(np.degrees(np.arccos(np.dot(*bonds)/np.prod(lengths)))),
                optimizer_success=bool(minimum.success), optimizer_message=str(minimum.message),
                iterations=int(minimum.nit), gradient_tolerance=1e-7, history=history,
                elapsed_seconds=perf_counter()-start)
    save(xc, "geometry.json", data)
    return data


def harmonic(xc):
    geometry = json.loads((folder(xc)/"geometry.json").read_text())
    coordinates = jnp.asarray(geometry["coordinates_bohr"])
    energy = molecular_energy(xc, np.asarray(coordinates))
    gradient = jax.jit(jax.grad(energy))
    derivative = np.asarray(gradient(coordinates))
    assert np.isfinite(derivative).all() and np.max(np.abs(derivative)) < 1e-7
    hvp = jax.jit(lambda r, v: jax.jvp(jax.grad(energy), (r,), (v,))[1])
    start = perf_counter()
    columns, column_seconds = [], []
    for index, direction in enumerate(np.eye(9).reshape(9, 3, 3)):
        column_start = perf_counter()
        column = np.asarray(hvp(coordinates, jnp.asarray(direction)))
        if not np.isfinite(column).all():
            raise RuntimeError(f"{xc}: Hessian column {index+1} is nonfinite.")
        columns.append(column.ravel())
        column_seconds.append(perf_counter()-column_start)
        print(xc, "Hessian column", index+1, "seconds", column_seconds[-1], flush=True)
    hessian = np.column_stack(columns)
    seconds = perf_counter()-start
    np.testing.assert_allclose(hessian, hessian.T, atol=2e-6, rtol=0)
    direction = np.random.default_rng(13).normal(size=(3, 3))
    direction /= np.linalg.norm(direction)
    product = np.asarray(hvp(coordinates, jnp.asarray(direction)))
    np.testing.assert_allclose(product.ravel(), hessian@direction.ravel(), atol=1e-7, rtol=0)
    checks = []
    for step in (1e-3, 3e-4, 1e-4):
        numerical = (gradient(coordinates+step*direction)-gradient(coordinates-step*direction))/(2*step)
        error = float(np.max(np.abs(numerical-product)))
        if not np.isfinite(error):
            raise RuntimeError(f"{xc}: displaced gradient failed.")
        checks.append(dict(step_bohr=step, max_hvp_error=error))
        print(xc, "HVP FD", step, error, flush=True)
    assert checks[-1]["max_hvp_error"] < 2e-6
    translations = np.tile(np.eye(3), (3, 1))
    np.testing.assert_allclose(hessian@translations, 0., atol=1e-8, rtol=0)
    centered = np.asarray(coordinates)-np.mean(coordinates, axis=0)
    rotations = np.column_stack([np.cross(centered, axis).ravel() for axis in np.eye(3)])
    isotopes = {}
    for label, masses in MASSES.items():
        modes = normal_modes(hessian, coordinates, masses)
        assert modes.rigid_rank == 6 and np.all(modes.frequencies_cm1 > 0)
        vectors = modes.displacements.reshape(9, 3)
        mass = np.repeat(np.asarray(masses)*AMU_TO_ELECTRON_MASS, 3)
        np.testing.assert_allclose(vectors.T@(mass[:, None]*vectors), np.eye(3), atol=1e-10, rtol=0)
        isotopes[label] = dict(masses_amu=masses, frequencies_cm1=modes.frequencies_cm1.tolist(),
                              displacements=modes.displacements.tolist(), rigid_residual=modes.rigid_residual)
        print(xc, label, modes.frequencies_cm1, flush=True)
    data = metadata(xc)
    data.update(coordinates_bohr=np.asarray(coordinates).tolist(), energy_hartree=geometry["energy_hartree"],
                gradient_hartree_per_bohr=derivative.tolist(), hessian_hartree_per_bohr2=hessian.tolist(),
                isotopes=isotopes, finite_difference_direction=direction.tolist(), finite_difference_checks=checks,
                raw_hessian_asymmetry=float(np.max(np.abs(hessian-hessian.T))),
                raw_translation_residual_hartree_per_bohr2=float(np.max(np.abs(hessian@translations))),
                raw_rotation_residual_hartree_per_bohr=float(np.max(np.abs(hessian@rotations))),
                hessian_seconds_including_compilation=seconds, hessian_column_seconds=column_seconds,
                source_input=str(folder(xc)/"geometry.json"))
    save(xc, "hessian.json", data)
    return data


def spectra(xc):
    path = folder(xc)/"hessian.json"
    data = json.loads(path.read_text())
    energy = molecular_energy(xc, data["coordinates_bohr"])
    return calculate_spectrum(energy, data, folder(xc), source_input=path)


def run(xc):
    optimize(xc)
    harmonic(xc)
    return spectra(xc)


if __name__ == "__main__":
    run(XC)

# Measured on jy_gpu_1/c20, CPU float64, Cartesian 6-31G*, level 3, 2026-10-15.
# PBE0: E=-76.3256740977825 Eh; max nuclear gradient=5.316e-9 Eh/Bohr.
# H2O frequencies / cm^-1: [1722.625337, 3799.703345, 3925.064132]
# IR / km mol^-1: [81.96686731, 4.31034960, 29.53086978]
# Raman / Angstrom^4 amu^-1: [7.34429944, 75.21781057, 37.48955308]
# B3LYP (VWN-RPA): E=-76.40895384515667 Eh; max gradient=6.050e-9.
# H2O frequencies / cm^-1: [1713.059491, 3727.047499, 3849.025367]
# IR: [75.78588709, 1.69215106, 19.38562122]
# Raman: [7.97183806, 78.84768139, 39.08407321]
# Original WB97XD+CHG: E=-76.38707313719108 Eh; max gradient=8.299e-9.
# H2O frequencies / cm^-1: [1721.958912, 3829.104708, 3955.984306]
# IR: [86.53866466, 5.06518008, 31.62826714]
# Raman: [7.15964254, 74.51332169, 37.18351273]
# Nine-column Hessian seconds: PBE0 547.807; B3LYP 649.131; WB97XD 996.851.
# Spectral AD + both mode differences: 1633.565 / 1921.625 / 2800.513 s.
# These scopes include compilation, exclude seed SCF, and have concurrent load.
# WB97XD reused saved geometry/Hessian in a separate bounded spectra() job.
# Raw rotation residuals: 2.468e-5 / 1.735e-5 / 5.263e-4 Eh/Bohr.
# All are reproduced independently; internal projection is not grid convergence.
