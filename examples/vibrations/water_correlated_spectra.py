"""V05: water MP2/CCSD optimization, harmonic Hessian and static IR/Raman.

Each stage is an ordinary callable for a separate bounded job. The total
energy retains the HF reference response and correlated amplitude response.
The correlated dipole is -dE_total/dF, not the HF stationary dipole.
"""
import hashlib
import json
import platform
import runpy
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf
from gradscf_tools.correlated_response import CorrelatedEnergy
from gradscf_tools.vibrations import normal_modes, AMU_TO_ELECTRON_MASS


METHOD = "ccsd"
SYMBOLS = ["O", "H", "H"]
INITIAL = np.array([[0., 0., 0.], [0., 1.43, 1.11], [0., -1.43, 1.11]])
MASSES = {"H2O": [15.99491461957, 1.00782503223, 1.00782503223],
          "D2O": [15.99491461957, 2.01410177812, 2.01410177812]}


def folder(method):
    if method not in ("mp2", "ccsd"):
        raise ValueError("Choose mp2 or ccsd.")
    return Path("reproducibility/ad_showcase") / f"V05_water_{method}_spectra"


def molecular_energy(method, coordinates):
    mol = gto.M(atom=list(zip(SYMBOLS, coordinates)), basis="6-31g*", unit="Bohr", cart=True)
    mf = scf.RHF(mol, conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                 max_cycle=200, integral_backend="native").run()
    return CorrelatedEnergy(mf, method)


def save(method, name, values):
    output = folder(method)
    output.mkdir(parents=True, exist_ok=True)
    source = ("src/gradscf_tools/correlated_response.py", "src/gradscf_tools/molecular_response.py",
              "src/gradscf/scf/orbital_optimization.py", "src/gradscf/solvers/nonlinear/iterate.py",
              "src/gradscf/solvers/nonlinear/root.py", "src/gradscf/solvers/eigen/response.py")
    values.update(method=method.upper(), label=method.upper(), xc="HF", basis="6-31g*", cart=True,
                  symbols=SYMBOLS, grid_level=None, grid_response=False, dispersion="none",
                  frozen_orbitals=None, reference="RHF", integral_backend="native",
                  dipole_expression="-dE_total/dF; relaxed HF orbitals and correlated amplitudes",
                  hf_gradient_tolerance=1e-10, implicit_tolerance=1e-10,
                  cc_energy_tolerance=1e-11 if method == "ccsd" else None,
                  cc_residual_tolerance=1e-10 if method == "ccsd" else None,
                  hostname=platform.node(), python_version=platform.python_version(),
                  jax_version=jax.__version__, dtype="float64", devices=[str(d) for d in jax.devices()],
                  runtime_source_hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source})
    (output/name).write_text(json.dumps(values, indent=2)+"\n")
    return values


def optimize(method):
    energy = molecular_energy(method, INITIAL)
    evaluate = jax.jit(jax.value_and_grad(energy))
    history = []

    def objective(flat):
        coordinates = flat.reshape(3, 3)
        value, gradient = evaluate(coordinates)
        value, gradient = float(value), np.asarray(gradient)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise RuntimeError("HF, correlation or response did not converge.")
        history.append(dict(energy_hartree=value, coordinates_bohr=coordinates.tolist(),
                            max_gradient=float(np.max(np.abs(gradient)))))
        print(method, "optimization", len(history), value, history[-1]["max_gradient"], flush=True)
        return value, gradient.ravel()

    start = perf_counter()
    result = minimize(objective, INITIAL.ravel(), jac=True, method="BFGS",
                      options={"gtol": 1e-7, "maxiter": 80})
    coordinates = result.x.reshape(3, 3)
    value, gradient = evaluate(coordinates)
    gradient = np.asarray(gradient)
    assert np.isfinite(gradient).all() and np.max(np.abs(gradient)) < 1e-7
    bonds = coordinates[1:]-coordinates[0]
    lengths = np.linalg.norm(bonds, axis=1)
    return save(method, "geometry.json", dict(
        coordinates_bohr=coordinates.tolist(), energy_hartree=float(value),
        gradient_hartree_per_bohr=gradient.tolist(), bond_lengths_bohr=lengths.tolist(),
        angle_degrees=float(np.degrees(np.arccos(np.dot(*bonds)/np.prod(lengths)))),
        history=history, iterations=int(result.nit), optimizer_success=bool(result.success),
        optimizer_message=str(result.message), gradient_tolerance=1e-7,
        elapsed_seconds=perf_counter()-start))


def harmonic(method):
    geometry = json.loads((folder(method)/"geometry.json").read_text())
    energy = molecular_energy(method, geometry["coordinates_bohr"])
    coordinates = energy.coordinates
    gradient = jax.jit(jax.grad(energy))
    derivative = np.asarray(gradient(coordinates))
    assert np.isfinite(derivative).all() and np.max(np.abs(derivative)) < 1e-7
    hvp = jax.jit(lambda r, v: jax.jvp(jax.grad(energy), (r,), (v,))[1])
    start = perf_counter()
    columns, seconds = [], []
    for index, direction in enumerate(np.eye(9).reshape(9, 3, 3)):
        tick = perf_counter()
        column = np.asarray(hvp(coordinates, jnp.asarray(direction)))
        assert np.isfinite(column).all()
        columns.append(column.ravel())
        seconds.append(perf_counter()-tick)
        print(method, "Hessian column", index+1, "seconds", seconds[-1], flush=True)
    hessian = np.column_stack(columns)
    elapsed = perf_counter()-start
    np.testing.assert_allclose(hessian, hessian.T, atol=2e-6, rtol=0)
    direction = np.random.default_rng(13).normal(size=(3, 3))
    direction /= np.linalg.norm(direction)
    product = np.asarray(hvp(coordinates, jnp.asarray(direction)))
    checks = []
    for step in (1e-3, 3e-4, 1e-4):
        finite = (gradient(coordinates+step*direction)-gradient(coordinates-step*direction))/(2*step)
        checks.append(dict(step_bohr=step, max_hvp_error=float(np.max(np.abs(product-finite)))))
    assert checks[-1]["max_hvp_error"] < 2e-6
    translations = np.tile(np.eye(3), (3, 1))
    centered = np.asarray(coordinates)-np.mean(coordinates, axis=0)
    rotations = np.column_stack([np.cross(axis, centered).ravel() for axis in np.eye(3)])
    np.testing.assert_allclose(hessian@translations, 0., atol=1e-8, rtol=0)
    isotopes = {}
    for label, masses in MASSES.items():
        modes = normal_modes(hessian, coordinates, masses)
        assert modes.rigid_rank == 6 and np.all(modes.frequencies_cm1 > 0)
        vectors = modes.displacements.reshape(9, 3)
        mass = np.repeat(np.asarray(masses)*AMU_TO_ELECTRON_MASS, 3)
        np.testing.assert_allclose(vectors.T@(mass[:, None]*vectors), np.eye(3), atol=1e-10, rtol=0)
        isotopes[label] = dict(masses_amu=masses, frequencies_cm1=modes.frequencies_cm1.tolist(),
                              displacements=modes.displacements.tolist(), rigid_residual=modes.rigid_residual)
        print(method, label, modes.frequencies_cm1, flush=True)
    return save(method, "hessian.json", dict(
        coordinates_bohr=np.asarray(coordinates).tolist(), energy_hartree=geometry["energy_hartree"],
        gradient_hartree_per_bohr=derivative.tolist(), hessian_hartree_per_bohr2=hessian.tolist(),
        isotopes=isotopes, finite_difference_direction=direction.tolist(), finite_difference_checks=checks,
        raw_hessian_asymmetry=float(np.max(np.abs(hessian-hessian.T))),
        raw_translation_residual_hartree_per_bohr2=float(np.max(np.abs(hessian@translations))),
        raw_rotation_residual_hartree_per_bohr=float(np.max(np.abs(hessian@rotations))),
        hessian_seconds_including_compilation=elapsed, hessian_column_seconds=seconds,
        source_input=str(folder(method)/"geometry.json")))


def spectra(method):
    path = folder(method)/"hessian.json"
    harmonic_data = json.loads(path.read_text())
    energy = molecular_energy(method, harmonic_data["coordinates_bohr"])
    calculate = runpy.run_path(str(Path(__file__).with_name("water_ir_raman.py")))["calculate_spectrum"]
    return calculate(energy, harmonic_data, folder(method), source_input=path)


if __name__ == "__main__":
    optimize(METHOD)
    harmonic(METHOD)
    spectra(METHOD)

# CCSD measured on jy_gpu_1, CPU float64, 2026-10-15:
# E = -76.20807348173228 Eh; max geometry gradient = 2.524e-8 Eh/Bohr.
# H2O / cm^-1: [1745.87483002, 3753.89516964, 3880.05170521]
# D2O / cm^-1: [1277.74920454, 2706.17292007, 2842.42528297]
# IR / km mol^-1: [85.86684897, 2.93620326, 24.64488164]
# Static Raman / Angstrom^4 amu^-1: [7.29753722, 77.04678579, 39.44537936]
# Independent full Hessian max error = 5.269e-9 Eh/Bohr^2.
# Independent finite-field dmu/dQ and dalpha/dQ errors = 3.778e-7 / 2.904e-6 au.
# MP2 measured on jy_gpu_1, CPU float64, 2026-10-15:
# E = -76.1992441658066 Eh; max geometry gradient = 2.165e-8 Eh/Bohr.
# H2O / cm^-1: [1735.45521017, 3776.01974678, 3917.52892768]
# D2O / cm^-1: [1270.14510302, 2722.07600328, 2869.82204539]
# IR / km mol^-1: [88.80512678, 5.56438577, 39.03654797]
# Static Raman / Angstrom^4 amu^-1: [7.34727938, 73.01717760, 35.90614060]
# Independent full Hessian max error = 5.238e-9 Eh/Bohr^2.
# Independent finite-field dmu/dQ and dalpha/dQ errors = 3.402e-7 / 3.496e-6 au.
