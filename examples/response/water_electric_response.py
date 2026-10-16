"""R01/R02: water dipole and static polarizability from E(R,F).

Run the matching HF/PBE geometry optimization first. Geometry is fixed for
the electric-field derivatives. Electrons and nuclei use the same origin.
"""
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import dft, gto, scf, dfpt
from gradscf_tools.molecular_response import MolecularEnergy


METHODS = ("HF", "PBE")
RESULTS = Path("reproducibility/ad_showcase")
OUTPUTS = {"HF": RESULTS / "R01_R02_water_electric",
           "PBE": RESULTS / "R01_R02_water_pbe_electric"}


def run(method, output=None):
    """Run a method at its own optimized geometry; output may isolate a check."""
    method = method.upper()
    if method not in METHODS:
        raise ValueError("Choose HF or PBE.")
    tag = "rhf" if method == "HF" else "pbe"
    geometry = json.loads((RESULTS / f"G04_water_{tag}_optimization/gradscf.json").read_text())
    mol = gto.M(atom=list(zip(geometry["symbols"], geometry["coordinates_bohr"])),
                basis=geometry["basis"], unit="Bohr", cart=True)
    if method == "HF":
        mf = scf.RHF(mol, conv_tol=1e-12, conv_tol_density=1e-11,
                     integral_backend="native").run()
    else:
        mf = dft.RKS(mol, xc="pbe", grids_level=geometry["grid_level"],
                     conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                     max_cycle=150, integral_backend="native").run()
    energy = MolecularEnergy(mf)
    print(f"{method}: compiling complete dipole and polarizability response", flush=True)
    coordinates, field = energy.coordinates, jnp.zeros(3)
    dipole = jax.jit(lambda r, f: -jax.grad(energy, argnums=1)(r, f))
    polarizability = jax.jit(lambda r, f: -jax.hessian(energy, argnums=1)(r, f))
    start = perf_counter()
    mu, alpha = dipole(coordinates, field), polarizability(coordinates, field)
    alpha.block_until_ready()
    cold = perf_counter()-start
    start = perf_counter()
    polarizability(coordinates, field).block_until_ready()
    warm = perf_counter()-start
    if not np.isfinite(mu).all() or not np.isfinite(alpha).all():
        raise RuntimeError("The electronic field response is nonfinite.")
    np.testing.assert_allclose(alpha, alpha.T, atol=1e-9, rtol=0)

    # An independent response construction uses the existing DFPT facade.
    reference = mf.to_reference()
    problem = dfpt.DFPT(mf)
    alpha_dfpt = []
    for operator in reference.dipole_integrals:
        response = problem.response(potential=operator)
        assert bool(response.converged)
        alpha_dfpt.append(-np.einsum("pq,xqp->x", response.density,
                                     reference.dipole_integrals))
    alpha_dfpt = np.asarray(alpha_dfpt).T
    np.testing.assert_allclose(alpha, alpha_dfpt, atol=1e-7, rtol=0)

    checks = []
    for step in (1e-3, 3e-4, 1e-4):
        numerical = np.column_stack([
            (np.asarray(dipole(coordinates, field+step*axis))
             - np.asarray(dipole(coordinates, field-step*axis))) / (2*step)
            for axis in jnp.eye(3)])
        if not np.isfinite(numerical).all():
            raise RuntimeError(f"A finite-field solve failed at field step {step:.1e}.")
        error = float(np.max(np.abs(numerical-alpha)))
        checks.append(dict(field_step_au=step, max_alpha_error=error))
        print(f"field step={step:.1e} au; max alpha error={error:.3e} Bohr^3", flush=True)
    np.testing.assert_allclose(numerical, alpha, atol=2e-5, rtol=0)

    output = OUTPUTS[method] if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = dict(method="RHF" if method == "HF" else "RKS", xc=method, basis=geometry["basis"], cart=True, symbols=geometry["symbols"],
                coordinates_bohr=geometry["coordinates_bohr"], dipole_e_bohr=np.asarray(mu).tolist(),
                polarizability_bohr3=np.asarray(alpha).tolist(), alpha_dfpt_bohr3=alpha_dfpt.tolist(),
                finite_field_checks=checks, cold_dipole_and_alpha_seconds=cold,
                warm_alpha_seconds=warm, origin_bohr=[0., 0., 0.],
                electronic_response="implicit", implicit_tolerance=1e-10,
                orbital_gradient_tolerance=1e-9)
    data.update(grid_level=geometry.get("grid_level"), grid_response=method == "PBE",
                source_input=str(RESULTS / f"G04_water_{tag}_optimization/gradscf.json"),
                raw_rotation_residual_hartree_per_bohr=geometry.get("raw_rotation_residual_hartree_per_bohr"),
                jax_version=jax.__version__, numpy_version=np.__version__,
                python_version=platform.python_version(), hostname=platform.node(),
                devices=[str(device) for device in jax.devices()], dtype="float64",
                runtime_source_hashes={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                      for name in ("src/gradscf_tools/molecular_response.py",
                                                   "src/gradscf/scf/orbital_optimization.py",
                                                   "src/gradscf/scf/uks.py",
                                               "src/gradscf/scf/rsh.py",
                                               "src/gradscf/integrals/grids/ao.py") if Path(name).exists()})
    (output / "gradscf.json").write_text(json.dumps(data, indent=2) + "\n")
    print("Dipole / (e Bohr):", np.asarray(mu))
    print("Static polarizability / Bohr^3:\n", np.asarray(alpha))
    print(f"Cold dipole+alpha / warm alpha: {cold:.3f} / {warm:.3f} s")
    return data


if __name__ == "__main__":
    for method in METHODS:
        run(method)


# RHF measured on jy_gpu_1, CPU float64, 2026-10-15:
# Dipole / (e Bohr): [0.00000000 0.00000000 0.86511021]
# Static polarizability / Bohr^3 (off-diagonal components below 1e-12):
# [[2.77839338 0.00000000 0.00000000]
#  [0.00000000 6.67847834 0.00000000]
#  [0.00000000 0.00000000 4.80758276]]
# field steps 1e-3 / 3e-4 / 1e-4 au: max alpha errors 3.320e-5 / 2.988e-6 / 3.320e-7
# Cold dipole+alpha / warm alpha: 41.347 / 2.465 s

# PBE measured on jy_gpu_1, CPU float64, 2026-10-15:
# Dipole / (e Bohr): [0.00000000 0.00000000 0.81370544]
# Static polarizability diagonal / Bohr^3: [2.89070164 7.56564482 5.66545562]
# field steps 1e-3 / 3e-4 / 1e-4 au: max alpha errors 3.468e-5 / 3.121e-6 / 3.468e-7
# Cold dipole+alpha / warm alpha: 457.727 / 153.619 s
