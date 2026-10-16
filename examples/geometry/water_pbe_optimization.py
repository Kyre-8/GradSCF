"""G04: optimize water with PBE, a moving level-3 grid and native AD forces.

Run from the repository root on jy_gpu_1. SciPy BFGS optimizes all nine
Cartesian coordinates. The shared MolecularEnergy adapter rebuilds every
moving integral/grid input and uses implicit response of the relaxed orbitals.
"""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import jax
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

from gradscf import dft, gto
from gradscf_tools.molecular_response import MolecularEnergy


grid_level = 3
mol = gto.M(atom="O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11",
            basis="6-31g*", unit="Bohr", cart=True)
seed_start = perf_counter()
mf = dft.RKS(mol, xc="pbe", grids_level=grid_level, max_cycle=150,
             conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
             integral_backend="native").run()
seed_seconds = perf_counter() - seed_start
energy = MolecularEnergy(mf)
value_and_gradient = jax.jit(jax.value_and_grad(energy))
history = []


def objective(flat_coordinates):
    coordinates = flat_coordinates.reshape(-1, 3)
    start = perf_counter()
    value, derivative = value_and_gradient(coordinates)
    value, derivative = float(value), np.asarray(derivative)
    if not np.isfinite(value) or not np.isfinite(derivative).all():
        raise RuntimeError("A PBE electronic solve or its implicit gradient failed.")
    history.append(dict(energy_hartree=value, coordinates_bohr=coordinates.tolist(),
                        max_gradient=float(np.max(np.abs(derivative))),
                        evaluation_seconds=perf_counter()-start))
    print(f"Energy/gradient call {len(history):2d}: E={value:.12f} Eh, "
          f"max |gradient|={history[-1]['max_gradient']:.3e} Eh/Bohr", flush=True)
    return value, derivative.ravel()


start = perf_counter()
minimum = minimize(objective, np.asarray(energy.coordinates).ravel(), jac=True,
                   method="BFGS", options={"gtol": 1e-7, "maxiter": 80})
coordinates = minimum.x.reshape(-1, 3)
value, gradient = value_and_gradient(coordinates)
gradient = np.asarray(gradient)
if not np.isfinite(gradient).all() or np.max(np.abs(gradient)) > 1e-7:
    raise RuntimeError(f"PBE geometry did not converge: {minimum.message}")
np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-8, rtol=0)
np.testing.assert_allclose(np.cross(coordinates, gradient).sum(axis=0),
                           0., atol=1e-8, rtol=0)
elapsed = perf_counter() - start
bonds = coordinates[1:] - coordinates[0]
lengths = np.linalg.norm(bonds, axis=1)
angle = np.degrees(np.arccos(np.dot(*bonds) / np.prod(lengths)))
root = Path(__file__).resolve().parents[2]
revision_file = root / "SOURCE_REVISION"
revision = (revision_file.read_text().strip() if revision_file.exists() else
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=root).strip())
output = Path("reproducibility/ad_showcase/G04_water_pbe_optimization")
output.mkdir(parents=True, exist_ok=True)
data = dict(
    method="RKS", xc="PBE", basis=mol.basis, cart=True, symbols=["O", "H", "H"],
    coordinates_bohr=coordinates.tolist(), energy_hartree=float(value),
    gradient_hartree_per_bohr=gradient.tolist(), bond_lengths_bohr=lengths.tolist(),
    angle_degrees=float(angle), optimizer_success=bool(minimum.success),
    optimizer_message=str(minimum.message), iterations=int(minimum.nit),
    gradient_tolerance=1e-7, grid_level=grid_level, grid_response=True,
    integral_backend="native", grid_ao_backend="jax", orbital_gradient_tolerance=1e-9,
    implicit_tolerance=1e-10, elapsed_seconds=elapsed, seed_scf_seconds=seed_seconds,
    history=history, dtype="float64", jax_version=jax.__version__,
    numpy_version=np.__version__, python_version=platform.python_version(),
    hostname=platform.node(), devices=[str(device) for device in jax.devices()],
    source_revision=revision,
    molecular_energy_sha256=hashlib.sha256(
        (root / "src/gradscf_tools/molecular_response.py").read_bytes()).hexdigest(),
)
(output / "gradscf.json").write_text(json.dumps(data, indent=2) + "\n")
print("Bond lengths / Bohr:", lengths)
print(f"HOH angle / degree: {angle:.8f}; optimizer: {minimum.message}")
print(f"Optimization and final gradient check: {elapsed:.3f} s")
print("Stationary geometry accepted; water_pbe_hessian.py checks the internal curvatures.")

# Measured on jy_gpu_1 (CPU, float64, 2026-10-15):
# BFGS: 8 iterations, 9 objective calls; optimization terminated successfully.
# E = -76.322499095446 Eh; max |gradient| = 4.778e-08 Eh/Bohr
# Bond lengths / Bohr: [1.84580194 1.84580194]
# HOH angle / degree: 102.77964569
# Optimization and final gradient check: 403.268 s; seed SCF: 9.848 s.
# Hessian positivity is checked separately in water_pbe_hessian.py.
