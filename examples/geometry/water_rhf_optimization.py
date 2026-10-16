"""G04: optimize water/Cartesian 6-31G* with stationary RHF AD gradients.

Run from the repository root on jy_gpu_1. No external electronic-structure
program supplies energies or forces. BFGS optimizes all nine Cartesian
coordinates; translational/rotational invariance comes from the energy.
"""
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

from gradscf_tools.molecular_ep import NativeRHF


engine = NativeRHF("O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11", basis="6-31g*")
history = []


def objective(flat_coordinates):
    coordinates = flat_coordinates.reshape(-1, 3)
    state, gradient = engine.evaluate(coordinates)
    energy = float(state.total_energy)
    history.append(dict(energy_hartree=energy, coordinates_bohr=coordinates.tolist(),
                        max_gradient=float(np.max(np.abs(gradient)))))
    print(f"SCF call {len(history):2d}: E={energy:.12f} Eh, "
          f"max |gradient|={history[-1]['max_gradient']:.3e} Eh/Bohr", flush=True)
    return energy, gradient.ravel()


start = perf_counter()
minimum = minimize(objective, engine.coordinates.ravel(), jac=True, method="BFGS",
                   options={"gtol": 1e-7, "maxiter": 80})
coordinates = minimum.x.reshape(-1, 3)
reference, gradient = engine.evaluate(coordinates)
if np.max(np.abs(gradient)) > 1e-7:
    raise RuntimeError(f"Geometry did not converge: {minimum.message}")
np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-8, rtol=0)
np.testing.assert_allclose(np.cross(coordinates, gradient).sum(axis=0), 0., atol=1e-8, rtol=0)
bonds = coordinates[1:] - coordinates[0]
lengths = np.linalg.norm(bonds, axis=1)
angle = np.degrees(np.arccos(np.dot(*bonds) / np.prod(lengths)))
output = Path("reproducibility/ad_showcase/G04_water_rhf_optimization")
output.mkdir(parents=True, exist_ok=True)
data = dict(method="RHF", basis="6-31g*", cart=True, symbols=["O", "H", "H"],
            coordinates_bohr=coordinates.tolist(), energy_hartree=float(reference.total_energy),
            gradient_hartree_per_bohr=gradient.tolist(), bond_lengths_bohr=lengths.tolist(),
            angle_degrees=float(angle), optimizer_success=bool(minimum.success),
            optimizer_message=str(minimum.message), iterations=int(minimum.nit),
            gradient_tolerance=1e-7, elapsed_seconds=perf_counter()-start,
            history=history)
(output / "gradscf.json").write_text(json.dumps(data, indent=2) + "\n")
print("Bond lengths / Bohr:", lengths)
print(f"HOH angle / degree: {angle:.8f}")
print("Stationary geometry accepted; Hessian stability is checked by water_rhf_hessian.py.")

# jy_gpu_1, CPU float64, 2026-10-15:
# BFGS: 7 iterations, 8 SCF evaluations; optimization terminated successfully.
# E = -76.010746508460 Eh; max |gradient| = 4.290e-08 Eh/Bohr
# Bond lengths / Bohr: [1.79017327 1.79017327]
# HOH angle / degree: 105.49974398
