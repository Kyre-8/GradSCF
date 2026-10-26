"""G05/V04: relaxed RHF Hessian, normal modes and H2O/D2O isotope shifts.

First run water_rhf_optimization.py. Hessian columns are analytic HVPs from
native second integrals and implicit SCF response. Displaced gradients only
validate the result. Isotopes change masses, keeping the electronic Hessian.
"""
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf_tools.molecular_ep import NativeRHF
from gradscf_tools.vibrations import normal_modes, AMU_TO_ELECTRON_MASS, HARTREE_TO_WAVENUMBER


geometry = json.loads(Path("reproducibility/ad_showcase/G04_water_rhf_optimization/gradscf.json").read_text())
coordinates = np.array(geometry["coordinates_bohr"])
engine = NativeRHF(list(zip(geometry["symbols"], coordinates)), basis=geometry["basis"])
state, gradient = engine.evaluate(coordinates)
assert np.max(np.abs(gradient)) < 1e-7
response = engine.response(coordinates, state)
start = perf_counter()
columns = []
for direction in np.eye(coordinates.size).reshape(-1, *coordinates.shape):
    product, _ = response(direction)
    columns.append(np.asarray(product).ravel())
hessian = np.column_stack(columns)
seconds = perf_counter() - start

# A non-axis-aligned direction also tests the HVP without a dense Jacobian.
direction = np.random.default_rng(13).normal(size=coordinates.shape)
direction /= np.linalg.norm(direction)
product, _ = response(direction)
np.testing.assert_allclose(np.asarray(product).ravel(), hessian @ direction.ravel(),
                           atol=1e-8, rtol=0)
checks = []
for step in (1e-3, 3e-4, 1e-4):
    _, plus = engine.evaluate(coordinates + step*direction, init_density=state.density_matrix)
    _, minus = engine.evaluate(coordinates - step*direction, init_density=state.density_matrix)
    if not np.isfinite(plus).all() or not np.isfinite(minus).all():
        raise RuntimeError(f"A displaced gradient failed at step {step:.1e} Bohr.")
    error = float(np.max(np.abs((plus-minus)/(2*step) - product)))
    checks.append(dict(step_bohr=step, max_hvp_error=error))
    print(f"step={step:.1e} Bohr; HVP error={error:.3e} Eh/Bohr^2", flush=True)
assert checks[-1]["max_hvp_error"] < 2e-6

masses = {"H2O": [15.99491461957, 1.00782503223, 1.00782503223],
          "D2O": [15.99491461957, 2.01410177812, 2.01410177812]}
isotopes = {}
for label, isotope_masses in masses.items():
    modes = normal_modes(hessian, coordinates, isotope_masses)
    assert modes.rigid_rank == 6 and np.all(modes.frequencies_cm1 > 0)
    assert modes.rigid_residual < 1e-9
    isotopes[label] = dict(masses_amu=isotope_masses,
                          frequencies_cm1=modes.frequencies_cm1.tolist(),
                          displacements=modes.displacements.tolist(),
                          rigid_residual=modes.rigid_residual)
    print(label, "harmonic frequencies / cm^-1:", modes.frequencies_cm1)

output = Path("reproducibility/ad_showcase/G05_water_rhf_hessian")
output.mkdir(parents=True, exist_ok=True)
data = dict(method="RHF", basis=geometry["basis"], cart=True, symbols=geometry["symbols"],
            coordinates_bohr=coordinates.tolist(), hessian_hartree_per_bohr2=hessian.tolist(),
            raw_hessian_asymmetry=float(np.max(np.abs(hessian-hessian.T))),
            finite_difference_checks=checks, isotopes=isotopes,
            hessian_seconds_including_compilation=seconds,
            amu_to_electron_mass=AMU_TO_ELECTRON_MASS,
            hartree_to_wavenumber=HARTREE_TO_WAVENUMBER)
(output / "gradscf.json").write_text(json.dumps(data, indent=2) + "\n")

# jy_gpu_1, CPU float64, 2026-10-15:
# step=1.0e-03 Bohr; HVP error=4.002e-07 Eh/Bohr^2
# step=3.0e-04 Bohr; HVP error=4.510e-08 Eh/Bohr^2
# step=1.0e-04 Bohr; HVP error=4.888e-08 Eh/Bohr^2
# H2O harmonic frequencies / cm^-1: [1826.55340970 4070.45989209 4188.70487667]
# D2O harmonic frequencies / cm^-1: [1336.45825428 2935.12348091 3070.43066224]
