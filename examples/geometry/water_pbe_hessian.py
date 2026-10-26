"""G05/V04: fully relaxed PBE Hessian and H2O/D2O harmonic isotope shifts.

First run water_pbe_optimization.py. Each Hessian column is a JVP of the
complete AD gradient, including native integrals, moving grid and implicit
orbital response. Gradient differences validate the Hessian; they do not
construct it. Isotopes change masses while retaining the electronic Hessian.
"""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import dft, gto
from gradscf_tools.molecular_response import MolecularEnergy
from gradscf_tools.vibrations import normal_modes, AMU_TO_ELECTRON_MASS, HARTREE_TO_WAVENUMBER


geometry = json.loads(Path(
    "reproducibility/ad_showcase/G04_water_pbe_optimization/gradscf.json").read_text())
coordinates = jnp.asarray(geometry["coordinates_bohr"])
mol = gto.M(atom=list(zip(geometry["symbols"], np.asarray(coordinates))),
            basis=geometry["basis"], unit="Bohr", cart=True)
mf = dft.RKS(mol, xc="pbe", grids_level=geometry["grid_level"], max_cycle=150,
             conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
             integral_backend="native").run()
energy = MolecularEnergy(mf)
gradient = jax.jit(jax.grad(energy))
derivative = np.asarray(gradient(coordinates))
assert np.isfinite(derivative).all() and np.max(np.abs(derivative)) < 1e-7
hvp = jax.jit(lambda r, v: jax.jvp(jax.grad(energy), (r,), (v,))[1])

start = perf_counter()
columns, column_seconds = [], []
for index, direction in enumerate(np.eye(coordinates.size).reshape(-1, *coordinates.shape)):
    column_start = perf_counter()
    column = np.asarray(hvp(coordinates, jnp.asarray(direction)))
    if not np.isfinite(column).all():
        raise RuntimeError(f"PBE Hessian column {index} is nonfinite.")
    columns.append(column.ravel())
    column_seconds.append(perf_counter()-column_start)
    print(f"AD Hessian column {index+1}/9: {column_seconds[-1]:.3f} s", flush=True)
hessian = np.column_stack(columns)
seconds = perf_counter() - start
np.testing.assert_allclose(hessian, hessian.T, atol=2e-6, rtol=0)

# A reproducible mixed Cartesian direction tests the full Hessian action.
direction = np.random.default_rng(13).normal(size=coordinates.shape)
direction /= np.linalg.norm(direction)
product = np.asarray(hvp(coordinates, jnp.asarray(direction)))
np.testing.assert_allclose(product.ravel(), hessian @ direction.ravel(), atol=1e-7, rtol=0)
checks = []
for step in (1e-3, 3e-4, 1e-4):
    plus = np.asarray(gradient(coordinates + step*direction))
    minus = np.asarray(gradient(coordinates - step*direction))
    if not np.isfinite(plus).all() or not np.isfinite(minus).all():
        raise RuntimeError(f"A displaced PBE gradient failed at step {step:.1e} Bohr.")
    error = float(np.max(np.abs((plus-minus)/(2*step) - product)))
    checks.append(dict(step_bohr=step, max_hvp_error=error))
    print(f"step={step:.1e} Bohr; AD-HVP error={error:.3e} Eh/Bohr^2", flush=True)
assert checks[-1]["max_hvp_error"] < 2e-6

# Inspect raw rigid-motion residuals before any normal-mode projection.
translations = np.tile(np.eye(3), (len(coordinates), 1))
translation_residual = float(np.max(np.abs(hessian @ translations)))
np.testing.assert_allclose(hessian @ translations, 0., atol=1e-8, rtol=0)
centered = np.asarray(coordinates) - np.mean(np.asarray(coordinates), axis=0)
rotations = np.column_stack([np.cross(centered, axis).ravel() for axis in np.eye(3)])
rotation_residual = float(np.max(np.abs(hessian @ rotations)))

masses = {"H2O": [15.99491461957, 1.00782503223, 1.00782503223],
          "D2O": [15.99491461957, 2.01410177812, 2.01410177812]}
isotopes = {}
for label, isotope_masses in masses.items():
    modes = normal_modes(hessian, coordinates, isotope_masses)
    assert modes.rigid_rank == 6 and np.all(modes.frequencies_cm1 > 0)
    vectors = modes.displacements.reshape(coordinates.size, -1)
    mass = np.repeat(np.asarray(isotope_masses) * AMU_TO_ELECTRON_MASS, 3)
    np.testing.assert_allclose(vectors.T @ (mass[:, None] * vectors), np.eye(3),
                               atol=1e-10, rtol=0)
    isotopes[label] = dict(masses_amu=isotope_masses,
                          frequencies_cm1=modes.frequencies_cm1.tolist(),
                          displacements=modes.displacements.tolist(),
                          rigid_residual=modes.rigid_residual)
    print(label, "harmonic frequencies / cm^-1:", modes.frequencies_cm1, flush=True)
print(f"Raw translation / rotation residual: {translation_residual:.3e} / "
      f"{rotation_residual:.3e} (Eh/Bohr^2 / Eh/Bohr)")

root = Path(__file__).resolve().parents[2]
output = Path("reproducibility/ad_showcase/G05_water_pbe_hessian")
output.mkdir(parents=True, exist_ok=True)
data = dict(
    method="RKS", xc="PBE", basis=geometry["basis"], cart=True, symbols=geometry["symbols"],
    coordinates_bohr=np.asarray(coordinates).tolist(), energy_hartree=float(mf.e_tot),
    gradient_hartree_per_bohr=derivative.tolist(), grid_level=geometry["grid_level"],
    grid_response=True, hessian_hartree_per_bohr2=hessian.tolist(),
    raw_hessian_asymmetry=float(np.max(np.abs(hessian-hessian.T))),
    raw_translation_residual_hartree_per_bohr2=translation_residual,
    raw_rotation_residual_hartree_per_bohr=rotation_residual,
    finite_difference_direction=direction.tolist(), finite_difference_checks=checks,
    isotopes=isotopes, hessian_seconds_including_compilation=seconds,
    hessian_column_seconds=column_seconds, integral_backend="native", dtype="float64",
    orbital_gradient_tolerance=1e-9, implicit_tolerance=1e-10,
    amu_to_electron_mass=AMU_TO_ELECTRON_MASS, hartree_to_wavenumber=HARTREE_TO_WAVENUMBER,
    jax_version=jax.__version__, numpy_version=np.__version__,
    python_version=platform.python_version(), hostname=platform.node(),
    devices=[str(device) for device in jax.devices()], source_revision=geometry["source_revision"],
    molecular_energy_sha256=hashlib.sha256(
        (root / "src/gradscf_tools/molecular_response.py").read_bytes()).hexdigest(),
    normal_modes_sha256=hashlib.sha256(
        (root / "src/gradscf_tools/vibrations.py").read_bytes()).hexdigest(),
)
(output / "gradscf.json").write_text(json.dumps(data, indent=2) + "\n")

# Measured on jy_gpu_1 (CPU, float64, 2026-10-15):
# step=1.0e-03 Bohr; AD-HVP error=3.297e-07 Eh/Bohr^2
# step=3.0e-04 Bohr; AD-HVP error=2.965e-08 Eh/Bohr^2
# step=1.0e-04 Bohr; AD-HVP error=3.306e-09 Eh/Bohr^2
# H2O harmonic frequencies / cm^-1: [1683.33619806 3625.11502281 3751.50858455]
# D2O harmonic frequencies / cm^-1: [1231.62800328 2614.08086154 2746.75575713]
# Raw translation / rotation residual: 2.469e-12 / 6.433e-05 (Eh/Bohr^2 / Eh/Bohr)
# Nine Hessian columns, including HVP compilation: 530.052 s.
# Finite angular-grid rotation residuals remain in the saved raw Hessian.
