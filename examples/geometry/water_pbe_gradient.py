"""Water/6-31G*: relaxed PBE gradient with a moving molecular grid.

Run from the repository root with PYTHONPATH=src. Coordinates are in Bohr;
energy is in Hartree and gradients are in Hartree/Bohr. The geometry is not
optimized. JAX differentiates native AO integrals, basis centers, atom-centered
grid coordinates, Becke weights and AO grid values through the public input
builder. The shared orbital solver supplies implicit electronic response.
"""
from dataclasses import replace
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import dft, gto, scf
from gradscf.integrals.grids import build_molecular_grid_from_spec
from gradscf.scf.inputs import build_rks_integral_inputs


grid_level = 3
mol = gto.M(
    atom="O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11",
    basis="6-31g*", unit="Bohr", cart=True,
)
mf = dft.RKS(mol, xc="pbe", grids_level=grid_level,
             conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
             max_cycle=150, integral_backend="native").run()
if not mf.converged:
    raise RuntimeError("The initial PBE calculation did not converge.")
spec = mol.to_spec()
coordinates = jnp.asarray(spec.coords_bohr)
coefficients = jnp.asarray(mf.mo_coeff)
occupations = np.stack([np.asarray(mf.mo_occ) / 2] * 2)
grid_size = len(build_molecular_grid_from_spec(spec, level=grid_level)[1])


def energy(coords):
    """Relax the orbitals after rebuilding all geometry-dependent inputs."""
    inputs = build_rks_integral_inputs(
        atom=replace(spec, coords_bohr=coords), basis=mol.basis,
        xc_spec="pbe", grids_level=grid_level, cart=True,
        integral_backend="native", grid_ao_backend="jax",
        include_dipole_integrals=False,
    )
    result = scf.minimize_roks_from_integrals(
        overlap=inputs.overlap, hcore=inputs.hcore, eri=inputs.eri_pair_matrix,
        nuclear_repulsion=inputs.nuclear_repulsion,
        ao=inputs.ao, ao_deriv1=inputs.ao_deriv1,
        grid_weights=inputs.grid_weights, xc_spec="pbe",
        mo_coeff=coefficients, mo_occ=occupations, orthonormalize_initial=True,
        max_iterations=300, gradient_tolerance=1e-9,
        differentiation=scf.SCFDifferentiationConfig(
            mode="implicit", tolerance=1e-10, max_iter=100, restart=40,
        ),
    )
    return jnp.where(result.stationary, result.total_energy, jnp.nan)


value_and_gradient = jax.jit(jax.value_and_grad(energy))
start = perf_counter()
value, gradient = value_and_gradient(coordinates)
gradient.block_until_ready()
cold_seconds = perf_counter() - start
start = perf_counter()
value, gradient = value_and_gradient(coordinates)
gradient.block_until_ready()
warm_seconds = perf_counter() - start
np.testing.assert_allclose(value, mf.e_tot, atol=1e-9, rtol=0)
assert np.isfinite(np.asarray(gradient)).all()
print(f"E = {float(value):.12f} Eh; grid points = {grid_size}", flush=True)
print("Gradient / (Eh/Bohr):\n", np.asarray(gradient), flush=True)

# Every central-difference energy rebuilds the moving grid and relaxes orbitals.
forward = jax.jit(energy)
directions = np.eye(coordinates.size).reshape((-1,) + coordinates.shape)
steps = (1e-3, 3e-4, 1e-4)
finite_differences = []
for step in steps:
    derivative = np.array([
        (float(forward(coordinates + step * direction))
         - float(forward(coordinates - step * direction))) / (2 * step)
        for direction in directions
    ]).reshape(coordinates.shape)
    if not np.isfinite(derivative).all():
        raise RuntimeError(f"A displaced PBE solve failed at step {step:.1e} Bohr.")
    finite_differences.append(derivative)
    error = np.max(np.abs(derivative - np.asarray(gradient)))
    print(f"step = {step:.1e} Bohr; max AD-FD error = {error:.3e} Eh/Bohr",
          flush=True)
np.testing.assert_allclose(gradient, finite_differences[-1], atol=1e-6, rtol=0)
force = -np.asarray(gradient)
net_force = force.sum(axis=0)
torque = np.cross(np.asarray(coordinates), force).sum(axis=0)
np.testing.assert_allclose(net_force, 0, atol=1e-8, rtol=0)
# This symmetric geometry also suppresses the finite angular-grid torque.
np.testing.assert_allclose(torque, 0, atol=1e-8, rtol=0)

output = Path("reproducibility/ad_showcase/G01_water_pbe_gradient")
output.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
revision_file = root / "SOURCE_REVISION"
revision = (revision_file.read_text().strip() if revision_file.exists() else
            subprocess.check_output(["git", "rev-parse", "HEAD"],
                                    text=True, cwd=root).strip())
data = dict(
    method="RKS", xc="PBE", basis=mol.basis, cart=True, symbols=["O", "H", "H"],
    coordinates_bohr=np.asarray(coordinates).tolist(), energy_hartree=float(value),
    gradient_hartree_per_bohr=np.asarray(gradient).tolist(),
    finite_difference_steps_bohr=list(steps),
    finite_difference_gradients=np.asarray(finite_differences).tolist(),
    net_force=net_force.tolist(), torque=torque.tolist(),
    grid_level=grid_level, grid_points=grid_size,
    grid_radial="Treutler-Ahlrichs", grid_angular="Lebedev / NWChem pruning",
    grid_partition="original Becke / Treutler radii adjustment",
    grid_response=True, grid_ao_backend="jax",
    cold_value_and_gradient_seconds=cold_seconds,
    warm_value_and_gradient_seconds=warm_seconds,
    scf_energy_tolerance=1e-12, scf_density_tolerance=1e-11,
    scf_gradient_tolerance=1e-10, orbital_gradient_tolerance=1e-9,
    implicit_tolerance=1e-10, dtype="float64", integral_backend="native",
    jax_version=jax.__version__, numpy_version=np.__version__,
    python_version=platform.python_version(), hostname=platform.node(),
    machine=platform.machine(), devices=[str(device) for device in jax.devices()],
    source_revision=revision,
)
(output / "gradscf.json").write_text(json.dumps(data, indent=2) + "\n")
print(f"Cold / warm value_and_grad: {cold_seconds:.3f} / {warm_seconds:.3f} s")

# Measured on jy_gpu_1 (CPU, float64, 2026-10-15):
# E = -76.321861744206 Eh; grid points = 33698
# Gradient / (Eh/Bohr):
# [[ 0.00000000  0.00000000  0.02346375]
#  [ 0.00000000 -0.01283828 -0.01173188]
#  [ 0.00000000  0.01283828 -0.01173188]]
# step = 1.0e-03 Bohr; max AD-FD error = 1.369e-07 Eh/Bohr
# step = 3.0e-04 Bohr; max AD-FD error = 1.163e-08 Eh/Bohr
# step = 1.0e-04 Bohr; max AD-FD error = 7.916e-09 Eh/Bohr
# Cold / warm value_and_grad: 104.170 / 20.958 s
# These timings include both energy and gradient; they are not backward-only.
