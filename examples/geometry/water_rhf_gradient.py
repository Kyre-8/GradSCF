"""Water/6-31G*: nuclear gradients through native integrals and implicit RHF.

Run from the repository root with PYTHONPATH=src. Coordinates are in Bohr;
energy is in Hartree and gradients are in Hartree/Bohr. The geometry is
deliberately not optimized, so the forces are nonzero.

The ordinary RHF calculation supplies an initial orbital frame. Each call to
energy solves the moving-geometry problem again, using the shared orbital
solver. Both basis centers and nuclei move. In particular, differentiating
the overlap includes the moving-basis contribution to the force.
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

from gradscf import gto, integrals, scf


mol = gto.M(
    atom="O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11",
    basis="6-31g*", unit="Bohr", cart=True,
)
mf = scf.RHF(mol, conv_tol=1e-12, conv_tol_density=1e-11,
             max_cycle=150, integral_backend="native").run()
if not mf.converged:
    raise RuntimeError("The initial RHF calculation did not converge.")

topology, parameters = integrals.prepare_basis(mol.to_spec(), mol.basis)
plan = integrals.make_plan(topology, backend="native")
coordinates = jnp.asarray(parameters.nuclear_coords)
charges = jnp.asarray(topology.nuclear_charges)
# Map each shell center to its nucleus once, outside the AD calculation.
owners = np.argmin(np.linalg.norm(
    np.asarray(parameters.centers)[:, None] - np.asarray(coordinates)[None],
    axis=-1,
), axis=1)
coefficients = jnp.asarray(mf.mo_coeff)
occupations = np.stack([np.asarray(mf.mo_occ) / 2] * 2)
nao = coefficients.shape[0]


def energy(coords):
    """Relaxed RHF energy with an implicit orbital-response backward rule."""
    moving = replace(parameters, centers=coords[owners], nuclear_coords=coords)
    result = scf.minimize_roks_from_integrals(
        overlap=plan.evaluate("overlap", moving),
        hcore=plan.evaluate("kinetic", moving) + plan.evaluate("nuclear", moving),
        eri=plan.evaluate("eri", moving),
        nuclear_repulsion=scf.nuclear_repulsion_energy(coords, charges),
        mo_coeff=coefficients, mo_occ=occupations, xc_spec="hf",
        ao=jnp.zeros((0, nao)), ao_deriv1=jnp.zeros((4, 0, nao)),
        grid_weights=jnp.zeros(0), orthonormalize_initial=True,
        max_iterations=300, gradient_tolerance=1e-9,
        differentiation=scf.SCFDifferentiationConfig(
            mode="implicit", tolerance=1e-10, max_iter=100, restart=40,
        ),
    )
    # A nonstationary solution must not become a plausible-looking force.
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

# Central differences reoptimize the electronic state at each displacement.
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
        raise RuntimeError(f"A displaced RHF solve failed at step {step:.1e} Bohr.")
    finite_differences.append(derivative)
    error = np.max(np.abs(derivative - np.asarray(gradient)))
    print(f"step = {step:.1e} Bohr; max AD-FD error = {error:.3e} Eh/Bohr",
          flush=True)
np.testing.assert_allclose(gradient, finite_differences[-1], atol=1e-6, rtol=0)
force = -np.asarray(gradient)
net_force = force.sum(axis=0)
torque = np.cross(np.asarray(coordinates), force).sum(axis=0)
np.testing.assert_allclose(net_force, 0, atol=1e-8, rtol=0)
np.testing.assert_allclose(torque, 0, atol=1e-8, rtol=0)

output = Path("reproducibility/ad_showcase/G01_water_rhf_gradient")
output.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
# Exported remote snapshots carry the base revision without a .git directory.
revision_file = root / "SOURCE_REVISION"
revision = (revision_file.read_text().strip() if revision_file.exists() else
            subprocess.check_output(["git", "rev-parse", "HEAD"],
                                    text=True, cwd=root).strip())
data = dict(
    method="RHF", basis=mol.basis, cart=True, symbols=["O", "H", "H"],
    coordinates_bohr=np.asarray(coordinates).tolist(), energy_hartree=float(value),
    gradient_hartree_per_bohr=np.asarray(gradient).tolist(),
    finite_difference_steps_bohr=list(steps),
    finite_difference_gradients=np.asarray(finite_differences).tolist(),
    net_force=net_force.tolist(), torque=torque.tolist(),
    cold_value_and_gradient_seconds=cold_seconds,
    warm_value_and_gradient_seconds=warm_seconds,
    scf_energy_tolerance=1e-12, orbital_gradient_tolerance=1e-9,
    implicit_tolerance=1e-10, dtype="float64", integral_backend="native",
    jax_version=jax.__version__, numpy_version=np.__version__,
    python_version=platform.python_version(), hostname=platform.node(),
    machine=platform.machine(), devices=[str(device) for device in jax.devices()],
    source_revision=revision,
)
(output / "gradscf.json").write_text(json.dumps(data, indent=2) + "\n")
print(f"E = {float(value):.12f} Eh")
print("Gradient / (Eh/Bohr):\n", np.asarray(gradient))
print(f"Cold / warm value_and_grad: {cold_seconds:.3f} / {warm_seconds:.3f} s")

# Measured on jy_gpu_1 (CPU, float64, 2026-10-15):
# step = 1.0e-03 Bohr; max AD-FD error = 1.399e-07 Eh/Bohr
# step = 3.0e-04 Bohr; max AD-FD error = 1.267e-08 Eh/Bohr
# step = 1.0e-04 Bohr; max AD-FD error = 1.747e-09 Eh/Bohr
# E = -76.010492818245 Eh
# Gradient / (Eh/Bohr):
# [[ 0.00000000  0.00000000 -0.01604143]
#  [ 0.00000000  0.00791327  0.00802071]
#  [ 0.00000000 -0.00791327  0.00802071]]
# Cold / warm value_and_grad: 13.767 / 0.593 s
# Timings include both the energy and gradient, not backward alone.
