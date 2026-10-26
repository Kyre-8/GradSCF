"""X10 gate: complete AD Hessian at the independently validated X03 S1 minimum.

Run the small H2 higher-response tests first, on the new spectral-response
snapshot. Every column is a JVP of the relaxed CIS total-energy gradient;
displaced gradients are validation only. X03's archived FD curvature is unchanged.
"""
import hashlib
import json
from pathlib import Path
import platform
import resource
from time import perf_counter

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf
from gradscf.solvers import EigenResponseConfig, LinearSolverConfig
from gradscf_tools.excited import CISGeometry
from gradscf_tools.vibrations import normal_modes, HARTREE_TO_WAVENUMBER, AMU_TO_ELECTRON_MASS


input_folder = Path("reproducibility/ad_showcase/X03_formaldehyde_cis_optimization")
native = json.loads((input_folder/"gradscf.json").read_text())
reference = json.loads((input_folder/"pyscf.json").read_text())
native_hash = hashlib.sha256((input_folder/"gradscf.json").read_bytes()).hexdigest()
if not native["complete"] or not reference["passed"] or reference["native_sha256"] != native_hash:
    raise RuntimeError("Independently validate the exact X03 minimum first.")
output = Path("reproducibility/ad_showcase/X10_formaldehyde_cis_emission")
output.mkdir(parents=True, exist_ok=True)
started = perf_counter()
coordinates = np.asarray(native["coordinates_bohr"])
mol = gto.M(atom=list(zip(native["symbols"], coordinates)), basis=native["basis"], unit="Bohr", cart=True)
mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-12,
             conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=180).run()
if not mf.converged:
    raise RuntimeError("Initial RHF failed to converge.")
response = EigenResponseConfig(linear_config=LinearSolverConfig(
    method="direct", max_dense=512, rtol=1e-10, atol=0.))
model = CISGeometry(mf, nroots=6, gradient_tolerance=1e-10, response=response)
response_dimension = model.nocc*(mf.mo_coeff.shape[1]-model.nocc)
if response_dimension > response.linear_config.max_dense:
    raise RuntimeError("This example exceeds its explicit direct-response capacity.")
root = native["selected_root"]
energy, gradient, state = model.total_gradient(coordinates, root)
np.testing.assert_allclose(energy, native["total_excited_energy_hartree"], atol=1e-8, rtol=0)
if np.max(np.abs(gradient)) > 1e-6:
    raise RuntimeError("The imported minimum is not stationary with this source.")
anchor = model.anchor(coordinates, state, root)
columns, timings = [], []
for index, direction in enumerate(np.eye(coordinates.size).reshape(-1, *coordinates.shape)):
    start = perf_counter()
    columns.append(model.total_hessian_vector(coordinates, root, direction).ravel())
    timings.append(perf_counter()-start)
    print(f"AD Hessian column {index+1:2d}/{coordinates.size}: {timings[-1]:.2f} s", flush=True)
hessian = np.column_stack(columns)
asymmetry = float(np.max(np.abs(hessian-hessian.T)))
np.testing.assert_allclose(hessian, hessian.T, atol=2e-6, rtol=0)

translations = np.column_stack([np.broadcast_to(axis, coordinates.shape).ravel() for axis in np.eye(3)])
centered = coordinates-coordinates.mean(axis=0)
rotations = np.column_stack([np.cross(centered, axis).ravel() for axis in np.eye(3)])
rotated_gradients = np.column_stack([np.cross(gradient, axis).ravel() for axis in np.eye(3)])
translation_residual = float(np.max(np.abs(hessian@translations)))
rotation_residual = float(np.max(np.abs(hessian@rotations)))
rotation_covariance_error = float(np.max(np.abs(hessian@rotations-rotated_gradients)))
np.testing.assert_allclose(hessian@translations, 0., atol=1e-7, rtol=0)
# A finite residual gradient gives a finite rotational Hessian action. Check
# covariance with that gradient before projecting rigid motions for modes.
np.testing.assert_allclose(hessian@rotations, rotated_gradients, atol=2e-6, rtol=0)

rng = np.random.default_rng(1010)
directions = rng.normal(size=(3, *coordinates.shape))
directions /= np.linalg.norm(directions, axis=(1, 2))[:, None, None]
finite_checks, tracking = [], []
for number, direction in enumerate(directions):
    product = model.total_hessian_vector(coordinates, root, direction)
    np.testing.assert_allclose(product.ravel(), hessian@direction.ravel(), atol=1e-7, rtol=0)
    for step in (1e-3, 3e-4, 1e-4):
        displaced = []
        for sign in (1, -1):
            moved = coordinates+sign*step*direction
            _, g, displaced_state = model.total_gradient(moved, root)
            match = model.match(moved, displaced_state, anchor)
            if match["root"] != root:
                raise RuntimeError("The local Hessian displacement changed selected root.")
            displaced.append(g)
            tracking.append(dict(direction=number, step_bohr=step, sign=sign, **match))
        error = float(np.max(np.abs((displaced[0]-displaced[1])/(2*step)-product)))
        finite_checks.append(dict(direction=number, step_bohr=step, max_hvp_error=error))
        print(f"direction {number}, step {step:.1e} Bohr: HVP-FD error {error:.3e} Eh/Bohr^2", flush=True)
        if step <= 3e-4 and error > 2e-6:
            raise RuntimeError("The native Hessian-vector finite-difference gate failed.")

masses = [12., 15.99491461957, 1.00782503223, 1.00782503223]
modes = normal_modes(hessian, coordinates, masses)
if modes.rigid_rank != 6 or not np.all(modes.frequencies_cm1 > 0):
    raise RuntimeError("The excited minimum does not have six positive harmonic modes.")
zpe = float(.5*np.sum(modes.frequencies_cm1)/HARTREE_TO_WAVENUMBER)
repository = Path(__file__).resolve().parents[2]
source = hashlib.sha256()
for path in sorted((repository/"src").rglob("*.py")):
    source.update(path.relative_to(repository).as_posix().encode())
    source.update(path.read_bytes())
report = dict(
    complete=True, method="RHF/CIS singlet", basis=native["basis"], cart=True,
    symbols=native["symbols"], coordinates_bohr=coordinates.tolist(), selected_root=root, nroots=6,
    ground_energy_hartree=float(state.ground_energy), excitation_hartree=float(state.excitation_energies[root]),
    total_excited_energy_hartree=energy, gradient_hartree_per_bohr=gradient.tolist(),
    max_cartesian_gradient=float(np.max(np.abs(gradient))), hessian_hartree_per_bohr2=hessian.tolist(),
    hessian_method="JVP of relaxed native CIS total-energy gradient; full Cartesian AD Hessian",
    cis_response_solver="shared direct", cis_response_dimension=response_dimension, cis_response_max_dense=512,
    cis_response_rtol=response.linear_config.rtol, cis_response_atol=response.linear_config.atol,
    hessian_column_seconds=timings, raw_hessian_asymmetry=asymmetry,
    raw_translation_residual=translation_residual, raw_rotation_residual=rotation_residual,
    rotation_covariance_error=rotation_covariance_error, directions=directions.tolist(),
    finite_difference_checks=finite_checks, tracking=tracking, minimum_gap_hartree=1e-4,
    masses_amu=masses, frequencies_cm1=modes.frequencies_cm1.tolist(),
    displacements=modes.displacements.tolist(), rigid_rank=modes.rigid_rank,
    mass_weighted_rigid_residual=modes.rigid_residual, harmonic_zpe_hartree=zpe,
    amu_to_electron_mass=AMU_TO_ELECTRON_MASS, hartree_to_wavenumber=HARTREE_TO_WAVENUMBER,
    x03_native_sha256=native_hash, source_revision=native["source_revision"],
    source_python_sha256=source.hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    elapsed_seconds=perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    hostname=platform.node(), python=platform.python_version(), jax=jax.__version__,
    dtype="float64", devices=[str(device) for device in jax.devices()],
)
(output/"excited_ad.json").write_text(json.dumps(report, indent=2)+"\n")
np.savetxt(output/"excited_hessian.csv", hessian, delimiter=",")
print("Excited harmonic frequencies / cm^-1:", modes.frequencies_cm1, flush=True)
print(f"Native excited-state AD Hessian gates passed in {report['elapsed_seconds']:.2f} s", flush=True)

# Measured 2026-10-15 on jy_gpu_1, CPU float64, two threads: 597.66 s.
# Raw Hessian asymmetry 1.900e-13 Eh/Bohr^2; rotation covariance error 4.254e-12.
# Three 1e-4 Bohr directional FD errors: 2.307e-9, 1.553e-9, 1.495e-9 Eh/Bohr^2.
# Six positive modes, 496.965149 to 3294.652759 cm^-1; shared direct CIS response
# uses dimension 272, max_dense 512, rtol 1e-10, atol 0. No global default changes.
