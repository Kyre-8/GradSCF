"""X10: paired RHF/CIS minima, vertical gaps, adiabatic gap and harmonic 0-0.

Run formaldehyde_cis_hessian.py and its independent comparison first. Ground
optimization/response use NativeRHF; excited energies use the shared CIS helper.
All four electronic energies use the same Cartesian 6-31+G* model.
"""
import hashlib
import json
from pathlib import Path
import platform
import resource
from time import perf_counter

import jax
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf
from gradscf_tools.excited import CISGeometry
from gradscf_tools.molecular_ep import NativeRHF
from gradscf_tools.vibrations import normal_modes, HARTREE_TO_WAVENUMBER


output = Path("reproducibility/ad_showcase/X10_formaldehyde_cis_emission")
excited = json.loads((output/"excited_ad.json").read_text())
independent = json.loads((output/"excited_pyscf.json").read_text())
excited_hash = hashlib.sha256((output/"excited_ad.json").read_bytes()).hexdigest()
if not excited["complete"] or not independent["passed"] or independent["native_sha256"] != excited_hash:
    raise RuntimeError("Independently validate the complete excited AD Hessian first.")
initial = json.loads(Path("reproducibility/ad_showcase/X02_formaldehyde_cis_gradients/gradscf.json").read_text())
coordinates0 = np.asarray(initial["coordinates_bohr"])
config = scf.RKSConfig(xc_spec="hf", conv_tol=1e-12, conv_tol_density=1e-11,
                       conv_tol_grad=1e-10, max_cycle=180)
engine = NativeRHF(list(zip(excited["symbols"], coordinates0)), basis=excited["basis"], config=config)
started = perf_counter()
evaluations = []


def objective(flat_coordinates):
    coordinates = flat_coordinates.reshape(-1, 3)
    state, gradient = engine.evaluate(coordinates)
    energy = float(state.total_energy)
    evaluations.append(dict(energy_hartree=energy, coordinates_bohr=coordinates.tolist(),
                            max_cartesian_gradient=float(np.max(np.abs(gradient)))))
    print(f"RHF objective {len(evaluations):2d}: E={energy:.12f}, max|g|={evaluations[-1]['max_cartesian_gradient']:.3e}", flush=True)
    return energy, gradient.ravel()


minimum = minimize(objective, coordinates0.ravel(), jac=True, method="BFGS",
                   options=dict(gtol=3e-8, maxiter=100))
coordinates = minimum.x.reshape(-1, 3)
ground, gradient = engine.evaluate(coordinates)
if not minimum.success or np.max(np.abs(gradient)) > 1e-7:
    raise RuntimeError(f"Ground minimum did not converge: {minimum.message}")
np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-8, rtol=0)
np.testing.assert_allclose(np.cross(coordinates, gradient).sum(axis=0), 0., atol=1e-8, rtol=0)
response = engine.response(coordinates, ground)
columns, column_seconds = [], []
for index, direction in enumerate(np.eye(coordinates.size).reshape(-1, *coordinates.shape)):
    start = perf_counter()
    product, _ = response(direction)
    columns.append(product.ravel())
    column_seconds.append(perf_counter()-start)
    print(f"RHF Hessian column {index+1:2d}: {column_seconds[-1]:.2f} s", flush=True)
hessian = np.column_stack(columns)
np.testing.assert_allclose(hessian, hessian.T, atol=2e-6, rtol=0)
translations = np.tile(np.eye(3), (len(coordinates), 1))
centered = coordinates-coordinates.mean(axis=0)
rotations = np.column_stack([np.cross(centered, axis).ravel() for axis in np.eye(3)])
rotated_gradients = np.column_stack([np.cross(gradient, axis).ravel() for axis in np.eye(3)])
np.testing.assert_allclose(hessian@translations, 0., atol=1e-7, rtol=0)
np.testing.assert_allclose(hessian@rotations, rotated_gradients, atol=2e-6, rtol=0)
direction = np.random.default_rng(2010).normal(size=coordinates.shape)
direction /= np.linalg.norm(direction)
product, _ = response(direction)
np.testing.assert_allclose(product.ravel(), hessian@direction.ravel(), atol=1e-7, rtol=0)
checks = []
for step in (1e-3, 3e-4, 1e-4):
    _, plus = engine.evaluate(coordinates+step*direction, init_density=ground.density_matrix)
    _, minus = engine.evaluate(coordinates-step*direction, init_density=ground.density_matrix)
    error = float(np.max(np.abs((plus-minus)/(2*step)-product)))
    checks.append(dict(step_bohr=step, max_hvp_error=error))
    if step <= 3e-4 and error > 2e-6:
        raise RuntimeError("Ground-state Hessian finite-difference gate failed.")
modes = normal_modes(hessian, coordinates, excited["masses_amu"])
if modes.rigid_rank != 6 or not np.all(modes.frequencies_cm1 > 0):
    raise RuntimeError("Ground structure is not a harmonic minimum.")
ground_zpe = float(.5*np.sum(modes.frequencies_cm1)/HARTREE_TO_WAVENUMBER)

mol = gto.M(atom=list(zip(excited["symbols"], coordinates)), basis=excited["basis"], unit="Bohr", cart=True)
mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-12,
             conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=180).run()
model = CISGeometry(mf, nroots=excited["nroots"], gradient_tolerance=1e-10)
state = model.state(coordinates)
# Confirm that the ground-geometry state connects to the validated excited
# minimum by a local continuation, retaining every overlap/gap diagnostic.
anchor = model.anchor(coordinates, state, excited["selected_root"])
ground_excitation = float(state.excitation_energies[anchor.root])
continuation = []
excited_coordinates = np.asarray(excited["coordinates_bohr"])
for fraction in np.linspace(0., 1., 11)[1:]:
    moved = (1-fraction)*coordinates+fraction*excited_coordinates
    next_state = model.state(moved)
    match = model.match(moved, next_state, anchor)
    continuation.append(dict(fraction=float(fraction), **match))
    anchor = model.anchor(moved, next_state, match["root"])
if anchor.root != excited["selected_root"]:
    raise RuntimeError("The connected state does not match the validated excited minimum.")
np.testing.assert_allclose(float(state.ground_energy), ground.total_energy, atol=1e-8, rtol=0)
e0g = float(ground.total_energy)
e1g = e0g+ground_excitation
e0e = excited["ground_energy_hartree"]
e1e = excited["total_excited_energy_hartree"]
energies = dict(ground_at_ground=e0g, excited_at_ground=e1g,
                ground_at_excited=e0e, excited_at_excited=e1e)
gaps = dict(vertical_absorption=e1g-e0g, vertical_emission=e1e-e0e,
            adiabatic_electronic=e1e-e0g,
            harmonic_zero_zero=e1e-e0g+excited["harmonic_zpe_hartree"]-ground_zpe,
            excited_relaxation=e1g-e1e, ground_reorganization=e0e-e0g,
            stokes_shift=(e1g-e0g)-(e1e-e0e))
np.testing.assert_allclose(gaps["stokes_shift"], gaps["excited_relaxation"]+gaps["ground_reorganization"],
                           atol=1e-12, rtol=0)
if min(gaps["excited_relaxation"], gaps["ground_reorganization"]) < -1e-8:
    raise RuntimeError("The paired minima have inconsistent relaxation energies.")
repository = Path(__file__).resolve().parents[2]
source = hashlib.sha256()
for path in sorted((repository/"src").rglob("*.py")):
    source.update(path.relative_to(repository).as_posix().encode())
    source.update(path.read_bytes())
report = dict(
    complete=True, method="RHF/CIS singlet", basis=excited["basis"], cart=True, symbols=excited["symbols"],
    ground_coordinates_bohr=coordinates.tolist(), excited_coordinates_bohr=excited_coordinates.tolist(),
    electronic_energies_hartree=energies, gaps_hartree=gaps, ground_harmonic_zpe_hartree=ground_zpe,
    excited_harmonic_zpe_hartree=excited["harmonic_zpe_hartree"],
    ground_gradient_hartree_per_bohr=gradient.tolist(), ground_max_cartesian_gradient=float(np.max(np.abs(gradient))),
    ground_hessian_hartree_per_bohr2=hessian.tolist(), ground_hessian_column_seconds=column_seconds,
    ground_raw_hessian_asymmetry=float(np.max(np.abs(hessian-hessian.T))),
    ground_raw_translation_residual=float(np.max(np.abs(hessian@translations))),
    ground_raw_rotation_residual=float(np.max(np.abs(hessian@rotations))),
    ground_rotation_covariance_error=float(np.max(np.abs(hessian@rotations-rotated_gradients))),
    ground_finite_difference_checks=checks, masses_amu=excited["masses_amu"],
    ground_frequencies_cm1=modes.frequencies_cm1.tolist(), ground_displacements=modes.displacements.tolist(),
    ground_mass_weighted_rigid_residual=modes.rigid_residual, continuation=continuation,
    optimizer_success=bool(minimum.success), optimizer_message=str(minimum.message), iterations=int(minimum.nit),
    objective_calls=len(evaluations), evaluations=evaluations, excited_native_sha256=excited_hash,
    source_revision=excited["source_revision"], source_python_sha256=source.hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    elapsed_seconds=perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    hostname=platform.node(), python=platform.python_version(), jax=jax.__version__, dtype="float64",
)
(output/"emission.json").write_text(json.dumps(report, indent=2)+"\n")
np.savetxt(output/"ground_hessian.csv", hessian, delimiter=",")
print("Paired electronic energies / Hartree:", energies, flush=True)
print("Gaps and reorganization energies / Hartree:", gaps, flush=True)
print(f"Native paired-minimum gates passed in {report['elapsed_seconds']:.2f} s", flush=True)

# Measured 2026-10-15 on jy_gpu_1, CPU float64, two threads: 244.08 s.
# Ground BFGS: 8 iterations / 10 objective calls, max Cartesian |g| 2.260e-8 Eh/Bohr.
# Gaps / eV: absorption 4.717640637; emission 4.041417611;
# electronic adiabatic 4.511492146; harmonic 0-0 4.402847465.
# These are one pair of local RHF/CIS minima; no anharmonic/expt/basis-convergence claim.
