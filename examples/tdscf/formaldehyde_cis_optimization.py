"""X03: optimize a tracked total CIS excited-state energy for formaldehyde.

Run X02 first. Six Cartesian chart coordinates remove translations/rotations:
C is fixed at zero, O lies on +z, H1 lies in the yz plane, and H2 is free.
An H2 x displacement breaks the planar starting symmetry. Existing SciPy
L-BFGS-B uses native first derivatives; final curvature is explicitly finite
differences of gradients, not an AD Hessian of the first-order CIS rule.
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


input_folder = Path("reproducibility/ad_showcase/X02_formaldehyde_cis_gradients")
input_data = json.loads((input_folder / "gradscf.json").read_text())
reference_check = json.loads((input_folder / "pyscf.json").read_text())
if not input_data["complete"] or not reference_check["passed"]:
    raise RuntimeError("Complete and independently validate X02 before optimization.")
if reference_check["native_sha256"] != hashlib.sha256((input_folder / "gradscf.json").read_bytes()).hexdigest():
    raise RuntimeError("The X02 reference comparison is stale.")
output = Path("reproducibility/ad_showcase/X03_formaldehyde_cis_optimization")
output.mkdir(parents=True, exist_ok=True)
started = perf_counter()
coordinates = np.asarray(input_data["coordinates_bohr"])
mol = gto.M(atom=list(zip(input_data["symbols"], coordinates)), basis=input_data["basis"],
            unit="Bohr", cart=True)
mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-12,
             conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=180).run()
if not mf.converged:
    raise RuntimeError("Initial RHF did not converge.")
model = CISGeometry(mf, nroots=6, gradient_tolerance=1e-10)
planar_state = model.state(coordinates)
anchor = model.anchor(coordinates, planar_state, input_data["selected_root"])


def geometry(q):
    """A six-dimensional chart in Bohr; it permits pyramidal distortion."""
    return np.array([[0., 0., 0.], [0., 0., q[0]],
                     [0., q[1], q[2]], [q[3], q[4], q[5]]])


def chart_gradient(gradient):
    return np.array([gradient[1, 2], gradient[2, 1], gradient[2, 2],
                     gradient[3, 0], gradient[3, 1], gradient[3, 2]])


def tracked_value_gradient(q, reference):
    r = geometry(q)
    energy, gradient, state = model.total_gradient(r, reference.root)
    match = model.match(r, state, reference)
    if match["root"] != reference.root:
        energy, gradient, state = model.total_gradient(r, match["root"])
        match = model.match(r, state, reference)
    return energy, gradient, state, match


q0 = np.array([coordinates[1, 2], coordinates[2, 1], coordinates[2, 2],
               .15, coordinates[3, 1], coordinates[3, 2]])
bounds = [(1.8, 3.4), (.7, 2.8), (-2., -.25), (-1.6, 1.6), (-2.8, -.7), (-2., -.25)]
history, evaluations, cache = [], [], {}


def objective(q):
    energy, gradient, state, match = tracked_value_gradient(q, anchor)
    cache.update(q=np.array(q), energy=energy, gradient=gradient, state=state, match=match)
    evaluations.append(dict(energy_hartree=energy, chart_bohr=np.asarray(q).tolist(),
                            max_cartesian_gradient=float(np.max(np.abs(gradient))), **match))
    return energy, chart_gradient(gradient)


def accepted(q):
    global anchor
    if not cache or not np.array_equal(cache["q"], q):
        objective(q)
    r = geometry(q)
    history.append(dict(iteration=len(history), coordinates_bohr=r.tolist(),
                        energy_hartree=cache["energy"],
                        excitation_hartree=float(cache["state"].excitation_energies[cache["match"]["root"]]),
                        max_cartesian_gradient=float(np.max(np.abs(cache["gradient"]))),
                        **cache["match"]))
    # Trial points never update this anchor. Only accepted optimization steps do.
    anchor = model.anchor(r, cache["state"], cache["match"]["root"])
    (output / "progress.json").write_text(json.dumps(dict(complete=False, history=history), indent=2)+"\n")
    print(f"accepted {len(history)-1:2d}: E={cache['energy']:.12f} Eh, "
          f"max|g|={history[-1]['max_cartesian_gradient']:.3e}, overlap={history[-1]['overlap']:.6f}", flush=True)


objective(q0)
accepted(q0)
minimum = minimize(objective, q0, jac=True, method="L-BFGS-B", bounds=bounds, callback=accepted,
                   options=dict(gtol=3e-7, ftol=1e-15, maxiter=100, maxls=30, maxcor=10))
energy, gradient, state, final_match = tracked_value_gradient(minimum.x, anchor)
final_coordinates = geometry(minimum.x)
if not minimum.success or np.max(np.abs(gradient)) > 1e-6:
    raise RuntimeError(f"Excited-state optimization did not meet the full Cartesian criterion: {minimum.message}")
if any(min(value-low, high-value) < 1e-4 for value, (low, high) in zip(minimum.x, bounds)):
    raise RuntimeError("A chart bound is active; do not report an unconstrained minimum.")
np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-7, rtol=0)
np.testing.assert_allclose(np.cross(final_coordinates, gradient).sum(axis=0), 0., atol=1e-7, rtol=0)
final_anchor = model.anchor(final_coordinates, state, final_match["root"])

# Independent finite differences of first gradients test local curvature.
steps = (1e-3, 3e-4)
curvatures, curvature_tracking = [], []
for step in steps:
    columns = []
    for axis, direction in enumerate(np.eye(6)):
        displaced = []
        for sign in (1, -1):
            _, g, _, match = tracked_value_gradient(minimum.x+sign*step*direction, final_anchor)
            displaced.append(chart_gradient(g))
            curvature_tracking.append(dict(step_bohr=step, axis=axis, sign=sign, **match))
        columns.append((displaced[0]-displaced[1])/(2*step))
    hessian = np.array(columns).T
    asymmetry = float(np.max(np.abs(hessian-hessian.T)))
    if asymmetry > 2e-5:
        raise RuntimeError(f"Raw finite-difference curvature asymmetry is too large: {asymmetry}")
    curvatures.append(hessian)
    print(f"Curvature step {step:.1e} Bohr: raw asymmetry {asymmetry:.3e} Eh/Bohr^2", flush=True)
np.testing.assert_allclose(curvatures[-1], curvatures[0], atol=2e-5, rtol=0)
eigenvalues = np.linalg.eigvalsh((curvatures[-1]+curvatures[-1].T)/2)
if np.min(eigenvalues) <= 1e-6:
    raise RuntimeError("The stationary geometry is not a strictly positive minimum in the six-coordinate chart.")

repository = Path(__file__).resolve().parents[2]
source = hashlib.sha256()
for path in sorted((repository / "src").rglob("*.py")):
    source.update(path.relative_to(repository).as_posix().encode())
    source.update(path.read_bytes())
report = dict(
    complete=True, method="RHF/CIS singlet", basis=input_data["basis"], cart=True,
    symbols=input_data["symbols"], coordinates_bohr=final_coordinates.tolist(),
    initial_coordinates_bohr=geometry(q0).tolist(), planar_reference_coordinates_bohr=coordinates.tolist(),
    chart_bohr=minimum.x.tolist(), chart_definition=["O_z", "H1_y", "H1_z", "H2_x", "H2_y", "H2_z"],
    chart_bounds_bohr=bounds, out_of_plane_seed_bohr=.15,
    selected_root=final_match["root"], nroots=6, ground_energy_hartree=float(state.ground_energy),
    excitation_hartree=float(state.excitation_energies[final_match["root"]]),
    total_excited_energy_hartree=energy, gradient_hartree_per_bohr=gradient.tolist(),
    max_cartesian_gradient=float(np.max(np.abs(gradient))), gradient_tolerance=1e-6,
    optimizer="SciPy L-BFGS-B", optimizer_success=bool(minimum.success), optimizer_message=str(minimum.message),
    iterations=int(minimum.nit), objective_calls=len(evaluations), history=history, evaluations=evaluations,
    final_match=final_match, curvature_tracking=curvature_tracking,
    curvature_method="Central differences of first AD gradients in the six-coordinate chart; not an AD Hessian",
    curvature_steps_bohr=list(steps), chart_hessians_hartree_per_bohr2=np.asarray(curvatures).tolist(),
    chart_curvature_eigenvalues_hartree_per_bohr2=eigenvalues.tolist(),
    curvature_absolute_tolerance=2e-5, minimum_curvature=1e-6,
    tracking_metric=input_data["tracking_metric"], minimum_overlap=.85, minimum_matching_margin=.20,
    minimum_gap_hartree=1e-4, source_revision=input_data["source_revision"],
    source_python_sha256=source.hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    x02_input_sha256=hashlib.sha256((input_folder / "gradscf.json").read_bytes()).hexdigest(),
    elapsed_seconds=perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    hostname=platform.node(), python=platform.python_version(), jax=jax.__version__, numpy=np.__version__,
    dtype="float64", devices=[str(device) for device in jax.devices()],
    trajectory_unit="Angstrom", angstrom_per_bohr=0.529177210903,
)
(output / "gradscf.json").write_text(json.dumps(report, indent=2)+"\n")
np.savetxt(output / "chart_curvature.csv", curvatures[-1], delimiter=",",
           header=",".join(report["chart_definition"]), comments="")
with (output / "trajectory.xyz").open("w") as handle:
    for item in history:
        handle.write(f"4\nCIS accepted step {item['iteration']}; Angstrom; E={item['energy_hartree']:.12f} Eh\n")
        for symbol, xyz in zip(report["symbols"], np.asarray(item["coordinates_bohr"])*report["angstrom_per_bohr"]):
            handle.write(f"{symbol:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")
print("Final coordinates / Bohr:\n", final_coordinates, flush=True)
print("Chart curvature eigenvalues / (Eh/Bohr^2):", eigenvalues, flush=True)
print(f"Tracked S1 minimum accepted; max|g|={report['max_cartesian_gradient']:.3e}; elapsed {report['elapsed_seconds']:.2f} s", flush=True)

# Measured 2026-10-15 on jy_gpu_1, CPU float64, two threads:
# 26 iterations / 34 objective calls; E_excited = -113.705362557861 Eh.
# max Cartesian |g| = 5.877e-8 Eh/Bohr; minimum chart curvature = 0.00357898
# Eh/Bohr^2. Curvature is finite differences of first gradients, not AD Hessian.
# Native optimization + curvature: 622.66 s; independent comparison passed.
