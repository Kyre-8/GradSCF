"""X07: tracked bright/dark CIS intensity response along RHF normal modes.

Use the independently validated X10 ground minimum and harmonic modes.
The selected states continue X05's S1 and brightest-of-six S4. Derivatives
come from relaxed native HF/CIS transition properties, not a fitted spectrum.
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
from gradscf_tools.excited import CISAnchor, CISGeometry, singlet_cis_densities
from gradscf_tools.vibrations import AMU_TO_ELECTRON_MASS


inputs = Path("reproducibility/ad_showcase")
paired_path = inputs/"X10_formaldehyde_cis_emission/emission.json"
paired = json.loads(paired_path.read_text())
paired_ref = json.loads((paired_path.parent/"emission_pyscf.json").read_text())
fixed_path = inputs/"X04_formaldehyde_cis_density/gradscf.json"
fixed = json.loads(fixed_path.read_text())
fixed_ref = json.loads((fixed_path.parent/"pyscf.json").read_text())
character_path = inputs/"X05_formaldehyde_cis_ntos/gradscf.json"
character = json.loads(character_path.read_text())
character_ref = json.loads((character_path.parent/"pyscf.json").read_text())
for path, data, checked in ((paired_path, paired, paired_ref), (fixed_path, fixed, fixed_ref),
                             (character_path, character, character_ref)):
    if not data["complete"] or not checked["passed"] or checked["native_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise RuntimeError("Use the independently validated X04/X05/X10 input records.")
output = inputs/"X07_formaldehyde_oscillator_response"
output.mkdir(parents=True, exist_ok=True)
(output/"gradscf.json").write_text('{"complete": false}\n')
started = perf_counter()
coordinates = np.asarray(paired["ground_coordinates_bohr"])
masses = np.asarray(paired["masses_amu"])
# Convert dR/dQ from the shared helper's electron-mass units to amu units.
modes = np.asarray(paired["ground_displacements"])*np.sqrt(AMU_TO_ELECTRON_MASS)
np.testing.assert_allclose(np.einsum("aim,ain,a->mn", modes, modes, masses), np.eye(6), atol=1e-10, rtol=0)
mol = gto.M(atom=list(zip(paired["symbols"], coordinates)), basis=paired["basis"], unit="Bohr", cart=True)
mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-12,
             conv_tol_density=1e-11, conv_tol_grad=1e-11, max_cycle=180).run()
if not mf.converged:
    raise RuntimeError("RHF did not converge.")
policy = EigenResponseConfig(target="eigenpairs",
    linear_config=LinearSolverConfig(method="direct", max_dense=512, rtol=1e-11))
model = CISGeometry(mf, nroots=6, gradient_tolerance=1e-11, response=policy)
initial_coordinates = np.asarray(fixed["coordinates_bohr"])
anchors = [CISAnchor(initial_coordinates, np.asarray(fixed["coefficients_ao"]),
                     np.asarray(fixed["amplitudes"])[root], root) for root in character["selected_roots"]]
continuation = []
for fraction in np.linspace(0., 1., 9)[1:]:
    moved = initial_coordinates+fraction*(coordinates-initial_coordinates)
    state = model.state(moved)
    matches = [model.match(moved, state, anchor) for anchor in anchors]
    if len({match["root"] for match in matches}) != 2:
        raise RuntimeError("Continuation assigned both anchors to one root.")
    continuation.append(dict(fraction=float(fraction), matches=matches))
    # Update only after both states pass the declared overlap/margin/gap gate.
    anchors = [model.anchor(moved, state, match["root"]) for match in matches]
initial_matches = continuation[-1]["matches"]
roots = [match["root"] for match in initial_matches]
if len(set(roots)) != 2:
    raise RuntimeError("The dark and bright state anchors matched the same root.")
values, jacobians, transitions, anchors = [], [], [], []
for root in roots:
    value, jacobian, current = model.transition_derivatives(coordinates, root)
    values.append(value)
    jacobians.append(jacobian)
    anchors.append(model.anchor(coordinates, current, root))
    transitions.append(np.asarray(singlet_cis_densities(current.coefficients, current.amplitudes[root], nocc=model.nocc).transition))
values, jacobians = np.asarray(values), np.asarray(jacobians)
mode_response = np.einsum("sxai,aim->sxm", jacobians, modes)
if values[0, 3] > 1e-16 or values[1, 3] < 1e-3:
    raise RuntimeError("The selected states do not retain the intended dark/bright character.")
np.testing.assert_allclose(jacobians[0, 3], 0., atol=1e-9, rtol=0)
np.testing.assert_allclose(np.sum(jacobians[:, 3], axis=1), 0., atol=2e-7, rtol=0)
# state is the accepted continuation endpoint at these same coordinates.
dark_gamma = (2/3)*float(state.excitation_energies[roots[0]])*np.sum(mode_response[0, :3]**2, axis=0)
tracking = []


def evaluate(r):
    # Return moments in the continuous transition-density phase relative to
    # each baseline anchor. Oscillator strengths themselves need no phase fix.
    first, current = model.transition_properties(r, roots[0])
    cross = np.asarray(model.overlap_engine.cross_overlap(coordinates, r))
    result = []
    for anchor in anchors:
        match = model.match(r, current, anchor)
        root = match["root"]
        value = (first if root == roots[0] else model.transition_properties(r, root)[0]).copy()
        occupied = anchor.coefficients[:, :model.nocc].T@cross@np.asarray(current.coefficients)[:, :model.nocc]
        virtual = anchor.coefficients[:, model.nocc:].T@cross@np.asarray(current.coefficients)[:, model.nocc:]
        inner = np.sum(anchor.amplitude*(occupied@np.asarray(current.amplitudes[root])@virtual.T))
        value[:3] *= 1. if inner >= 0 else -1.
        tracking.append(match)
        result.append(value)
    return np.asarray(result)


checks = []
for mode in range(6):
    for step in (2e-3, 6e-4, 2e-4):
        plus, minus = evaluate(coordinates+step*modes[:, :, mode]), evaluate(coordinates-step*modes[:, :, mode])
        finite = (plus-minus)/(2*step)
        quadratic = float((plus[0, 3]+minus[0, 3]-2*values[0, 3])/(2*step**2))
        error = float(np.max(np.abs(finite-mode_response[:, :, mode])))
        checks.append(dict(mode=mode, step_bohr_sqrt_amu=step, derivative=finite.tolist(),
                            max_error=error, dark_quadratic_coefficient=quadratic,
                            dark_quadratic_error=abs(quadratic-dark_gamma[mode])))
        if step == 2e-4:
            np.testing.assert_allclose(finite, mode_response[:, :, mode], atol=2e-6, rtol=0)
            np.testing.assert_allclose(quadratic, dark_gamma[mode], atol=2e-6, rtol=0)
    print("Mode", mode+1, "final moment/f derivative error", error, flush=True)
selected_modes = sorted(set((int(np.argmax(dark_gamma)), int(np.argmax(np.abs(mode_response[1, 3]))))))
scans = []
for mode in selected_modes:
    for displacement in np.linspace(-.06, .06, 13):
        observed = evaluate(coordinates+displacement*modes[:, :, mode])
        scans.append(dict(mode=mode, displacement_bohr_sqrt_amu=float(displacement),
                          values=observed.tolist(), dark_quadratic_prediction=float(dark_gamma[mode]*displacement**2),
                          bright_linear_prediction=float(values[1, 3]+mode_response[1, 3, mode]*displacement)))
repository = Path(__file__).resolve().parents[2]
digest = hashlib.sha256()
for path in sorted((repository/"src").rglob("*.py")):
    digest.update(path.relative_to(repository).as_posix().encode())
    digest.update(path.read_bytes())
report = dict(
    complete=True, method="RHF/CIS singlet", basis=paired["basis"], cart=True, symbols=paired["symbols"],
    coordinates_bohr=coordinates.tolist(), masses_amu=masses.tolist(), modes_bohr_per_Q=modes.tolist(),
    mode_coordinate_units="Bohr sqrt(amu); R=R0+Q*mode, with mode.T M_amu mode=1",
    frequencies_cm1=paired["ground_frequencies_cm1"], nroots=6, nocc=model.nocc, selected_roots=roots,
    original_state_labels=[f"S{root+1}" for root in character["selected_roots"]], initial_state_matches=initial_matches,
    initial_coordinates_bohr=initial_coordinates.tolist(), continuation=continuation,
    excitation_energies_hartree=np.asarray(state.excitation_energies).tolist(),
    values=values.tolist(), value_columns=["mu_x_au", "mu_y_au", "mu_z_au", "oscillator_strength"],
    cartesian_jacobians=jacobians.tolist(), mode_derivatives=mode_response.tolist(),
    baseline_transition_densities_ao=np.asarray(transitions).tolist(),
    dark_quadratic_coefficients=dark_gamma.tolist(), finite_difference_checks=checks,
    selected_modes=selected_modes, scans=scans, tracking=tracking,
    input_sha256={str(path.relative_to(inputs)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (paired_path, fixed_path, character_path)},
    source_python_sha256=digest.hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    elapsed_seconds=perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    hostname=platform.node(), python=platform.python_version(), jax=jax.__version__, dtype="float64",
    scope="Fixed ground-state harmonic mode directions; intensity modulation, not a vibronic spectrum or linewidth",
)
(output/"gradscf.json").write_text(json.dumps(report, indent=2)+"\n")
print("X07 native transition/intensity-response gates passed", report["elapsed_seconds"], "s", flush=True)

# Measured jy_gpu_1 CPU float64, OMP/BLAS=2: native gates passed in 717.00 s.
# Eight accepted geometry increments were required after a direct match gave overlap .722308.
# Six-mode finest-step moment/f error <=1.023e-6; S1 f(0)=2.89e-31, S4 f(0)=.182118727539.
# Gamma_6=.004972609839 per (Bohr sqrt(amu))^2; this is intensity modulation, not a vibronic spectrum.
