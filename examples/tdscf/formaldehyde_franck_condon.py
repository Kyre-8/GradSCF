"""X11: unit-Condon Franck-Condon probabilities from the validated X10 minima.

This is a common-ground-Eckart-slice double-harmonic model. Raw endpoint mode
overlap is diagnosed, never silently made orthogonal. The final curvature is
projected into the ground vibrational tangent space with a common flat kinetic
operator. Its frequencies/zero-point energy differ from the original X10 modes.
All numerical work runs on jy_gpu_1; no CLI and no new electronic solver.
"""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import trapezoid
from scipy.spatial.transform import Rotation

jax.config.update("jax_enable_x64", True)

from gradscf.solvers import EigenSolverConfig, solve_hermitian
from gradscf_tools.franck_condon import ground_state_overlaps, zero_zero_probability


INPUT = Path("reproducibility/ad_showcase/X10_formaldehyde_cis_emission")
OUTPUT = Path("reproducibility/ad_showcase/X11_formaldehyde_franck_condon")
MAX_QUANTA = 30
CUTOFFS = (4, 8, 12, 16, 20, 24, 28, 30)
HARTREE_TO_EV = 27.211386245988


def load_inputs():
    ground = json.loads((INPUT/"emission.json").read_text())
    excited = json.loads((INPUT/"excited_ad.json").read_text())
    hashes = {}
    for name, refname in (("emission.json", "emission_pyscf.json"),
                          ("excited_ad.json", "excited_pyscf.json")):
        hashes[name] = hashlib.sha256((INPUT/name).read_bytes()).hexdigest()
        reference = json.loads((INPUT/refname).read_text())
        assert reference["passed"] and reference["native_sha256"] == hashes[name]
    assert ground["complete"] and excited["complete"]
    assert ground["excited_native_sha256"] == hashes["excited_ad.json"]
    assert ground["symbols"] == excited["symbols"]
    assert ground["ground_max_cartesian_gradient"] < 1e-6
    assert excited["max_cartesian_gradient"] < 1e-6
    return ground, excited, hashes


def rigid_frame(coordinates, masses_electron):
    root = np.sqrt(masses_electron)[:, None]
    columns = []
    for axis in np.eye(3):
        columns.extend([(np.broadcast_to(axis, coordinates.shape)*root).ravel(),
                        (np.cross(coordinates, axis)*root).ravel()])
    frame, singular, _ = np.linalg.svd(np.asarray(columns).T, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0]*1e-10))
    if rank != 6:
        raise ValueError("This nonlinear-molecule example requires six rigid motions.")
    return frame


def common_model(ground, excited):
    masses = np.asarray(ground["masses_amu"])
    masses_electron = masses*excited["amu_to_electron_mass"]
    root_mass = np.sqrt(np.repeat(masses_electron, 3))
    rg = np.asarray(ground["ground_coordinates_bohr"])
    re = np.asarray(ground["excited_coordinates_bohr"])
    rg = rg-np.average(rg, axis=0, weights=masses)
    re = re-np.average(re, axis=0, weights=masses)
    left, _, right = np.linalg.svd((re*masses[:, None]).T@rg)
    rotation = left@np.diag([1., 1., np.linalg.det(left@right)])@right
    re = re@rotation
    lg = np.asarray(ground["ground_displacements"]).reshape(-1, 6)*root_mass[:, None]
    le = np.einsum("aim,ij->ajm", np.asarray(excited["displacements"]), rotation).reshape(-1, 6)*root_mass[:, None]
    for modes, coordinates in ((lg, rg), (le, re)):
        np.testing.assert_allclose(modes.T@modes, np.eye(6), atol=1e-10, rtol=0)
        np.testing.assert_allclose(rigid_frame(coordinates, masses_electron).T@modes, 0., atol=1e-10, rtol=0)
    cm = excited["hartree_to_wavenumber"]
    wi = np.asarray(ground["ground_frequencies_cm1"])/cm
    original_wf = np.asarray(excited["frequencies_cm1"])/cm
    if np.any(wi <= 0) or np.any(original_wf <= 0):
        raise ValueError("Both input surfaces must have positive internal frequencies.")
    # Use the validated vibrational reconstruction, with rigid directions
    # excluded only after X10's raw Hessian/stationarity checks. Retain the
    # comparison with the raw Cartesian Hessian below.
    dynamic = (le*original_wf**2)@le.T
    block_rotation = np.kron(np.eye(len(masses)), rotation.T)
    raw_hessian = block_rotation@np.asarray(excited["hessian_hartree_per_bohr2"])@block_rotation.T
    if np.max(abs(raw_hessian-raw_hessian.T)) > 2e-6:
        raise ValueError("The raw excited Hessian is not symmetric within the X10 gate.")
    raw_dynamic = .5*(raw_hessian+raw_hessian.T)/np.outer(root_mass, root_mass)
    # δ is ground minus excited, so the complete restricted potential is
    # E_e,min + c + b.T Q + Q.T Ke Q/2, not the measured vertical gap plus c.
    delta = (rg-re).ravel()*root_mass
    curvature = lg.T@dynamic@lg
    linear = lg.T@dynamic@delta
    constant = .5*delta@dynamic@delta
    center = -np.linalg.solve(curvature, linear)
    minimum_offset = constant+.5*linear@center
    solved = solve_hermitian(jnp.asarray(curvature), config=EigenSolverConfig(
        method="dense", nroots=6, atol=1e-12))
    assert np.all(solved.converged)
    eigenvalues = np.asarray(solved.raw_values)[:6]
    if np.any(eigenvalues <= 0) or minimum_offset < -1e-10:
        raise ValueError("The restricted harmonic potential is not positive/stable.")
    wf = np.sqrt(eigenvalues)
    vectors = np.asarray(solved.raw_vectors)[:, :6]
    canonical = vectors.T
    shift = -canonical@center
    raw_j = le.T@lg
    raw_projected = lg.T@raw_dynamic@lg
    raw_eigenvalues = np.linalg.eigvalsh(raw_projected)
    if np.any(raw_eigenvalues <= 0):
        raise ValueError("The raw-Hessian common-slice comparison is not positive.")
    np.testing.assert_allclose(canonical.T@canonical, np.eye(6), atol=1e-12, rtol=0)
    np.testing.assert_allclose(curvature@center+linear, 0., atol=1e-14, rtol=0)
    electronic_gap = ground["gaps_hartree"]["adiabatic_electronic"]
    zero_zero = electronic_gap+minimum_offset+.5*(wf.sum()-wi.sum())
    return dict(initial_frequencies_hartree=wi, final_frequencies_hartree=wf,
        canonical_rotation=canonical, shift=shift, curvature=curvature,
        linear=linear, constant=constant, center=center, minimum_offset=minimum_offset,
        ground_coordinates_centered_bohr=rg, excited_coordinates_aligned_bohr=re,
        excited_to_ground_rotation=rotation, mass_weighted_delta=delta,
        raw_duschinsky_overlap=raw_j, raw_J_singular_values=np.linalg.svd(raw_j, compute_uv=False),
        raw_J_determinant=np.linalg.det(raw_j), raw_J_orthogonality_max=np.max(abs(raw_j.T@raw_j-np.eye(6))),
        model_J_orthogonality_max=np.max(abs(canonical.T@canonical-np.eye(6))),
        model_J_abs_determinant=abs(np.linalg.det(canonical)),
        eckart_residual_amu_bohr2=np.linalg.norm(np.sum(masses[:, None]*np.cross(rg, re), axis=0)),
        delta_outside_ground_tangent_norm=np.linalg.norm(delta-lg@(lg.T@delta)),
        center_vs_projected_displacement_error=np.linalg.norm(center+lg.T@delta),
        raw_vs_vibrational_dynamic_max=np.max(abs(raw_dynamic-dynamic)),
        raw_vs_vibrational_curvature_max=np.max(abs(raw_projected-curvature)),
        raw_projected_frequencies_cm1=np.sqrt(raw_eigenvalues)*cm,
        original_excited_frequencies_cm1=original_wf*cm, model_excited_frequencies_cm1=wf*cm,
        projection_frequency_changes_cm1=(wf-original_wf)*cm,
        model_zpe_change_from_X10_hartree=.5*(wf-original_wf).sum(),
        model_zero_zero_hartree=zero_zero, X10_zero_zero_hartree=ground["gaps_hartree"]["harmonic_zero_zero"],
        model_vertical_gap_hartree=electronic_gap+constant,
        X10_vertical_gap_hartree=ground["gaps_hartree"]["vertical_absorption"],
        model_excited_reorganization_hartree=constant-minimum_offset,
        X10_excited_reorganization_hartree=ground["gaps_hartree"]["excited_relaxation"],
        analytic_mean_photon_energy_hartree=electronic_gap+constant+.25*(np.sum(np.diag(curvature)/wi)-wi.sum()))


def invariance_checks(ground, excited, model):
    baseline_states, baseline = ground_state_overlaps(model["initial_frequencies_hartree"],
        model["final_frequencies_hartree"], model["canonical_rotation"], model["shift"], max_quanta=4)
    results = []
    for operation in ("independent_rigid_frames", "mode_signs_and_permutations"):
        g, e = deepcopy(ground), deepcopy(excited)
        if operation == "independent_rigid_frames":
            for which, rotvec, translation in (("ground", [.2, -.3, .1], [.4, -.2, .3]),
                                               ("excited", [-.1, .4, .2], [-.3, .6, .1])):
                rotation = Rotation.from_rotvec(rotvec).as_matrix()
                key = f"{which}_coordinates_bohr"
                g[key] = np.asarray(g[key])@rotation+translation
                if which == "ground":
                    g["ground_displacements"] = np.einsum("aim,ij->ajm", np.asarray(g["ground_displacements"]), rotation)
                else:
                    e["displacements"] = np.einsum("aim,ij->ajm", np.asarray(e["displacements"]), rotation)
                    block = np.kron(np.eye(4), rotation.T)
                    e["hessian_hartree_per_bohr2"] = block@np.asarray(e["hessian_hartree_per_bohr2"])@block.T
        else:
            permutation = np.array([3, 0, 5, 2, 1, 4])
            signs = np.array([1., -1., 1., -1., -1., 1.])
            g["ground_displacements"] = np.asarray(g["ground_displacements"])[..., permutation]*signs
            g["ground_frequencies_cm1"] = np.asarray(g["ground_frequencies_cm1"])[permutation]
            e["displacements"] = np.asarray(e["displacements"])[..., permutation]*signs
            e["frequencies_cm1"] = np.asarray(e["frequencies_cm1"])[permutation]
        changed = common_model(g, e)
        if operation == "independent_rigid_frames":
            expected_linear, expected_curvature, expected_center = model["linear"], model["curvature"], model["center"]
        else:
            expected_linear = signs*model["linear"][permutation]
            expected_curvature = model["curvature"][np.ix_(permutation, permutation)]*np.outer(signs, signs)
            expected_center = signs*model["center"][permutation]
        for key, expected in (("linear", expected_linear), ("curvature", expected_curvature)):
            np.testing.assert_allclose(changed[key], expected, atol=1e-13, rtol=0)
        np.testing.assert_allclose(changed["center"], expected_center, atol=1e-9, rtol=0)
        for key in ("constant", "minimum_offset"):
            np.testing.assert_allclose(changed[key], model[key], atol=1e-13, rtol=0)
        np.testing.assert_allclose(changed["final_frequencies_hartree"], model["final_frequencies_hartree"], atol=1e-12, rtol=0)
        states, amplitudes = ground_state_overlaps(changed["initial_frequencies_hartree"],
            changed["final_frequencies_hartree"], changed["canonical_rotation"], changed["shift"], max_quanta=4)
        np.testing.assert_array_equal(states, baseline_states)
        error = float(np.max(abs(amplitudes**2-baseline**2)))
        np.testing.assert_allclose(amplitudes**2, baseline**2, atol=5e-11, rtol=0)
        np.testing.assert_allclose(changed["model_zero_zero_hartree"], model["model_zero_zero_hartree"], atol=1e-12, rtol=0)
        results.append(dict(operation=operation, maximum_probability_error=error,
            linear_covariance_error=float(np.max(abs(changed["linear"]-expected_linear))),
            curvature_covariance_error=float(np.max(abs(changed["curvature"]-expected_curvature))),
            center_covariance_error=float(np.max(abs(changed["center"]-expected_center))),
            constant_error_hartree=abs(changed["constant"]-model["constant"]),
            minimum_offset_error_hartree=abs(changed["minimum_offset"]-model["minimum_offset"]),
            maximum_frequency_error_hartree=float(np.max(abs(changed["final_frequencies_hartree"]-model["final_frequencies_hartree"]))),
            zero_zero_energy_error_hartree=abs(changed["model_zero_zero_hartree"]-model["model_zero_zero_hartree"])))
    return results


def profile(axis, energies, weights, sigma):
    # Screening affects only plotting and has an explicit omitted-weight bound.
    order = np.argsort(weights)[::-1]
    target = max(0., float(weights.sum())-1e-11)
    count = min(len(order), int(np.searchsorted(np.cumsum(weights[order]), target))+1)
    kept = order[:count]
    omitted = float(weights.sum()-weights[kept].sum())
    curve = np.zeros_like(axis)
    for start in range(0, len(kept), 256):
        index = kept[start:start+256]
        delta = (axis[:, None]-energies[index][None, :])/sigma
        curve += np.exp(-.5*delta**2)@weights[index]/(np.sqrt(2*np.pi)*sigma)
    return curve, omitted


def main():
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ground, excited, hashes = load_inputs()
    model = common_model(ground, excited)
    checks = invariance_checks(ground, excited, model)
    print("raw J orthogonality error", model["raw_J_orthogonality_max"], flush=True)
    print("model frequency shifts / cm^-1", model["projection_frequency_changes_cm1"], flush=True)
    print("Building total-quanta FC amplitudes through", MAX_QUANTA, flush=True)
    states, amplitudes = ground_state_overlaps(model["initial_frequencies_hartree"],
        model["final_frequencies_hartree"], model["canonical_rotation"], model["shift"], max_quanta=MAX_QUANTA)
    weights = amplitudes**2
    total_quanta = states.sum(axis=1)
    energy_hartree = model["model_zero_zero_hartree"]+states@model["final_frequencies_hartree"]
    energy_ev = energy_hartree*HARTREE_TO_EV
    axis = np.linspace(model["model_zero_zero_hartree"]*HARTREE_TO_EV-.15,
                       model["model_zero_zero_hartree"]*HARTREE_TO_EV+1.7, 2501)
    convergence, previous = [], None
    for cutoff in CUTOFFS:
        selected = total_quanta <= cutoff
        current = [profile(axis, energy_ev[selected], weights[selected], sigma) for sigma in (.02, .01)]
        curves = [item[0] for item in current]
        relative = None if previous is None else [float(trapezoid(abs(a-b), axis)/trapezoid(a, axis))
                                                 for a, b in zip(curves, previous)]
        retained = float(weights[selected].sum())
        convergence.append(dict(max_quanta=cutoff, states=int(selected.sum()), raw_probability_sum=retained,
            missing_probability=1-retained, gaussian_sigma_ev=[.02, .01], relative_profile_l1_change=relative,
            profile_screened_probability=[item[1] for item in current],
            finite_window_probability=[float(trapezoid(curve, axis)) for curve in curves]))
        previous = curves
        print("FC cutoff", cutoff, "probability", retained, "profile change", relative, flush=True)
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-8, rtol=0)
    assert max(convergence[-1]["relative_profile_l1_change"]) < 1e-5
    moment = float(weights@energy_hartree)
    np.testing.assert_allclose(moment, model["analytic_mean_photon_energy_hartree"], atol=2e-8, rtol=0)
    zero_zero = lambda scale: zero_zero_probability(model["initial_frequencies_hartree"],
        model["final_frequencies_hartree"], model["canonical_rotation"], scale*jnp.asarray(model["shift"]))
    probability, derivative = jax.jit(jax.value_and_grad(zero_zero))(1.)
    finite = []
    for step in (1e-3, 3e-4, 1e-4):
        fd = float((zero_zero(1+step)-zero_zero(1-step))/(2*step))
        finite.append(dict(step=step, derivative=fd, absolute_error=abs(fd-float(derivative))))
    assert finite[-1]["absolute_error"] < 5e-8
    np.testing.assert_allclose(probability, weights[0], atol=1e-13, rtol=0)
    np.savez_compressed(OUTPUT/"franck_condon_states.npz", occupations=states.astype(np.uint8),
                        amplitudes=amplitudes, photon_energies_hartree=energy_hartree)
    with (OUTPUT/"spectrum.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["photon_energy_ev", "probability_density_sigma_0.02ev", "probability_density_sigma_0.01ev"])
        writer.writerows(zip(axis, *previous))
    dominant = np.argsort(weights)[-40:][::-1]
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root/"src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode()); digest.update(path.read_bytes())
    report = dict(complete=True, model="common-ground-Eckart-slice double-harmonic model",
        symbols=ground["symbols"], basis=ground["basis"], electronic_model=ground["method"],
        input_hashes=hashes, input_source_python_sha256=ground["source_python_sha256"],
        dynamic_matrix="Excited vibrational reconstruction Le diag(omega_e^2) Le.T; raw Hessian comparison retained",
        coordinate_units="Q: bohr sqrt(electron mass); frequencies: Hartree/hbar with hbar=1",
        convention="Q_final = canonical_rotation @ Q_ground + shift; |det(canonical_rotation)|=1",
        harmonic_model=model, invariance_checks=checks, convergence=convergence,
        raw_probability_sum=float(weights.sum()), raw_first_energy_moment_hartree=moment,
        first_moment_error_hartree=abs(moment-model["analytic_mean_photon_energy_hartree"]),
        zero_zero_probability=float(probability), displacement_scale_derivative=float(derivative),
        displacement_sensitivity_finite_differences=finite,
        sensitivity_scope="Only the fixed model shift is scaled; frequencies/rotation held fixed. No SCF/minimum/mode-response derivative.",
        dominant_lines=[dict(occupations=states[i].tolist(), photon_energy_ev=float(energy_ev[i]), probability=float(weights[i])) for i in dominant],
        source_python_sha256=digest.hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256((root/"src/gradscf_tools/franck_condon.py").read_bytes()).hexdigest(),
        elapsed_seconds=perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__, dtype="float64",
        scope="Zero-temperature unit-Condon probability envelope on a common flat vibrational slice. No absolute dark-state absorption, HT, rotation-vibration coupling or full curvilinear FC claim.")
    (OUTPUT/"gradscf.json").write_text(json.dumps(report, indent=2, default=lambda x: np.asarray(x).tolist())+"\n")
    print("X11 common-slice, covariance, FC normalization/moment, cutoff and 0-0 AD/FD gates PASS", flush=True)


if __name__ == "__main__":
    main()
