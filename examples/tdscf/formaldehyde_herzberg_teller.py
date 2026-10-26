"""X12: linear Herzberg-Teller activation of the nearly dark formaldehyde S1.

Reuse the frozen X11 common-ground-Eckart-slice harmonic model and X07's
complete SCF/CIS transition-moment nuclear response at the same ground minimum.
All Cartesian dipole components are included. FC, HT and signed interference
are reported separately; the physical total is the squared summed amplitude.
No CLI, new electronic calculation, frozen-MO approximation, or new FC solver.
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
import numpy as np
from scipy.integrate import trapezoid
from scipy.spatial.transform import Rotation

jax.config.update("jax_enable_x64", True)

from gradscf_tools.franck_condon import ground_state_coordinate_overlaps
from formaldehyde_franck_condon import common_model, load_inputs, HARTREE_TO_EV


BASE = Path("reproducibility/ad_showcase")
OUTPUT = BASE/"X12_formaldehyde_herzberg_teller"
MAX_QUANTA = 30
CUTOFFS = (4, 8, 12, 16, 20, 24, 28, 30)
COMPONENTS = ("fc", "ht", "interference", "total")


def load_model_and_response():
    ground, excited, hashes = load_inputs()
    fc_path = BASE/"X11_formaldehyde_franck_condon/gradscf.json"
    electronic_path = BASE/"X07_formaldehyde_oscillator_response/gradscf.json"
    fc = json.loads(fc_path.read_text())
    electronic = json.loads(electronic_path.read_text())
    reference_path = electronic_path.with_name("pyscf.json")
    reference = json.loads(reference_path.read_text())
    assert fc["complete"] and electronic["complete"] and reference["passed"]
    assert reference["native_sha256"] == hashlib.sha256(electronic_path.read_bytes()).hexdigest()
    assert fc["input_hashes"] == hashes
    assert electronic["input_sha256"]["X10_formaldehyde_cis_emission/emission.json"] == hashes["emission.json"]
    assert electronic["basis"] == ground["basis"] and electronic["symbols"] == ground["symbols"]
    assert electronic["mode_coordinate_units"].startswith("Bohr sqrt(amu)")
    np.testing.assert_allclose(electronic["coordinates_bohr"], ground["ground_coordinates_bohr"], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(electronic["masses_amu"], ground["masses_amu"])
    np.testing.assert_array_equal(electronic["frequencies_cm1"], ground["ground_frequencies_cm1"])
    mass_ratio = excited["amu_to_electron_mass"]
    modes = np.asarray(ground["ground_displacements"])
    converted_modes = np.asarray(electronic["modes_bohr_per_Q"])/np.sqrt(mass_ratio)
    np.testing.assert_allclose(converted_modes, modes, atol=1e-13, rtol=0)
    selected = electronic["selected_roots"].index(excited["selected_root"])
    assert excited["selected_root"] == 0
    dipole = np.asarray(electronic["values"])[selected, :3]
    cartesian = np.asarray(electronic["cartesian_jacobians"])[selected, :3]
    # Apply the amu -> electron-mass conversion exactly once. A separate
    # Cartesian contraction checks the stored mode derivative and its phase.
    derivatives = np.asarray(electronic["mode_derivatives"])[selected, :3]/np.sqrt(mass_ratio)
    contracted = np.einsum("xai,aim->xm", cartesian, modes)
    np.testing.assert_allclose(derivatives, contracted, atol=1e-13, rtol=0)
    omega = electronic["excitation_energies_hartree"][excited["selected_root"]]
    np.testing.assert_allclose(omega, ground["gaps_hartree"]["vertical_absorption"], atol=1e-8, rtol=0)
    baseline_strength = electronic["values"][selected][3]
    np.testing.assert_allclose((2/3)*omega*np.dot(dipole, dipole), baseline_strength, atol=1e-40, rtol=1e-12)
    model = {key: np.asarray(value) if isinstance(value, list) else value
             for key, value in fc["harmonic_model"].items()}
    rebuilt = common_model(ground, excited)
    for key in ("curvature", "linear", "constant", "minimum_offset", "final_frequencies_hartree", "model_zero_zero_hartree"):
        np.testing.assert_allclose(rebuilt[key], model[key], atol=1e-12, rtol=0)
    metadata = dict(
        coordinate_max_error_bohr=float(np.max(abs(np.asarray(electronic["coordinates_bohr"])-ground["ground_coordinates_bohr"]))),
        converted_mode_max_error=float(np.max(abs(converted_modes-modes))),
        cartesian_contraction_max_error=float(np.max(abs(derivatives-contracted))),
        input_mode_coordinate_units=electronic["mode_coordinate_units"],
        derivative_units="(e bohr)/(bohr sqrt(electron mass))", amu_to_electron_mass=mass_ratio,
        original_electronic_oscillator_strength=baseline_strength,
        input_hashes={"X07/gradscf.json": hashlib.sha256(electronic_path.read_bytes()).hexdigest(),
                      "X07/pyscf.json": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                      "X11/gradscf.json": hashlib.sha256(fc_path.read_bytes()).hexdigest(), **hashes},
        electronic_source_python_sha256=electronic["source_python_sha256"],
        harmonic_source_python_sha256=fc["source_python_sha256"],
        electronic_response="Full relaxed HF orbital, CIS eigenpair and moving AO dipole response from validated X07",
        phase="Signed dipole and derivatives share the same X07 baseline; harmonic surfaces carry no electronic phase.")
    return ground, excited, model, dipole, derivatives, cartesian, metadata


def amplitudes_and_strengths(overlap, coordinate, dipole, derivatives):
    fc = overlap[:, None]*dipole
    ht = coordinate@derivatives.T
    total = fc+ht
    strengths = dict(fc=np.sum(fc**2, axis=1), ht=np.sum(ht**2, axis=1),
        interference=2*np.sum(fc*ht, axis=1), total=np.sum(total**2, axis=1))
    np.testing.assert_allclose(strengths["total"], strengths["fc"]+strengths["ht"]+strengths["interference"],
                               atol=1e-30, rtol=3e-13)
    assert np.isfinite(total).all() and np.all(strengths["total"] >= 0)
    return fc, ht, total, strengths


def molecular_gauge_checks(ground, excited, model, dipole, derivatives, cartesian):
    states, overlap, coordinate = ground_state_coordinate_overlaps(model["initial_frequencies_hartree"],
        model["final_frequencies_hartree"], model["canonical_rotation"], model["shift"], max_quanta=6)
    _, _, original, original_parts = amplitudes_and_strengths(overlap, coordinate, dipole, derivatives)
    original_tensor = original[:, :, None]*original[:, None, :]
    original_energies = model["model_zero_zero_hartree"]+states@model["final_frequencies_hartree"]
    records = []
    for operation in ("electronic_phase", "original_mode_signs_and_permutations", "independent_endpoint_rigid_frames"):
        g, e = deepcopy(ground), deepcopy(excited)
        current_dipole, current_cartesian = dipole.copy(), cartesian.copy()
        expected_tensor = original_tensor
        if operation == "electronic_phase":
            current_dipole, current_cartesian = -dipole, -cartesian
        elif operation == "original_mode_signs_and_permutations":
            permutation, signs = np.array([3, 0, 5, 2, 1, 4]), np.array([1., -1., 1., -1., -1., 1.])
            g["ground_displacements"] = np.asarray(g["ground_displacements"])[..., permutation]*signs
            g["ground_frequencies_cm1"] = np.asarray(g["ground_frequencies_cm1"])[permutation]
            e["displacements"] = np.asarray(e["displacements"])[..., permutation]*signs
            e["frequencies_cm1"] = np.asarray(e["frequencies_cm1"])[permutation]
        else:
            for which, rotvec, translation in (("ground", [.2, -.3, .1], [.4, -.2, .3]),
                                               ("excited", [-.1, .4, .2], [-.3, .6, .1])):
                rotation = Rotation.from_rotvec(rotvec).as_matrix()
                key = f"{which}_coordinates_bohr"
                g[key] = np.asarray(g[key])@rotation+translation
                if which == "ground":
                    g["ground_displacements"] = np.einsum("aim,ij->ajm", np.asarray(g["ground_displacements"]), rotation)
                    current_dipole = rotation.T@dipole
                    current_cartesian = np.einsum("px,pai,ij->xaj", rotation, cartesian, rotation)
                    expected_tensor = np.einsum("pa,npq,qb->nab", rotation, original_tensor, rotation)
                else:
                    e["displacements"] = np.einsum("aim,ij->ajm", np.asarray(e["displacements"]), rotation)
                    block = np.kron(np.eye(4), rotation.T)
                    e["hessian_hartree_per_bohr2"] = block@np.asarray(e["hessian_hartree_per_bohr2"])@block.T
        changed = common_model(g, e)
        current_derivatives = np.einsum("xai,aim->xm", current_cartesian, np.asarray(g["ground_displacements"]))
        new_states, s, q = ground_state_coordinate_overlaps(changed["initial_frequencies_hartree"],
            changed["final_frequencies_hartree"], changed["canonical_rotation"], changed["shift"], max_quanta=6)
        _, _, moments, parts = amplitudes_and_strengths(s, q, current_dipole, current_derivatives)
        np.testing.assert_array_equal(new_states, states)
        tensor = moments[:, :, None]*moments[:, None, :]
        np.testing.assert_allclose(tensor, expected_tensor, atol=2e-13, rtol=1e-9)
        errors = {key: float(np.max(abs(parts[key]-original_parts[key]))) for key in COMPONENTS}
        for key in COMPONENTS:
            np.testing.assert_allclose(parts[key], original_parts[key], atol=2e-13, rtol=1e-9)
        energies = changed["model_zero_zero_hartree"]+states@changed["final_frequencies_hartree"]
        np.testing.assert_allclose(energies, original_energies, atol=1e-12, rtol=0)
        records.append(dict(operation=operation, max_quanta=6, maximum_component_strength_errors=errors,
            polarization_tensor_error=float(np.max(abs(tensor-expected_tensor))),
            photon_energy_error_hartree=float(np.max(abs(energies-original_energies)))))
    return records


def broaden_components(axis, energies, components, sigma):
    # Select by each component's absolute L1 weight, including signed
    # interference. Their union preserves every component without rescaling.
    keep = np.zeros(len(energies), dtype=bool)
    for key in COMPONENTS:
        weights = components[key]
        norm = float(np.sum(abs(weights)))
        if norm:
            order = np.argsort(abs(weights))[::-1]
            count = min(len(order), int(np.searchsorted(np.cumsum(abs(weights[order])), norm*(1-1e-12)))+1)
            keep[order[:count]] = True
    selected = np.flatnonzero(keep)
    omitted = {key: float(np.sum(abs(components[key][~keep]))) for key in COMPONENTS}
    curves = np.zeros((len(axis), len(COMPONENTS)))
    weights = np.column_stack([components[key] for key in COMPONENTS])
    for start in range(0, len(selected), 256):
        indices = selected[start:start+256]
        delta = (axis[:, None]-energies[indices][None, :])/sigma
        curves += np.exp(-.5*delta**2)@weights[indices]/(np.sqrt(2*np.pi)*sigma)
    return curves, omitted


def main():
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/"gradscf.json").write_text('{"complete": false}\n')
    ground, excited, model, dipole, derivatives, cartesian, input_checks = load_model_and_response()
    gauges = molecular_gauge_checks(ground, excited, model, dipole, derivatives, cartesian)
    print("X12 input units and combined molecular gauge gates PASS", input_checks["cartesian_contraction_max_error"], flush=True)
    print("Building FC and initial-coordinate overlaps through total quanta", MAX_QUANTA, flush=True)
    states, overlap, coordinate = ground_state_coordinate_overlaps(model["initial_frequencies_hartree"],
        model["final_frequencies_hartree"], model["canonical_rotation"], model["shift"], max_quanta=MAX_QUANTA)
    # Check the extension against the frozen X11 raw overlaps, including their
    # state ordering and signed vibrational phase convention.
    original_path = BASE/"X11_formaldehyde_franck_condon/franck_condon_states.npz"
    original_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()
    original_manifest = json.loads(original_path.with_name("verification_manifest.json").read_text())
    assert original_hash == original_manifest["remote_raw_artifact"]["sha256"]
    with np.load(original_path) as original:
        np.testing.assert_array_equal(states, original["occupations"])
        np.testing.assert_allclose(overlap, original["amplitudes"], atol=1e-14, rtol=1e-12)
    fc, ht, moments, strengths = amplitudes_and_strengths(overlap, coordinate, dipole, derivatives)
    omega = np.asarray(model["initial_frequencies_hartree"])
    expected_tensor = np.outer(dipole, dipole)+(derivatives/(2*omega))@derivatives.T
    expected_closure = float(np.trace(expected_tensor))
    measured_tensor = moments.T@moments
    np.testing.assert_allclose(measured_tensor, expected_tensor, atol=1e-14, rtol=2e-8)
    np.testing.assert_allclose(strengths["total"].sum(), expected_closure, atol=1e-14, rtol=2e-8)
    scaled_coordinate_closure = np.sqrt(2*omega)[:, None]*(coordinate.T@coordinate)*np.sqrt(2*omega)[None, :]
    np.testing.assert_allclose(scaled_coordinate_closure, np.eye(6), atol=5e-8, rtol=0)
    mean_coordinate = (overlap@coordinate)*np.sqrt(2*omega)
    np.testing.assert_allclose(mean_coordinate, 0., atol=3e-8, rtol=0)
    np.testing.assert_allclose(strengths["fc"].sum()/np.dot(dipole, dipole), 1., atol=1e-8, rtol=0)
    photon = model["model_zero_zero_hartree"]+states@model["final_frequencies_hartree"]
    assert np.all(photon > 0)
    oscillator = {key: (2/3)*photon*strengths[key] for key in COMPONENTS}
    energy_ev = photon*HARTREE_TO_EV
    axis = np.linspace(model["model_zero_zero_hartree"]*HARTREE_TO_EV-.15,
                       model["model_zero_zero_hartree"]*HARTREE_TO_EV+1.7, 2501)
    total_quanta = states.sum(axis=1)
    convergence, previous, previous_f = [], None, None
    for cutoff in CUTOFFS:
        selected = total_quanta <= cutoff
        parts = {key: oscillator[key][selected] for key in COMPONENTS}
        current = [broaden_components(axis, energy_ev[selected], parts, sigma) for sigma in (.02, .01)]
        curves = [item[0] for item in current]
        profile_changes = None if previous is None else [float(trapezoid(abs(a[:, 3]-b[:, 3]), axis)/trapezoid(a[:, 3], axis))
                                                         for a, b in zip(curves, previous)]
        f_sum = float(parts["total"].sum())
        closure = float(strengths["total"][selected].sum())
        convergence.append(dict(max_quanta=cutoff, states=int(selected.sum()),
            raw_dipole_strength_sums={key: float(strengths[key][selected].sum()) for key in COMPONENTS},
            raw_oscillator_strength_sums={key: float(parts[key].sum()) for key in COMPONENTS},
            closure_remainder=expected_closure-closure, relative_closure_remainder=(expected_closure-closure)/expected_closure,
            relative_total_f_change=None if previous_f is None else abs(f_sum-previous_f)/f_sum,
            relative_total_profile_l1_changes=profile_changes,
            gaussian_sigma_ev=[.02, .01], plot_omitted_absolute_f=[item[1] for item in current],
            finite_window_integrated_f=[{key: float(trapezoid(curve[:, index], axis)) for index, key in enumerate(COMPONENTS)} for curve in curves]))
        previous, previous_f = curves, f_sum
        print("X12 cutoff", cutoff, "relative closure remainder", convergence[-1]["relative_closure_remainder"],
              "sum f", f_sum, "profile change", profile_changes, flush=True)
    assert abs(convergence[-1]["relative_closure_remainder"]) < 2e-8
    assert convergence[-1]["relative_total_f_change"] < 2e-7
    assert max(convergence[-1]["relative_total_profile_l1_changes"]) < 1e-5
    with (OUTPUT/"spectrum.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["photon_energy_ev"]+[f"{key}_f_density_sigma_{sigma}ev" for sigma in (.02, .01) for key in COMPONENTS])
        writer.writerows(np.column_stack([axis, *previous]))
    np.savez_compressed(OUTPUT/"herzberg_teller_lines.npz", photon_energies_hartree=photon,
                        **{f"{key}_oscillator_strength": value for key, value in oscillator.items()})
    dominant = np.argsort(oscillator["total"])[-40:][::-1]
    root = Path(__file__).resolve().parents[2]
    source = hashlib.sha256()
    for path in sorted((root/"src").rglob("*.py")):
        source.update(path.relative_to(root).as_posix().encode()); source.update(path.read_bytes())
    report = dict(complete=True,
        model="Frozen X11 common-ground-Eckart-slice double-harmonic surfaces plus linear transition dipole at Rg",
        symbols=ground["symbols"], basis=ground["basis"], selected_root=0, harmonic_model=model,
        input_checks=input_checks, molecular_gauge_checks=gauges, dipole_au=dipole, derivatives_mu_per_Qme=derivatives,
        dipole_strength_units="(e bohr)^2", oscillator_strength_units="dimensionless; (2/3)*positive photon energy*sum_xyz |M|^2",
        initial_zero_point_rms_Q_amu=1/np.sqrt(2*omega*excited["amu_to_electron_mass"]),
        electronic_dipole_norm_squared=float(np.dot(dipole, dipole)), analytic_dipole_strength_closure=expected_closure,
        closure_polarization_tensor=expected_tensor, measured_polarization_tensor=measured_tensor,
        normalized_coordinate_closure_max_error=float(np.max(abs(scaled_coordinate_closure-np.eye(6)))),
        normalized_coordinate_mean_max_error=float(np.max(abs(mean_coordinate))),
        mode_HT_closure_contributions=np.sum(derivatives**2/(2*omega), axis=0), convergence=convergence,
        raw_X11_state_file_sha256=original_hash,
        raw_line_order="Identical occupations/order to the hash-bound frozen X11 raw state file; no state/component normalization.",
        dominant_lines=[dict(occupations=states[i], photon_energy_ev=float(energy_ev[i]),
            dipole_strengths={key: float(value[i]) for key, value in strengths.items()},
            oscillator_strengths={key: float(value[i]) for key, value in oscillator.items()},
            fc_moment_au=fc[i], ht_moment_au=ht[i], total_moment_au=moments[i]) for i in dominant],
        source_python_sha256=source.hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256((root/"src/gradscf_tools/franck_condon.py").read_bytes()).hexdigest(),
        elapsed_seconds=perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__, dtype="float64",
        scope="One local well, zero-temperature double-harmonic/linear-HT model using validated full electronic nuclear derivatives. No anharmonic/tunneling or experimental prediction; signed interference is not a negative absorption spectrum.")
    (OUTPUT/"gradscf.json").write_text(json.dumps(report, indent=2, default=lambda x: np.asarray(x).tolist())+"\n")
    print("X12 molecular phase/frame/units, closure, cutoff and FC/HT/interference gates PASS", flush=True)


if __name__ == "__main__":
    main()
