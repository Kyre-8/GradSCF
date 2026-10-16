"""Read-only validation of the fixed-reference water forward calculations.

All tolerances were fixed before the sweep. A small virtual window may be
inadequate even when every nonlinear and eigenvalue equation converges.
"""
import json
from pathlib import Path

import numpy as np

from water_forward_convergence import ACCEPTED, NMO, NOCC, OUTPUT
from gradscf_tools.excited import select_tracked_root, transition_density_overlaps


def maximum_difference(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def check_baseline():
    accepted = json.loads(ACCEPTED.read_text())
    actual = json.loads((OUTPUT / "nw200/result.json").read_text())
    errors = dict(qp_energy_hartree=maximum_difference(
                      actual["qp_energy_hartree"], accepted["qp_energy_hartree"]),
                  qp_weight=maximum_difference(actual["qp_weight"], accepted["qp_weight"]))
    gates = dict(qp_energy_hartree=1e-8, qp_weight=1e-6)
    for label, computed, reference in zip(("tda", "full"), actual["optical"], accepted["optical"]):
        errors[f"{label}_energy_hartree"] = maximum_difference(
            computed["energy_hartree"], reference["excitation_hartree"])
        errors[f"{label}_strength"] = maximum_difference(
            computed["strengths"], reference["oscillator_strength"])
        gates[f"{label}_energy_hartree"] = 1e-8
        gates[f"{label}_strength"] = 1e-7
    report = dict(errors=errors, gates=gates,
                  passed=all(errors[key] <= limit for key, limit in gates.items()))
    (OUTPUT / "baseline_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    assert report["passed"], "Preserve and diagnose a changed baseline before sweeping."
    return report


def compare_points(folder, reference):
    left = json.loads((folder / "result.json").read_text())
    right = json.loads((reference / "result.json").read_text())
    errors = dict(qp_energy_hartree=maximum_difference(
                      left["qp_energy_hartree"], right["qp_energy_hartree"]),
                  qp_weight=maximum_difference(left["qp_weight"], right["qp_weight"]))
    for label in ("tda", "full"):
        with np.load(folder / f"{label}.npz") as a, np.load(reference / f"{label}.npz") as b:
            errors[f"{label}_energy_hartree"] = maximum_difference(a["energy"], b["energy"])
            errors[f"{label}_strength"] = maximum_difference(a["strengths"], b["strengths"])
            errors[f"{label}_static_alpha"] = maximum_difference(a["static_alpha"], b["static_alpha"])
            errors[f"{label}_spectral_relative_l1"] = float(
                np.trapezoid(np.abs(a["cross_section"] - b["cross_section"]), a["grid_ev"])
                / np.trapezoid(np.abs(b["cross_section"]), b["grid_ev"]))
    return errors


def check_quadrature():
    rows = {str(nw): compare_points(OUTPUT / f"nw{nw}", OUTPUT / "nw400")
            for nw in (50, 100, 200)}
    gates = dict(qp_energy_hartree=1e-6, qp_weight=1e-5)
    for label in ("tda", "full"):
        gates.update({f"{label}_energy_hartree": 1e-6, f"{label}_strength": 1e-5,
                      f"{label}_static_alpha": 1e-4, f"{label}_spectral_relative_l1": 1e-4})
    passed = all(rows["200"][key] <= limit for key, limit in gates.items())
    report = dict(reference_nw=400, errors=rows, gates=gates, passed=passed)
    (OUTPUT / "quadrature_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    assert passed, "The declared 200-to-400-point quadrature gate failed."
    return report


def check_representations(resources=Path("artifacts")):
    """Compare identical six-state spaces; RSS includes each whole fresh process."""
    folder = OUTPUT / "representations"
    rows = []
    for label in ("tda", "full"):
        reference = json.loads((folder / f"{label}_dense_direct.json").read_text())
        for solver in ("dense", "davidson"):
            for screening in ("direct", "gmres"):
                name = f"{label}_{solver}_{screening}"
                result = json.loads((folder / f"{name}.json").read_text())
                errors = dict(energy_hartree=maximum_difference(
                    result["energy_hartree"], reference["energy_hartree"]),
                    strength=maximum_difference(result["strength"], reference["strength"]))
                rss_line = next(line for line in (resources / f"{name}.resources").read_text().splitlines()
                                if "Maximum resident set size" in line)
                rows.append(dict(name=name, errors=errors,
                    cold_seconds=result["cold_seconds"],
                    warm_median_seconds=float(np.median(result["warm_seconds"])),
                    process_rss_kib=int(rss_line.rsplit(":", 1)[1]),
                    xla_temporary_bytes=result["xla_temporary_bytes"],
                    bse_residual_hartree=max(result["recomputed_residual_hartree"]),
                    screening_relative_residual=max(result["screening_relative_residual"]),
                    passed=result["passed"] and errors["energy_hartree"] <= 2e-8
                           and errors["strength"] <= 2e-7))
    report = dict(rows=rows, energy_gate_hartree=2e-8, strength_gate=2e-7,
                  passed=all(row["passed"] for row in rows))
    (OUTPUT / "representation_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    assert report["passed"]
    return report


def tracked_optical_errors(folder, label):
    """Embed truncated amplitudes and use the shared state-tracking criteria."""
    with np.load(OUTPUT / f"nw200/{label}.npz") as stored:
        reference = dict(stored)
    with np.load(folder / f"{label}.npz") as stored:
        actual = dict(stored)
    amplitudes = actual["x"] + actual["y"]
    embedded = np.zeros((len(amplitudes), NOCC, NMO - NOCC))
    embedded[:, :, :amplitudes.shape[-1]] = amplitudes
    frame = np.eye(NMO)
    matches = []
    for root, amplitude in enumerate((reference["x"] + reference["y"])[:6]):
        scores = transition_density_overlaps(frame, amplitude, frame, embedded, frame, nocc=NOCC)
        try:
            match = select_tracked_root(scores, actual["energy"])
            selected = match["root"]
            match.update(reference_root=root,
                energy_error_hartree=float(abs(actual["energy"][selected] - reference["energy"][root])),
                strength_error=float(abs(actual["strengths"][selected] - reference["strengths"][root])))
        except RuntimeError as error:
            match = dict(reference_root=root, rejected=str(error),
                         best_overlap=float(np.max(scores)))
        matches.append(match)
    roots = [match["root"] for match in matches if "root" in match]
    tracked = len(roots) == 6 and len(set(roots)) == 6
    alpha = maximum_difference(actual["static_alpha"], reference["static_alpha"])
    spectral = float(np.trapezoid(np.abs(actual["cross_section"] - reference["cross_section"]),
                                 reference["grid_ev"])
                     / np.trapezoid(reference["cross_section"], reference["grid_ev"]))
    energy_errors = [match["energy_error_hartree"] for match in matches if "root" in match]
    strength_errors = [match["strength_error"] for match in matches if "root" in match]
    return dict(matches=matches, all_six_tracked=tracked,
                matched_energy_error_hartree=max(energy_errors) if energy_errors else None,
                matched_strength_error=max(strength_errors) if strength_errors else None,
                static_alpha_error=alpha, spectral_relative_l1=spectral)


def check_windows():
    """A failed accuracy target rejects a truncation, not a converged implementation."""
    reference = json.loads((OUTPUT / "nw200/result.json").read_text())
    rows = []
    for window in ("g", "w", "optical"):
        for count in (4, 8, 12, 14):
            # Count14 is exactly the unchanged full-window baseline, reused once.
            folder = OUTPUT / ("nw200" if count == 14 else f"{window}_virtual{count}")
            if not (folder / "result.json").exists():
                rows.append(dict(window=window, count=count, status="no_converged_result",
                                 source=str(folder)))
                continue
            result = json.loads((folder / "result.json").read_text())
            qe = maximum_difference(result["qp_energy_hartree"], reference["qp_energy_hartree"])
            qz = maximum_difference(result["qp_weight"], reference["qp_weight"])
            optical = {label: tracked_optical_errors(folder, label) for label in ("tda", "full")}
            accurate = qe <= 1e-6 and qz <= 1e-5
            for values in optical.values():
                accurate &= (values["all_six_tracked"]
                    and values["matched_energy_error_hartree"] <= 1e-6
                    and values["matched_strength_error"] <= 1e-5
                    and values["static_alpha_error"] <= 1e-4
                    and values["spectral_relative_l1"] <= 1e-4)
            rows.append(dict(window=window, count=count, source=str(folder),
                status="converged_local_roots", qp_energy_error_hartree=qe,
                qp_weight_error=qz, optical=optical, passes_truncation_targets=bool(accurate)))
    report = dict(rows=rows, qp_scope="Recorded native anchors define local root solves; positive Z is not a global main-peak certificate.",
        endpoint_scope="Full19MO reference within one fixed Gaussian basis; no basis-convergence claim.")
    (OUTPUT / "window_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row["window"], row["count"], row["status"],
              row.get("passes_truncation_targets"), flush=True)
    assert all(row.get("passes_truncation_targets") for row in rows if row["count"] == 14)
    return report


if __name__ == "__main__":
    check_baseline()

# Measured independent baseline comparison on jy_gpu_1:
# Max QP energy difference: 5.53e-12 Eh; Z difference: 5.43e-10.
# Max TDA/full-BSE energy differences: 8.09e-12 / 8.11e-12 Eh.
# All quadrature and representation gates pass. All window solves converge,
# but none of 4/8/12 virtuals meets all declared truncation accuracy targets.
# All six tracked optical states pass; minimum recorded overlap is 0.977545.
