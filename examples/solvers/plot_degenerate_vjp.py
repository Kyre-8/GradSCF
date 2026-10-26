"""Plot saved A03 projector VJPs and comparable process measurements.

Run on the measurement host after degenerate_vjp.py. No solver is imported.
Only complete, passing reports with converged, valid, finite, analytically
checked case VJPs enter response curves or performance comparisons. Expected
boundary rejection is a separate state. Missing adjoint diagnostics stay missing.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np


DEFAULT_INPUT = Path("reproducibility/ad_showcase/A03_degenerate_vjp")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
METHODS = {"dense": ("s", COLORS[0]), "davidson": ("o", COLORS[1])}
PROVENANCE = (
    "source_python_sha256", "script_sha256", "settings", "rank", "nparameters",
    "nprobes", "seed", "dtype", "response", "hostname", "python", "jax", "numpy",
    "devices", "threads", "persistent_compilation_cache_enabled",
)


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if np.isfinite(value) else None


def vector(value):
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return out if out.shape == (4,) else None


def cohort(report):
    if any(key not in report or report[key] is None for key in PROVENANCE):
        return None
    if not report["source_python_sha256"] or not report["script_sha256"]:
        return None
    return json.dumps({key: report[key] for key in PROVENANCE}, sort_keys=True)


def case_state(report, row):
    if report.get("complete") is not True or report.get("passed") is not True:
        return "unverified", "report incomplete or not passed"
    if row.get("passed") is not True or row.get("converged") is not True:
        return "unverified", "case not passed or primal not converged"
    gradient = vector(row.get("vjp"))
    if row.get("response_valid") is False:
        if (row.get("expected_boundary_rejection") is True
                and row.get("derivative_nonfinite") is True and gradient is not None
                and not np.isfinite(gradient).all()):
            return "rejected", "expected boundary rejection with nonfinite derivative"
        return "unverified", "invalid response without a verified expected rejection"
    reference = vector(row.get("reference_vjp"))
    if (row.get("response_valid") is not True or gradient is None or reference is None
            or not np.isfinite(gradient).all() or not np.isfinite(reference).all()):
        return "unverified", "missing or nonfinite VJP/reference"
    if not np.allclose(gradient, reference, atol=2e-7, rtol=2e-5):
        return "unverified", "componentwise analytic VJP gate failed"
    recorded_error = number(row.get("vjp_max_error"))
    if recorded_error is None:
        return "unverified", "analytic error not recorded"
    actual_error = float(np.max(np.abs(gradient-reference)))
    if not np.isclose(recorded_error, actual_error, atol=1e-15, rtol=1e-10):
        return "unverified", "recorded analytic error disagrees with saved VJP"
    return "verified", "complete report and analytic VJP gate passed"


def load_records(input_dir):
    records, notes = [], []
    paths = sorted(input_dir.glob("n*.json"))
    if not paths:
        raise ValueError(f"No n*.json reports found in {input_dir}")
    for path in paths:
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            notes.append(f"{path.name}: unreadable report ({error})")
            continue
        if (not isinstance(report, dict) or report.get("method") not in METHODS
                or not isinstance(report.get("dimension"), int)
                or report["dimension"] <= 0 or not isinstance(report.get("rank"), int)
                or report["rank"] <= 0 or report.get("nparameters") != 4
                or not isinstance(report.get("cases"), list)):
            notes.append(f"{path.name}: unsupported or missing report metadata")
            continue
        if not report["cases"]:
            notes.append(f"{path.name}: no recorded cases; complete={report.get('complete')}")
        for index, row in enumerate(report["cases"]):
            if not isinstance(row, dict):
                notes.append(f"{path.name}, case {index}: malformed case")
                continue
            delta, gap = number(row.get("internal_gap")), number(row.get("external_gap"))
            if delta is None or gap is None or delta < 0 or gap <= 0:
                notes.append(f"{path.name}, case {index}: missing or invalid spectral gaps")
                continue
            state, reason = case_state(report, row)
            records.append(dict(path=path, report=report, row=row, index=index,
                                delta=delta, gap=gap, state=state, reason=reason,
                                cohort=cohort(report)))
            if state == "unverified":
                notes.append(f"{path.name}, case {index}: omitted response/performance: {reason}")
    return records, notes, len(paths)


def unique_cases(records):
    selected = {}
    for item in sorted(records, key=lambda r: (-len(r["report"]["cases"]), str(r["path"]))):
        key = (item["report"]["method"], item["delta"], item["gap"])
        selected.setdefault(key, item)
    return list(selected.values())


def empty_panel(ax, message):
    ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color=".35", linespacing=1.5)
    ax.set_xticks([])
    ax.set_yticks([])


def gap_axis(ax, values, *, internal=False):
    ticks = sorted(set(values), reverse=True)
    if not ticks:
        return
    positive = [value for value in ticks if value > 0]
    if internal and positive:
        ax.set_xscale("symlog", linthresh=min(positive)/5, linscale=.7)
    elif not internal:
        ax.set_xscale("log")
    ax.set_xticks(ticks, [f"{value:g}" for value in ticks])
    ax.minorticks_off()
    if len(ticks) > 1:
        ax.invert_xaxis()
    elif ticks[0] == 0:
        ax.set_xlim(.5, -.5)
    else:
        ax.set_xlim(ticks[0]*1.3, ticks[0]/1.3)


def save_figure(fig, output_dir, stem):
    for extension in ("png", "pdf"):
        fig.savefig(output_dir/f"{stem}.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def response_figure(records, output_dir, notes):
    groups = defaultdict(list)
    for item in records:
        key = (item["cohort"] or str(item["path"]), item["report"]["dimension"])
        groups[key].append(item)
    selected = max(groups.values(), key=lambda rows: (
        len({(r["delta"], r["gap"]) for r in rows if r["state"] == "verified"}),
        rows[0]["report"]["dimension"],
        len({r["report"]["method"] for r in rows if r["state"] == "verified"})))
    report = selected[0]["report"]
    gap = max({r["gap"] for r in selected}, key=lambda value: (
        len({r["delta"] for r in selected if r["gap"] == value}), value))
    internal = [r for r in selected if r["gap"] == gap]
    external = [r for r in selected if r["delta"] == 0]
    good_internal = unique_cases([r for r in internal if r["state"] == "verified"])
    good_external = unique_cases([r for r in external if r["state"] == "verified"])
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    fig.subplots_adjust(left=.085, right=.97, top=.87, bottom=.19, hspace=.47, wspace=.29)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    reference_rows = {r["delta"]: r for r in good_internal}
    deltas = sorted(reference_rows, reverse=True)
    for parameter, color in enumerate(COLORS):
        ax_a.plot(deltas, [reference_rows[d]["row"]["reference_vjp"][parameter] for d in deltas],
                  color=color, lw=1.25)
        for method, (marker, _) in METHODS.items():
            rows = sorted((r for r in good_internal if r["report"]["method"] == method),
                          key=lambda r: r["delta"], reverse=True)
            ax_a.plot([r["delta"] for r in rows], [r["row"]["vjp"][parameter] for r in rows],
                      linestyle="none", marker=marker, color=color, mfc="none", ms=5.5)
    if good_internal:
        handles = [Line2D([], [], color=color, label=rf"$\partial L/\partial\theta_{{{i+1}}}$")
                   for i, color in enumerate(COLORS)]
        handles += [Line2D([], [], color=".25", marker=marker, mfc="none", linestyle="none", label=method)
                    for method, (marker, _) in METHODS.items()
                    if any(r["report"]["method"] == method for r in good_internal)]
        ax_a.legend(handles=handles, frameon=False, fontsize=8, ncol=3)
    else:
        empty_panel(ax_a, "No complete, analytically verified\ninternal-splitting scan")
    ax_a.set(title=f"a  Four-parameter projector VJP · exterior gap {gap:g}",
             xlabel=r"Internal spectral width $\delta$ (0 = exact degeneracy)",
             ylabel=r"$\partial L/\partial\theta_a$  (dimensionless)")

    for method, (marker, color) in METHODS.items():
        rows = sorted((r for r in good_internal if r["report"]["method"] == method),
                      key=lambda r: r["delta"], reverse=True)
        if rows:
            errors = [np.max(np.abs(vector(r["row"]["vjp"])-vector(r["row"]["reference_vjp"])))
                      for r in rows]
            ax_b.plot([r["delta"] for r in rows], errors, marker=marker, color=color,
                      mfc="none", label=method)
    ax_b.set_yscale("symlog", linthresh=1e-16)
    ax_b.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=2))
    error_formatter = ScalarFormatter(useMathText=True)
    error_formatter.set_powerlimits((0, 0))
    ax_b.yaxis.set_major_formatter(error_formatter)
    ax_b.set(title="b  Agreement with the analytic projector derivative",
             xlabel=r"Internal spectral width $\delta$", ylabel=r"$\max_a |g_a-g_a^{\rm analytic}|$")
    if good_internal:
        ax_b.legend(frameon=False, fontsize=9)
    else:
        empty_panel(ax_b, "No verified analytic-error records")
    for ax in (ax_a, ax_b):
        gap_axis(ax, [r["delta"] for r in internal], internal=True)

    references = {r["gap"]: r for r in good_external}
    gaps = sorted(references, reverse=True)
    if gaps:
        ax_c.plot(gaps, [np.linalg.norm(vector(references[g]["row"]["reference_vjp"])) for g in gaps],
                  "--", color=".35", lw=1.2, label="analytic")
    for method, (marker, color) in METHODS.items():
        rows = sorted((r for r in good_external if r["report"]["method"] == method),
                      key=lambda r: r["gap"], reverse=True)
        if rows:
            ax_c.plot([r["gap"] for r in rows], [np.linalg.norm(vector(r["row"]["vjp"])) for r in rows],
                      marker=marker, color=color, mfc="none", lw=1.2, label=method)
    ax_c.set_yscale("symlog", linthresh=1e-12)
    ax_c.set(title=r"c  Exterior-gap sensitivity · $\delta=0$",
             xlabel=r"Exterior gap $g$", ylabel=r"$\|\nabla_\theta L\|_2$  (dimensionless)")
    for rejected_gap in sorted({r["gap"] for r in external if r["state"] == "rejected"}):
        ax_c.axvline(rejected_gap, color=".65", ls=":", lw=.8)
        ax_c.plot([rejected_gap], [.06], "x", color="#B2182B", transform=ax_c.get_xaxis_transform(),
                  clip_on=False)
    if good_external:
        ax_c.legend(frameon=False, fontsize=9)
    else:
        empty_panel(ax_c, "No verified exterior-gap responses")

    levels = {"verified": 2, "rejected": 1, "unverified": 0}
    for index, (method, (marker, color)) in enumerate(METHODS.items()):
        rows = [r for r in external if r["report"]["method"] == method]
        for state, level in levels.items():
            chosen = [r for r in rows if r["state"] == state]
            if chosen:
                ax_d.scatter([r["gap"] for r in chosen], [level+(index-.5)*.12]*len(chosen),
                             marker=marker, s=35, facecolors="none", edgecolors=color)
    ax_d.set(title="d  Recorded response status", xlabel=r"Exterior gap $g$",
             yticks=[0, 1, 2], yticklabels=["Unverified / failed", "Expected rejection", "Verified response"],
             ylim=(-.4, 2.4))
    if not external:
        empty_panel(ax_d, "No exact-degeneracy exterior-gap records")
    for ax in (ax_c, ax_d):
        gap_axis(ax, [r["gap"] for r in external])
    for ax in axes.flat:
        ax.grid(axis="y", color=".9", lw=.6)
        ax.tick_params(labelsize=8.5)
    missing_adjoint = sum(r["row"].get("adjoint_residual") is None for r in selected)
    excluded = sum(r["state"] == "unverified" for r in selected)
    source = str(report.get("source_python_sha256", "missing"))[:10]
    fig.suptitle(f"Complete degenerate-subspace response · n = {report['dimension']}, k = {report['rank']}\n"
                 "Internal degeneracy is compatible with a smooth projector; exterior isolation controls response",
                 fontsize=12.5)
    fig.text(.085, .105, "a: lines = analytic, markers = AD. Zero is shown explicitly; no internal-gap regularization.\n"
             "Curves require complete/pass reports, converged/valid/pass cases, finite VJPs, and componentwise\n"
             r"$|g_a-g_a^{\rm analytic}|\leq 2\times10^{-7}+2\times10^{-5}|g_a^{\rm analytic}|$. "
             f"Excluded cases: {excluded}; missing adjoint residuals: {missing_adjoint} (not plotted).",
             fontsize=8, linespacing=1.5, va="top")
    provenance_note = ("matching source/script/settings/environment" if selected[0]["cohort"] is not None
                       else "single report; incomplete provenance prevents comparison")
    fig.text(.085, .035, f"Source {source}; {provenance_note}. "
             f"Other cohorts/dimensions omitted: {len(records)-len(selected)} cases.", fontsize=7.5, color=".35")
    notes.append("Response figure inputs: "+", ".join(sorted({r["path"].name for r in selected})))
    save_figure(fig, output_dir, "degenerate_vjp_response")


def performance_figure(records, output_dir, notes):
    groups, seen = defaultdict(list), set()
    for item in records:
        report, row = item["report"], item["row"]
        if item["state"] != "verified" or item["index"] != 0 or item["cohort"] is None:
            continue
        process = (report.get("hostname"), report.get("process_id"), report.get("started_utc"))
        if not isinstance(process[1], int) or not isinstance(process[2], str) or not process[2]:
            notes.append(f"{item['path'].name}: performance omitted; process provenance missing")
            continue
        timing, memory = row.get("timing_seconds"), row.get("process_peak_rss_mib")
        samples = timing.get("retained_pullback_warm") if isinstance(timing, dict) else None
        rss = memory.get("after_pullback") if isinstance(memory, dict) else None
        if (not isinstance(samples, list) or not samples
                or any(number(value) is None or value <= 0 for value in samples)
                or number(rss) is None or rss <= 0):
            notes.append(f"{item['path'].name}: performance omitted; warm samples/RSS missing or invalid")
            continue
        if process in seen:
            notes.append(f"{item['path'].name}: duplicate process report omitted from performance")
            continue
        seen.add(process)
        key = (item["cohort"], item["delta"], item["gap"], len(samples))
        groups[key].append(item)
    fig, axes = plt.subplots(1, 2, figsize=(10.9, 5.3))
    fig.subplots_adjust(left=.085, right=.975, top=.78, bottom=.25, wspace=.32)
    if not groups:
        for ax in axes:
            empty_panel(ax, "No comparable verified process measurements\n"
                        "Require matching provenance, a common first case,\nand recorded process ID/start time")
        fig.suptitle("A03 process measurements · no eligible data", fontsize=13)
        fig.text(.085, .11, "Dense and Davidson forward methods share the implicit GMRES backward.\n"
                 "Missing timings, memory or provenance are not replaced with zero.", fontsize=9)
        save_figure(fig, output_dir, "degenerate_vjp_performance")
        return
    selected = max(groups.values(), key=lambda rows: (
        len({(r["report"]["dimension"], r["report"]["method"]) for r in rows}),
        len({r["report"]["dimension"] for r in rows}), len(rows)))
    dimensions = sorted({r["report"]["dimension"] for r in selected})
    for index, (method, (marker, color)) in enumerate(METHODS.items()):
        rows = [r for r in selected if r["report"]["method"] == method]
        for ordinal, item in enumerate(rows):
            n = item["report"]["dimension"]
            x = dimensions.index(n)+(index-.5)*.16
            samples = np.asarray(item["row"]["timing_seconds"]["retained_pullback_warm"])
            center = np.median(samples)
            axes[0].errorbar(x, center, yerr=[[center-samples.min()], [samples.max()-center]],
                             fmt=marker, color=color, mfc="none", capsize=3,
                             label=method if ordinal == 0 else None)
            axes[1].plot(x, item["row"]["process_peak_rss_mib"]["after_pullback"],
                         marker=marker, color=color, mfc="none", linestyle="none",
                         label=method if ordinal == 0 else None)
    axes[0].set(title="a  Warm retained pullback", ylabel="Wall time / s", yscale="log")
    rss_max = max(r["row"]["process_peak_rss_mib"]["after_pullback"] for r in selected)
    axes[1].set(title="b  Process high-water RSS after first pullback", ylabel="Peak RSS / MiB",
                ylim=(0, 1.12*rss_max))
    for ax in axes:
        ax.set(xticks=range(len(dimensions)), xticklabels=[str(n) for n in dimensions],
               xlabel="Operator dimension n", xlim=(-.4, len(dimensions)-.6))
        ax.grid(axis="y", color=".9", lw=.6)
        ax.legend(frameon=False, fontsize=9)
    first = selected[0]
    fig.suptitle("Observed process cost · dense and Davidson share the implicit GMRES backward\n"
                 f"Common first case: internal width {first['delta']:g}, exterior gap {first['gap']:g}; "
                 "identical GMRES response settings", fontsize=12)
    fig.text(.085, .15, "Each point is an independent process; warm timing = median, whiskers = min/max.\n"
             "RSS is the first-case high-water mark after preparation and the first pullback, before FD/gauge checks.\n"
             "It includes imports and forward preparation; it is not backward-only allocated memory.\n"
             "Only matching source/script/settings/environment and validated responses are compared; missing points are omitted.",
             fontsize=8, linespacing=1.5, va="top")
    notes.append("Performance figure inputs: "+", ".join(sorted({r["path"].name for r in selected})))
    notes.append(f"Performance omitted {sum(len(rows) for rows in groups.values())-len(selected)} "
                 "otherwise eligible reports from unmatched cohorts/first cases.")
    save_figure(fig, output_dir, "degenerate_vjp_performance")


def run(input_dir=DEFAULT_INPUT, *, output_dir=None):
    """Run this example with ordinary Python parameters; no command-line parser."""
    args = SimpleNamespace(input_dir=Path(input_dir), output_dir=None if output_dir is None else Path(output_dir))
    try:
        records, notes, count = load_records(args.input_dir)
    except ValueError as error:
        raise ValueError(str(error))
    if not records:
        raise ValueError("No usable recorded cases. "+"; ".join(notes))
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.titlelocation": "left",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    response_figure(records, output_dir, notes)
    performance_figure(records, output_dir, notes)
    print(f"Read {count} reports; saved degenerate_vjp_response and degenerate_vjp_performance as PNG/PDF in {output_dir}")
    for note in notes:
        print(note)


if __name__ == "__main__":
    run()
