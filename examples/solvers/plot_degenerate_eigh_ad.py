"""Plot ordinary/shared projector VJP accuracy and explicit nonfinite status.

Run on the measurement host. This script imports no eigensolver or JAX.
NaN derivatives are status observations, never zero or finite error points.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import numpy as np


DEFAULT_INPUT = Path("reproducibility/ad_showcase/A03_degenerate_vjp/plain_eigh_comparison.json")
COLORS = {"ordinary": "#D55E00", "shared": "#0072B2", "nonfinite": "#B2182B"}


def checked_cases(data):
    if not all(data.get(key) is True for key in ("complete", "comparison_credible", "passed")):
        raise ValueError("Require a complete comparison with passing shared analytic-reference gates")
    cases = data.get("cases", [])
    seen = set()
    for row in cases:
        key = (row["dimension"], row["internal_width"])
        if key in seen:
            raise ValueError(f"Duplicate dimension/splitting record: {key}")
        seen.add(key)
        if row.get("comparison_credible") is not True or row.get("rank") != 4:
            raise ValueError("Every plotted case must have a credible complete rank-four response")
        if not np.isclose(row["exterior_gap"], .7, atol=1e-15, rtol=0):
            raise ValueError("This figure requires the controlled exterior gap 0.7")
        reference = np.asarray(row["reference_vjp"], dtype=float)
        if reference.shape != (4,) or not np.isfinite(reference).all():
            raise ValueError("Missing finite four-component analytic reference")
        for method in ("ordinary", "shared"):
            result = row[method]
            gradient = np.asarray(result.get("vjp"), dtype=float)
            if gradient.shape != (4,):
                raise ValueError(f"Missing four-component {method} VJP at {key}")
            finite = bool(np.isfinite(gradient).all())
            if result.get("finite") is not finite:
                raise ValueError(f"Inconsistent {method} finite status at {key}")
            recorded = result.get("max_absolute_error")
            if finite:
                actual = float(np.max(np.abs(gradient-reference)))
                if recorded is None or not np.isfinite(recorded) or not np.isclose(recorded, actual, atol=1e-18, rtol=1e-10):
                    raise ValueError(f"Saved {method} error does not match its VJP at {key}")
            elif recorded is not None:
                raise ValueError("Nonfinite VJPs must not carry a finite maximum-error value")
            if method == "shared" and (not finite or result.get("within_tolerance") is not True
                    or result.get("converged") is not True or result.get("response_valid") is not True):
                raise ValueError("Shared response gate is missing or failed")
        ordinary = row["ordinary"]
        eigenvalues = np.asarray(ordinary["raw_values"], dtype=float)
        raw_gaps = np.asarray(ordinary["raw_internal_gaps"], dtype=float)
        if (eigenvalues.shape != (5,) or raw_gaps.shape != (3,) or not np.isfinite(eigenvalues).all()
                or not np.allclose(raw_gaps, np.diff(eigenvalues[:4]), atol=1e-18, rtol=1e-10)
                or ordinary["raw_min_internal_gap"] != float(raw_gaps.min())):
            raise ValueError(f"Raw internal-gap diagnostics disagree at {key}")
    if not cases:
        raise ValueError("No comparison cases")
    return cases


def run(input_path=DEFAULT_INPUT, *, output_dir=None):
    """Run this example with ordinary Python parameters; no command-line parser."""
    args = SimpleNamespace(input=Path(input_path), output_dir=None if output_dir is None else Path(output_dir))
    data = json.loads(args.input.read_text())
    cases = checked_cases(data)
    dimensions = sorted({row["dimension"] for row in cases})
    if dimensions != [32, 256]:
        raise ValueError("The two-panel figure requires n=32 and n=256")
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(10.3, 6.6))
    grid = fig.add_gridspec(2, 2, height_ratios=(4.3, .65), left=.09, right=.97,
                          top=.76, bottom=.25, hspace=.07, wspace=.22)
    positive_errors = [row[method]["max_absolute_error"] for row in cases for method in ("ordinary", "shared")
                       if row[method]["finite"] and row[method]["max_absolute_error"] > 0]
    lower = 10.**np.floor(np.log10(min(positive_errors)))
    upper = 10.**np.ceil(np.log10(max(positive_errors)))
    exact_notes = []
    for column, dimension in enumerate(dimensions):
        rows = sorted((row for row in cases if row["dimension"] == dimension),
                      key=lambda row: row["internal_width"], reverse=True)
        widths = [row["internal_width"] for row in rows]
        if widths != [.1, 1e-3, 1e-6, 1e-9, 0.]:
            raise ValueError("Both dimensions must contain the complete five-point internal-width scan")
        ax = fig.add_subplot(grid[0, column])
        status = fig.add_subplot(grid[1, column], sharex=ax)
        for method, marker in (("ordinary", "o"), ("shared", "s")):
            errors = [row[method]["max_absolute_error"] if row[method]["finite"]
                      and row[method]["max_absolute_error"] > 0 else np.nan for row in rows]
            ax.plot(widths, errors, marker=marker, ms=5, mfc="white", mew=1.2,
                    color=COLORS[method], lw=1.4 if method == "ordinary" else 1., zorder=3)
            for row in rows:
                if row[method]["finite"] and row[method]["max_absolute_error"] == 0:
                    print(f"n={dimension}, delta={row['internal_width']}: {method} exact zero error omitted from logarithmic axis")
        ax.set(title=f"{'ab'[column]}  n = {dimension}", yscale="log", ylim=(lower*.6, upper*2.))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.grid(axis="y", color=".89", lw=.7)
        ax.tick_params(axis="x", labelbottom=False, length=0)
        if column == 0:
            ax.set_ylabel(r"Maximum VJP error  $\max_a |g_a-g_a^{\rm analytic}|$")
        status.set_xscale("symlog", linthresh=2e-10, linscale=.65)
        status.set_xlim(.19, -1.2e-10)
        status.set_xticks(widths, [r"$10^{-1}$", r"$10^{-3}$", r"$10^{-6}$", r"$10^{-9}$", "0"])
        status.set_ylim(-.7, 1.)
        status.set_yticks([])
        status.spines["left"].set_visible(False)
        status.spines["bottom"].set_color(".65")
        status.set_xlabel(r"Internal spectral width $\delta$  (0 = exact degeneracy)", labelpad=7)
        for row in rows:
            width, result = row["internal_width"], row["ordinary"]
            if result["finite"]:
                status.plot(width, 0., "o", color=COLORS["ordinary"], ms=3.5)
            else:
                status.plot(width, 0., "x", color=COLORS["nonfinite"], ms=7, mew=1.7)
                status.annotate("NaN", (width, 0.), xytext=(0, 9), textcoords="offset points",
                                ha="center", color=COLORS["nonfinite"], fontsize=9)
        status.text(.02, .88, "Ordinary AD status", transform=status.transAxes, fontsize=8, color=".35")
        exact = next(row for row in rows if row["internal_width"] == 0.)["ordinary"]
        exact_notes.append(f"n={dimension}: loss error {exact['loss_absolute_error']:.2e}, "
                           f"minimum raw internal gap {exact['raw_min_internal_gap']:.1e}")
    handles = [Line2D([], [], color=COLORS["ordinary"], marker="o", mfc="white", lw=1.4,
                      label="Ordinary eigh AD"),
               Line2D([], [], color=COLORS["shared"], marker="s", mfc="white", lw=1.,
                      label="Shared implicit AD (dense forward)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .835), frameon=False,
               ncol=2, fontsize=10, handlelength=2.5)
    fig.suptitle("Same smooth projector loss: ordinary eigh AD fails numerically at exact degeneracy\n"
                 r"$L=\langle W,B^{\mathsf{T}}PB\rangle$ · complete rank-4 cluster · fixed exterior gap $g=0.7$",
                 y=.965, fontsize=12)
    fig.text(.09, .145, "Exact-degeneracy forward check (dimensionless):\n"+"; ".join(exact_notes),
             fontsize=8.2, linespacing=1.5, va="top")
    fig.text(.09, .082, "Nonfinite ordinary VJPs appear only as red crosses in the separate status strip, never as zero errors.\n"
             f"Shared analytic-reference gates pass for every case. Observed with JAX {data['jax']}, CPU float64; "
             "this is an implementation result, not mathematical nondifferentiability.",
             fontsize=8.2, linespacing=1.5, va="top")
    for extension in ("png", "pdf"):
        target = output_dir/f"plain_eigh_backward_accuracy.{extension}"
        fig.savefig(target, dpi=260, bbox_inches="tight")
        print(f"Saved {target}")
    plt.close(fig)


if __name__ == "__main__":
    run()
