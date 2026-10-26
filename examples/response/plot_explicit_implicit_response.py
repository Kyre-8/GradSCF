"""Plot saved A01 phase measurements and checks; no numerical solver runs."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("reproducibility/ad_showcase/A01_explicit_implicit")
native_path = OUTPUT/"gradscf.json"
native = json.loads(native_path.read_text())
reference = json.loads((OUTPUT/"pyscf.json").read_text())
if not native["complete"] or not reference["passed"]:
    raise RuntimeError("Plot only complete, independently checked A01 data.")
if reference["native_sha256"] != hashlib.sha256(native_path.read_bytes()).hexdigest():
    raise RuntimeError("The independent reference belongs to another native run.")
methods, observables, modes = native["methods"], ("energy", "dipole_z"), ("explicit", "implicit")
colors = {"explicit": "#0072B2", "implicit": "#D55E00"}
phases = [("forward_cold_seconds", "Forward\nfirst"), ("forward_warm_seconds", "Forward\nwarm"),
          ("vjp_preparation_cold_seconds", "VJP prep\nfirst"), ("vjp_preparation_warm_seconds", "VJP prep\nwarm"),
          ("pullback_cold_seconds", "Pullback\nfirst"), ("pullback_warm_seconds", "Pullback\nwarm")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, len(methods), figsize=(12.8, 7.5), layout="constrained", squeeze=False)
summary = []
for column, method in enumerate(methods):
    for row, observable in enumerate(observables):
        ax = axes[row, column]
        for index, mode in enumerate(modes):
            case = next(c for c in native["cases"] if (c["method"], c["mode"], c["observable"]) == (method, mode, observable))
            values, lower, upper = [], [], []
            for key, _ in phases:
                samples = np.atleast_1d(case[key])
                center = np.median(samples)
                values.append(center)
                lower.append(center-samples.min())
                upper.append(samples.max()-center)
            ax.bar(np.arange(len(phases))+(index-.5)*.36, values, width=.34,
                   yerr=np.array([lower, upper]), capsize=2, color=colors[mode], label=mode)
            summary.append(dict(method=method, mode=mode, observable=observable,
                iterations=case["orbital_iterations"], timed_peak_rss_mib=case["timed_fresh_process_peak_rss_mib"],
                **{key: float(value) for (key, _), value in zip(phases, values)}))
        label = "E / Eh" if observable == "energy" else "μz / (e bohr)"
        ax.set(yscale="log", ylabel="Directly measured wall time / s", title=f"{method} · {label}",
               xticks=np.arange(len(phases)), xticklabels=[label for _, label in phases])
        ax.legend(frameon=False)
fig.suptitle("H₂ / 6-31G · identical initial orbitals and nonzero (bond, field) parameters\n"
             "First = first transformed call after seed SCF; warm = median, whiskers = min/max", fontsize=12)
for suffix in ("png", "pdf"):
    fig.savefig(OUTPUT/f"phase_timing.{suffix}", dpi=220)
plt.close(fig)
with (OUTPUT/"timing_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

fig, axes = plt.subplots(2, len(methods), figsize=(11.8, 7.3), layout="constrained", squeeze=False)
for column, method in enumerate(methods):
    ref = next(c for c in reference["cases"] if c["method"] == method)
    for observable, color in zip(observables, ("#0072B2", "#D55E00")):
        cases = [c for c in native["cases"] if c["method"] == method and c["observable"] == observable]
        steps = [c["step_theta"] for c in cases[0]["finite_difference_checks"]]
        errors = [max(c["finite_difference_checks"][i]["max_error"] for c in cases) for i in range(len(steps))]
        external = [max(value for key, value in check["errors"].items() if key.endswith(observable))
                    for check in ref["finite_difference_checks"]]
        label = "E/Eh" if observable == "energy" else "μz/(e bohr)"
        axes[0, column].loglog(steps, errors, "o-", color=color, label=f"{label}: native FD")
        axes[0, column].loglog(steps, external, "x--", color=color, label=f"{label}: PySCF FD")
    axes[0, column].axhline(2e-6, color=".5", linestyle=":", label="smallest-step gate")
    axes[0, column].set(title=f"{method} · max error over modes and θ components",
                        xlabel="Dimensionless finite-difference step h", ylabel="|AD − FD| for normalized observables")
    axes[0, column].legend(frameon=True, facecolor="white", framealpha=1, fontsize=8)
    for index, mode in enumerate(modes):
        values = [next(c["timed_fresh_process_peak_rss_mib"] for c in native["cases"]
                       if (c["method"], c["mode"], c["observable"]) == (method, mode, observable))
                  for observable in observables]
        axes[1, column].bar(np.arange(2)+(index-.5)*.35, values, width=.33, color=colors[mode], label=mode)
    axes[1, column].set(title=f"{method} · fresh-process high-water memory",
                        ylabel="Peak RSS / MiB", xticks=[0, 1], xticklabels=["Energy", "Dipole μz"], ylim=(0, None))
    axes[1, column].set_ylim(top=max(c["timed_fresh_process_peak_rss_mib"]
                                  for c in native["cases"] if c["method"] == method)*1.28)
    axes[1, column].legend(frameon=False, ncol=2, loc="upper center")
fig.suptitle("Physical-response checks and observed process memory\n"
             "RSS includes imports, seed SCF and timed phases; validation is outside that window", fontsize=12)
for suffix in ("png", "pdf"):
    fig.savefig(OUTPUT/f"validation_and_memory.{suffix}", dpi=220)
plt.close(fig)
(OUTPUT/"plot_settings.json").write_text(json.dumps(dict(
    timing="Each phase measured directly. VJP preparation includes a primal; no stacked totals or wall-time subtraction.",
    warm_summary="Median of five calls, with min/max whiskers.",
    memory="Fresh-process high-water RSS after timed phases; not backward-only allocated memory.",
    scope="Observed small-system shared-node wall times; not an asymptotic scaling or universal speedup claim."), indent=2)+"\n")
print("Saved A01 timing, validation and memory figures plus timing_summary.csv")
