"""Plot the serial, converged cycloalkane GW/full-BSE scaling benchmark."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

archive = Path("reproducibility/gw_bse/cycloalkane_scaling")
report = json.loads((archive / "scaling_20261002.json").read_text())
methods = ("g0w0", "evgw0", "evgw")
labels = (r"$G_0W_0$", r"ev$GW_0$", r"ev$GW$")
colors = ("#0072B2", "#D55E00", "#009E73")

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
for method, label, color in zip(methods, labels, colors):
    for engine, style, marker in (("gradscf", "-", "o"), ("molgw", "--", "s")):
        cases = [c for c in report["cases"] if c["method"] == method
                 and c[engine]["status"] == "completed"]
        cases.sort(key=lambda c: c["n"])
        median = "warm_median" if engine == "gradscf" else "median"
        for ax, stage in zip(axes, ("gw", "bse")):
            x = [c["n"] for c in cases]
            y = [c[engine][median][stage + "_seconds"] for c in cases]
            ax.plot(x, y, linestyle=style, marker=marker, color=color, lw=1.6, ms=5)
            # Warm-repeat range for JAX; three-run range for native MolGW.
            low = [min(s[stage + "_seconds"] for s in c[engine]["samples"][1 if engine == "gradscf" else 0:]) for c in cases]
            high = [max(s[stage + "_seconds"] for s in c[engine]["samples"][1 if engine == "gradscf" else 0:]) for c in cases]
            ax.fill_between(x, low, high, color=color, alpha=.09)

for ax, title in zip(axes, ("GW", "Full BSE: all singlet roots")):
    ax.set(xlabel=r"n in cycloalkane C$_n$H$_{2n}$", ylabel="Wall time (s)",
           title=title, xscale="log", yscale="log", xlim=(2.75, 8.25))
    ax.set_xticks((3, 4, 6, 8), labels=("3", "4", "6", "8"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", which="both", alpha=.18)
handles = [Line2D([], [], color=c, lw=2, label=l) for c, l in zip(colors, labels)]
handles += [Line2D([], [], color="#333333", linestyle=s, marker=m, label=l)
            for s, m, l in (("-", "o", "GradSCF, warm"), ("--", "s", "MolGW, native"))]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .91),
           ncol=3, frameon=False, fontsize=9)
fig.suptitle("Cycloalkane GW + BSE scaling (pre-review snapshot)", fontsize=14, y=.99)
fig.text(.5, .025, "STO-3G / Weigend · CPU float64 · serial repeats · BLAS/OpenMP limits: 1\n"
         "Completed, converged points only; shaded bands show repeat ranges.\n"
         "GradSCF default G0W0: n=6 timed out; n=8 exited unsuccessfully.",
         ha="center", va="bottom", fontsize=9, color="#444444")
fig.tight_layout(rect=(0, .15, 1, .88))
fig.savefig(archive / "scaling_20261002.png", dpi=220)
fig.savefig(archive / "scaling_20261002.pdf")
print(archive / "scaling_20261002.png")
