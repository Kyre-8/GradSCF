"""Plot the saved dimensionless A03 crossing and subspace diagnostics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path("reproducibility/ad_showcase/A03_degenerate_subspace")
data = json.loads((OUTPUT/"results.json").read_text())
assert data["complete"] and not data["cut_cluster_response_valid"]
rows = data["scan"]
t = np.array([row["parameter"] for row in rows])
e = np.array([row["eigenvalues"][:2] for row in rows])
slopes = np.array([row["sorted_root_slopes"] or [np.nan, np.nan] for row in rows])
ad = np.array([row["ad_derivatives"] for row in rows])
fd = np.array([row["finite_differences"][-1]["derivative"] for row in rows])
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), layout="constrained")
for i, color in enumerate(("#0072B2", "#D55E00")):
    axes[0, 0].plot(t, e[:, i], color=color, label=f"Sorted root {i+1}")
    for side in (t < 0, t > 0):
        axes[0, 1].plot(t[side], slopes[side, i], color=color,
                       label=f"Root {i+1}, t ≠ 0" if side[0] else None)
axes[0, 0].scatter([0, 0], [1, 1], color="black", zorder=4, s=25)
axes[0, 0].set(title="An internal crossing inside the selected pair", ylabel="Eigenvalue (dimensionless)")
axes[0, 1].set(title="Sorted-root derivatives have unequal one-sided limits", ylabel="dλ/dt (dimensionless)")
axes[0, 1].text(.03, .07, "No sorted-root derivative assigned at t = 0",
                 transform=axes[0, 1].transAxes, fontsize=9)
for i, (label, color) in enumerate((("probeᵀ P probe", "#0072B2"), ("tr(PA)", "#D55E00"))):
    axes[1, 0].plot(t, ad[:, i], color=color, label=f"AD: {label}")
    axes[1, 0].plot(t[::5], fd[::5, i], "x", color=color, label=f"Independent FD: {label}")
axes[1, 0].set(title="Whole-subspace responses remain smooth", ylabel="Observable derivative / dt")
axes[1, 1].semilogy(t, [row["boundary_gap"] for row in rows], color="#0072B2", label="Rank-2 exterior gap")
axes[1, 1].semilogy(t[t != 0], np.array([row["internal_gap"] for row in rows])[t != 0],
                     ".", color="#D55E00", label="Internal gap (zero at crossing)")
axes[1, 1].set(title="Isolation, not internal root splitting, controls validity", ylabel="Spectral gap")
axes[1, 1].text(.03, .06, "Rank 2 valid at t = 0; rank 1 cuts the cluster and is invalid",
                 transform=axes[1, 1].transAxes, fontsize=8)
for ax in axes.flat:
    ax.set_xlabel("Dimensionless parameter t")
    ax.axvline(0, color=".6", linestyle=":", lw=.8)
    ax.legend(frameon=False, fontsize=8)
axes[1, 1].legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1., .88))
fig.suptitle("A03 · fixed-rank spectral projector through an exact internal degeneracy", fontsize=13)
for suffix in ("png", "pdf"):
    fig.savefig(OUTPUT/f"crossing_response.{suffix}", dpi=220)
plt.close(fig)
print("Saved A03 crossing and response figure")
