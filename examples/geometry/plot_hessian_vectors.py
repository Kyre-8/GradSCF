"""G06: cold/warm HVP cost, all-column cost and process high-water memory."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


output = Path("reproducibility/ad_showcase/G06_hessian_vectors")
rows = [json.loads((output / f"{name}.json").read_text()) for name in ("H2", "water", "ethylene")]
x = np.arange(len(rows))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
for key, label in (("cold_hvp_seconds", "First HVP, including compile"),
                   ("warm_hvp_seconds", "Median warm HVP"),
                   ("warm_dense_hessian_seconds", "All 3N columns, warm")):
    axes[0].semilogy(x, [np.median(row[key]) for row in rows], "o-", label=label)
axes[0].set(ylabel="Elapsed time / s", title="Same shared directional API")
axes[0].legend(frameon=False, fontsize=8)
axes[1].plot(x, [row["process_peak_after_hvp_mib"]/1024 for row in rows], "o-", label="After HVP phase")
axes[1].plot(x, [row["process_peak_after_dense_mib"]/1024 for row in rows], "s--", label="After all columns")
axes[1].set(ylabel="Process high-water RSS / GiB", title="Fresh process for each molecule")
axes[1].legend(frameon=False, fontsize=8)
for row in rows:
    axes[2].loglog([c["step_bohr"] for c in row["finite_difference_checks"]],
                   [c["max_hvp_error"] for c in row["finite_difference_checks"]], "o-", label=row["name"])
axes[2].set(xlabel="Displacement step / Bohr", ylabel="Maximum HVP error / Eh Bohr⁻²",
            title="Independent gradient differences")
axes[2].legend(frameon=False, fontsize=8)
for ax in axes[:2]:
    ax.set(xticks=x, xticklabels=[f"{r['name']}\n{r['nao']} / {r['ncart']}" for r in rows],
           xlabel="Molecule · AO count / Cartesian coordinates")
fig.suptitle("RHF / Cartesian 6-31G* · native integrals · implicit SCF response")
for suffix in ("png", "pdf"):
    fig.savefig(output / f"hessian_vectors.{suffix}", dpi=220)
plt.close(fig)
