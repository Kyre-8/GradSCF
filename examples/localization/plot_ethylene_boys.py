"""Plot the archived A06 fixed-subspace Boys optimization and checks."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path("reproducibility/ad_showcase/A06_boys_localization")
data = json.loads((OUTPUT/"results.json").read_text())
assert data["complete"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), layout="constrained")
axes[0].plot(data["trajectory_scores"], "o-", ms=3, color="#0072B2")
axes[0].set(title="Selected localization trajectory", xlabel="Accepted rotation step", ylabel="Boys score / bohr²")
axes[1].bar(range(8), data["multistart_scores"], color="#0072B2")
axes[1].axhline(data["final_score"]-data["independent_pyscf_score_difference"],
                color="#D55E00", ls="--", label="Independent PySCF best")
axes[1].set(title="Eight starts expose stationary alternatives", xlabel="Starting rotation index", ylabel="Boys score / bohr²")
axes[1].legend(frameon=False, fontsize=8)
names = ["gradient_fd_error", "hvp_fd_error", "analytic_local_gradient_error", "orthogonality_error", "density_error"]
values = [data[name] for name in names]
axes[2].bar(range(5), np.maximum(values, 1e-18), color="#0072B2")
axes[2].set(yscale="log", xticks=range(5), xticklabels=["Grad FD", "HVP FD", "Analytic", "CᵀSC − I", "ΔD"],
            title="Derivative and invariance checks", ylabel="Absolute error (declared native units)")
axes[2].tick_params(axis="x", labelrotation=25)
fig.suptitle("Ethylene RHF / Cartesian cc-pVDZ · eight occupied orbitals · fixed occupied subspace", fontsize=12)
for suffix in ("png", "pdf"):
    fig.savefig(OUTPUT/f"boys_localization.{suffix}", dpi=220)
plt.close(fig)
print("Saved A06 Boys trajectory, multistart and validation figure")
