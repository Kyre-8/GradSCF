"""Plot saved A05 fixed-topology, fixed-auxiliary DF-HF exponent results."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path("reproducibility/ad_showcase/A05_basis_exponents")
native_path = OUTPUT/"gradscf.json"
data = json.loads(native_path.read_text())
reference = json.loads((OUTPUT/"pyscf.json").read_text())
assert data["complete"] and reference["passed"]
assert reference["native_sha256"] == hashlib.sha256(native_path.read_bytes()).hexdigest()
trajectory = data["trajectory"]
iterations = np.arange(len(trajectory))
energy = np.array([row["energy"] for row in trajectory])
scale = np.exp(np.array([row["theta"] for row in trajectory]))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), layout="constrained")
axes[0, 0].plot(iterations, 1000*(energy-energy[0]), "o-", color="#0072B2", ms=4)
axes[0, 0].set(title="Energy decrease at accepted basis updates", xlabel="Accepted optimizer iteration",
               ylabel="E_DF − E_DF(initial) / mEh")
for i, label in enumerate(("Contracted s exponents", "Diffuse s exponent")):
    axes[0, 1].plot(iterations, scale[:, i], "o-", ms=4, label=label)
for bound in np.exp(data["bounds_theta"]):
    axes[0, 1].axhline(bound, color=".5", linestyle=":")
axes[0, 1].set(title="Positive shared exponent scales", xlabel="Accepted optimizer iteration",
               ylabel="α / α₀ = exp(θ)", ylim=(.4, 2.15))
axes[0, 1].legend(frameon=False, fontsize=8)
for local, external, color in zip(data["derivative_checks"], reference["checks"], ("#0072B2", "#D55E00")):
    for checks, style, label in ((local["finite_differences"], "o-", "native"),
                                 (external["finite_differences"], "x--", "independent PySCF")):
        axes[1, 0].loglog([r["step"] for r in checks], [r["max_error"] for r in checks],
                           style, color=color, label=f"{local['point']}: {label}")
axes[1, 0].set(title="Log-exponent gradients versus reconverged FD", xlabel="Dimensionless step h",
               ylabel="Maximum |AD − FD| / Eh")
axes[1, 0].legend(frameon=False, fontsize=8)
errors = [max(r["normalization_error"] for r in data["evaluations"]),
          max(r["normalization_jacobian_max"] for r in data["derivative_checks"]),
          max(r["native_energy_error"] for r in reference["checks"])]
axes[1, 1].bar(range(3), np.maximum(errors, 1e-18), color="#0072B2")
axes[1, 1].set(yscale="log", xticks=range(3), xticklabels=["diag(S) − 1", "d diag(S)/dθ", "E_DF − PySCF\n/ Eh"],
               title="Normalization and independent energy checks", ylabel="Maximum absolute error")
for i, value in enumerate(errors):
    axes[1, 1].annotate(f"{value:.1e}", (i, max(value, 1e-18)), xytext=(0, 5), textcoords="offset points", ha="center")
axes[1, 1].set_ylim(top=max(errors)*20)
fig.suptitle("H₂ DF-HF · four main AOs · fixed def2-universal-jkfit auxiliary basis\n"
             "Two bounded log-exponent scales; normalized shells; fixed geometry and topology", fontsize=12)
for suffix in ("png", "pdf"):
    fig.savefig(OUTPUT/f"bounded_exponents.{suffix}", dpi=220)
plt.close(fig)
print("Saved A05 optimization and derivative validation figure")
