"""Plot saved A02 scaled response matrices and validated contractions."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("reproducibility/ad_showcase/A02_molecular_products")
native_path = OUTPUT/"gradscf.json"
data = json.loads(native_path.read_text())
reference = json.loads((OUTPUT/"pyscf.json").read_text())
if not data["complete"] or not reference["passed"]:
    raise RuntimeError("Only complete independently checked A02 products are plotted.")
if reference["native_sha256"] != hashlib.sha256(native_path.read_bytes()).hexdigest():
    raise RuntimeError("The independent comparison belongs to another native run.")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
labels = [name.replace("_", " ") for name in data["parameter_labels"]]
fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0), layout="constrained")
for ax, values, title in ((axes[0, 0], np.asarray(data["jacobian_theta"]), "J = ∂[E/Eh, μ/(e a₀)] / ∂θ"),
                          (axes[0, 1], np.asarray(data["hessian_theta"]), "H = ∂²(E/Eh) / ∂θ²")):
    maximum = np.max(np.abs(values))
    plot = ax.imshow(values, cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto")
    ax.set(title=title, xticks=np.arange(12), xticklabels=labels, xlabel="Dimensionless parameter θ")
    ax.tick_params(axis="x", labelrotation=50)
    ax.axvline(8.5, color=".2", linewidth=.8)
    fig.colorbar(plot, ax=ax, shrink=.86)
axes[0, 0].set(yticks=np.arange(4), yticklabels=["E/Eh", "μx/(e a₀)", "μy/(e a₀)", "μz/(e a₀)"])
axes[0, 1].set(yticks=np.arange(12), yticklabels=labels, ylabel="Dimensionless parameter θ")
axes[0, 1].axhline(8.5, color=".2", linewidth=.8)

names = ("jvp_dense", "vjp_dense", "hvp_dense", "adjoint", "hessian_symmetry")
errors = [data["contraction_errors"][name] for name in names]
axes[1, 0].bar(np.arange(len(names)), np.maximum(errors, 1e-17), color="#0072B2")
axes[1, 0].set(yscale="log", xticks=np.arange(len(names)),
    xticklabels=["JVP − Jv", "VJP − Jᵀw", "HVP − Hv", "Adjoint", "H − Hᵀ"],
    title="Dense and adjoint checks of normalized quantities", ylabel="Maximum absolute error")
for index, value in enumerate(errors):
    axes[1, 0].annotate(f"{value:.1e}", (index, max(value, 1e-17)),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
axes[1, 0].set_ylim(top=max(max(errors)*30, 1e-14))
native_steps = [c["step_theta"] for c in data["finite_difference_checks"]]
axes[1, 1].loglog(native_steps, [c["max_jvp_error"] for c in data["finite_difference_checks"]],
                   "o-", color="#0072B2", label="JVP: native observable FD")
axes[1, 1].loglog(native_steps, [c["max_hvp_error"] for c in data["finite_difference_checks"]],
                   "s-", color="#D55E00", label="HVP: native gradient FD")
axes[1, 1].loglog([c["step_theta"] for c in reference["jacobian_checks"]],
                   [c["errors"]["jvp"] for c in reference["jacobian_checks"]],
                   "x--", color="#0072B2", label="JVP: independent PySCF J_FD v")
axes[1, 1].loglog([c["step_theta"] for c in reference["hvp_checks"]],
                   [c["max_error"] for c in reference["hvp_checks"]],
                   "+--", color="#D55E00", label="HVP: independent four-energy FD")
axes[1, 1].set(title="Multistep physical finite-difference checks", xlabel="Dimensionless step h",
                ylabel="Maximum absolute error")
axes[1, 1].legend(frameon=False, fontsize=8)
fig.suptitle("Water RHF / Cartesian 6-31G* · full relaxed molecular response\n"
             "R = R₀ + 0.05 θR bohr; F = 0.002 θF au · E/Eh and μ/(e a₀) normalized", fontsize=12)
for suffix in ("png", "pdf"):
    fig.savefig(OUTPUT/f"molecular_products.{suffix}", dpi=220)
plt.close(fig)
np.savetxt(OUTPUT/"jacobian_theta.csv", data["jacobian_theta"], delimiter=",", header=",".join(labels), comments="")
np.savetxt(OUTPUT/"hessian_theta.csv", data["hessian_theta"], delimiter=",", header=",".join(labels), comments="")
(OUTPUT/"plot_settings.json").write_text(json.dumps(dict(
    matrix_units="Normalized observables versus dimensionless theta; H is d2(E/Eh)/dtheta2.",
    coordinate_scale_bohr=data["coordinate_scale_bohr"], field_scale_au=data["field_scale_au"],
    log_display_floor=1e-17, exact_errors_retained_in_json=True,
    scope="A small dense validation example; no unscaled Cartesian-Hessian or vibrational-frequency interpretation."), indent=2)+"\n")
print("Saved A02 scaled matrices, contraction checks and finite-difference figure")
