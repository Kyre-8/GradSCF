"""Read saved X02 data and compare ground/excited forces and gradients."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


output = Path("reproducibility/ad_showcase/X02_formaldehyde_cis_gradients")
native = json.loads((output / "gradscf.json").read_text())
reference = json.loads((output / "pyscf.json").read_text())
if not native["complete"] or not reference["passed"]:
    raise RuntimeError("Only complete, independently checked gradients are plotted.")
if reference["native_sha256"] != hashlib.sha256((output / "gradscf.json").read_bytes()).hexdigest():
    raise RuntimeError("The comparison belongs to a different native result file.")
r = np.asarray(native["coordinates_bohr"])
ground = np.asarray(native["ground_gradient_hartree_per_bohr"])
excited = np.asarray(native["total_excited_gradient_hartree_per_bohr"])
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), layout="constrained")
for ax, gradient, title, color in zip(axes[0], (ground, excited),
                                     ("Ground-state RHF forces", "Total S1 CIS forces"),
                                     ("#555555", "#0072B2")):
    for other in (1, 2, 3):
        ax.plot(r[[0, other], 1], r[[0, other], 2], color=".75", linewidth=2)
    ax.scatter(r[:, 1], r[:, 2], c=["#555555", "#B74141", "#cccccc", "#cccccc"],
               s=[110, 130, 65, 65], edgecolors="white", zorder=3)
    arrows = ax.quiver(r[:, 1], r[:, 2], -gradient[:, 1], -gradient[:, 2],
                       angles="xy", scale_units="xy", scale=.10, color=color, zorder=4)
    ax.quiverkey(arrows, .77, .95, .05, "0.05 Eh/Bohr", labelpos="S")
    ax.set(xlabel="y / Bohr", ylabel="z / Bohr", title=title,
           xlim=(-2.5, 2.5), ylim=(-2., 3.5), aspect="equal")

ax = axes[1, 0]
all_values = []
for name, label, color, marker in (("excitation", "dω/dR", "#D55E00", "o"),
                                   ("total_excited", "d(E₀+ω)/dR", "#0072B2", "s")):
    actual = np.asarray(native[f"{name}_gradient_hartree_per_bohr"]).ravel()
    expected = np.asarray(reference[f"{name}_gradient_hartree_per_bohr"]).ravel()
    all_values.extend(actual)
    all_values.extend(expected)
    ax.scatter(expected, actual, label=label, color=color, marker=marker, s=35, alpha=.75)
limits = [min(all_values)-.01, max(all_values)+.01]
ax.plot(limits, limits, "--", color=".6", linewidth=1, zorder=0)
ax.set(xlabel="PySCF analytic gradient / (Eh/Bohr)", ylabel="GradSCF AD gradient / (Eh/Bohr)",
       title="Two distinct energy derivatives")
ax.legend(frameon=False)
ax = axes[1, 1]
ad = np.asarray(native["directional_ad"])
errors = np.max(np.abs(np.asarray(native["finite_differences"])-ad[None]), axis=2)
for i, label, color in ((0, "Excitation energy", "#D55E00"), (1, "Total excited energy", "#0072B2")):
    ax.loglog(native["finite_difference_steps_bohr"], errors[:, i], "o-", color=color, label=label)
ax.axhline(1e-6, linestyle="--", color=".5", linewidth=1)
ax.set(xlabel="Central-difference step / Bohr", ylabel="Max directional AD-FD error / (Eh/Bohr)",
       title="Fixed-anchor state tracking")
ax.legend(frameon=False)
fig.suptitle("Formaldehyde · RHF/CIS · Cartesian 6-31+G*\n"
             "Fixed geometry; native integrals + implicit orbital response", fontsize=12)
for suffix in ("png", "pdf"):
    fig.savefig(output / f"gradient_checks.{suffix}", dpi=220)
plt.close(fig)
print("Saved gradient_checks.png/.pdf")

# Measured on jy_gpu_1, 2026-10-15: Saved gradient_checks.png/.pdf.
