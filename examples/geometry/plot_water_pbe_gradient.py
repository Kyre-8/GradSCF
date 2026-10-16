"""Plot saved PBE forces, AD validation and finite-grid precision."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

output = Path("reproducibility/ad_showcase/G01_water_pbe_gradient")
data = json.loads((output / "gradscf.json").read_text())
reference = json.loads((output / "pyscf.json").read_text())
gradient = np.asarray(data["gradient_hartree_per_bohr"])
analytic = np.asarray(reference["gradient_hartree_per_bohr"])
fine = np.asarray(reference["finer_grid"]["gradient_hartree_per_bohr"])
coordinates = np.asarray(data["coordinates_bohr"])
errors = np.max(np.abs(np.asarray(data["finite_difference_gradients"])
                       - gradient), axis=(1, 2))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
ax = axes[0, 0]
for hydrogen in (1, 2):
    ax.plot(coordinates[[0, hydrogen], 1], coordinates[[0, hydrogen], 2],
            color="0.6", linewidth=2, zorder=1)
ax.scatter(coordinates[:, 1], coordinates[:, 2], s=[160, 70, 70],
           c=["#b83737", "#5587ab", "#5587ab"], zorder=3)
arrows = ax.quiver(coordinates[:, 1], coordinates[:, 2],
                   -gradient[:, 1], -gradient[:, 2], angles="xy",
                   scale_units="xy", scale=0.08, color="#253c58", zorder=4)
ax.quiverkey(arrows, 0.73, 0.9, 0.02, "0.02 Eh/Bohr", labelpos="S")
ax.set(xlabel="y / Bohr", ylabel="z / Bohr", title="PBE nuclear forces",
       xlim=(-2.2, 2.2), ylim=(-0.7, 2.2), aspect="equal")

ax = axes[0, 1]
bounds = [min(analytic.min(), gradient.min()) - 0.005,
          max(analytic.max(), gradient.max()) + 0.005]
ax.plot(bounds, bounds, color="0.65", linestyle="--", linewidth=1)
ax.scatter(analytic.ravel(), gradient.ravel(), color="#337a9b", s=32)
ax.set(xlabel="PySCF gradient / (Eh/Bohr)", ylabel="AD gradient / (Eh/Bohr)",
       title="Matched level-3 grids, full grid response")
ax.text(0.05, 0.9, f"max error: {reference['max_gradient_error']:.2e}",
        transform=ax.transAxes)

ax = axes[1, 0]
ax.loglog(data["finite_difference_steps_bohr"], errors, "o-", color="#a45d20")
ax.axhline(1e-6, linestyle="--", linewidth=1, color="0.5", label="acceptance")
ax.set(xlabel="Central-difference step / Bohr",
       ylabel="Max AD-FD error / (Eh/Bohr)", title="All 9 Cartesian components")
ax.legend(frameon=False)

ax = axes[1, 1]
positions = np.arange(9)
ax.bar(positions - 0.18, np.abs(gradient - analytic).ravel(), 0.36,
       color="#337a9b", label="AD vs PySCF, same grid")
ax.bar(positions + 0.18, np.abs(analytic - fine).ravel(), 0.36,
       color="#c88335", label="PySCF level 3 vs level 5")
ax.set(xticks=positions,
       xticklabels=[f"{atom}{axis}" for atom in ("O", "H1", "H2")
                    for axis in ("x", "y", "z")],
       ylabel="Absolute gradient change / (Eh/Bohr)",
       title="Finite-grid precision is a separate limit")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(frameon=False, fontsize=9)
fig.suptitle("PBE / Cartesian 6-31G* · moving AO integrals + moving grid + implicit SCF")
for suffix in ("png", "pdf"):
    fig.savefig(output / f"gradient_checks.{suffix}", dpi=220)
plt.close(fig)

# Generated on jy_gpu_1, 2026-10-15, from the saved numerical results:
# reproducibility/ad_showcase/G01_water_pbe_gradient/gradient_checks.png
# reproducibility/ad_showcase/G01_water_pbe_gradient/gradient_checks.pdf
