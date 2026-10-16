"""Constrained RHF derivatives and independently selected spin branches."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


output = Path("reproducibility/ad_showcase/G07_h2_surface")
rows = json.loads((output / "gradscf.json").read_text())["records"]
reference = json.loads((output / "pyscf.json").read_text())["records"]
r = [row["bond_bohr"] for row in rows]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), layout="constrained")
axes[0, 0].plot(r, [v["rhf_energy_hartree"] for v in rows], "o-", label="Constrained RHF")
axes[0, 0].plot(r, [v["selected_energy_hartree"] for v in rows], "s--", label="Spin-followed HF")
axes[0, 0].set(ylabel="Energy / Eh", title="Primal branch selection")
axes[0, 0].legend(frameon=False)
axes[0, 1].plot(r, [v["spin_minimum_curvature"] for v in rows], "o-")
axes[0, 1].axhline(0., color="0.4", lw=.8)
axes[0, 1].set(ylabel="Spin-channel curvature / Eh", title="Restricted spin instability")
for axis, quantity, label in ((axes[1, 0], "gradient_hartree_per_bohr", "dE_RHF/dR / Eh Bohr⁻¹"),
                               (axes[1, 1], "curvature_hartree_per_bohr2", "d²E_RHF/dR² / Eh Bohr⁻²")):
    axis.plot(r, [v[quantity] for v in rows], "o-", label="Implicit AD")
    axis.plot(r, [v[quantity] for v in reference], "+", ms=9, label="PySCF analytic")
    axis.set(ylabel=label, title="Fixed restricted branch")
axes[1, 0].legend(frameon=False)
for ax in axes.ravel():
    ax.set_xlabel("H–H distance / Bohr")
    ax.axvspan(2.2, 2.6, color="#d8b66c", alpha=.2)
fig.suptitle("H₂ / Cartesian 6-31G* · shaded interval brackets the sampled spin-stability change")
for suffix in ("png", "pdf"):
    fig.savefig(output / f"response_surface.{suffix}", dpi=220)
plt.close(fig)
