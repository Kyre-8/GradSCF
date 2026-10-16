"""Nonlinear dipole and finite-field verification from saved HF-molecule data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

output = Path("reproducibility/ad_showcase/R03_hf_hyperpolarizability")
data = json.loads((output / "gradscf.json").read_text())
reference = json.loads((output / "pyscf.json").read_text())
fields = np.array(data["fields_au"])
nonlinear = np.array(data["dipoles_au"]) - data["mu_z"] - data["alpha_zz"]*fields
quadratic = .5*data["beta_zzz"]*fields**2
cubic = quadratic + data["gamma_zzzz"]*fields**3/6
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
axes[0].plot(fields, nonlinear*1e6, "o", ms=3, label="Relaxed SCF")
axes[0].plot(fields, quadratic*1e6, "--", label="Through beta")
axes[0].plot(fields, cubic*1e6, "-", label="Through gamma")
axes[0].set(xlabel="Electric field Fz / au", ylabel="Nonlinear dipole / (10⁻⁶ e Bohr)",
            title="Dipole beyond linear response")
axes[0].legend(frameon=False)
for axis, quantity in zip(axes[1:], ("beta", "gamma")):
    rows = data["finite_field_checks"]
    axis.loglog([x["field_step_au"] for x in rows], [x[quantity+"_error"] for x in rows],
                 "o-", label="AD of lower response + FD")
    rows = reference["finite_field_fits"]
    axis.loglog([x["window_au"] for x in rows], [x[quantity+"_error"] for x in rows],
                 "s--", label="Independent PySCF fit")
    axis.set(xlabel="Field step / fit half-window / au", ylabel="Absolute error / au",
             title=f"{quantity}: validation and noise floor")
axes[1].legend(frameon=False, fontsize=8)
fig.suptitle("Hydrogen fluoride · RHF / Cartesian 6-31+G* · fixed geometry")
for suffix in ("png", "pdf"):
    fig.savefig(output / f"nonlinear_response.{suffix}", dpi=220)
plt.close(fig)
