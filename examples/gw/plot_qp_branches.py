"""Plot recorded water QP branches and their directional derivative checks.

No electronic calculation is performed. Independent seed selection and local
branch continuation are deliberately shown as different numerical questions.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FOLDER = Path("reproducibility/ad_showcase/GW_BSE_response_gates")
FILES = {
    "Secant, independent seeds": ("water-qp-branch-secant.json", "#287da6"),
    "Newton, independent seeds": ("water-qp-branch-newton.json", "#b56824"),
    "Newton, continued branch": ("water-qp-branch-newton-continued.json", "#218971"),
}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
records = {label: json.loads((FOLDER/name).read_text()) for label, (name, _) in FILES.items()}
fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
table = []
for label, record in records.items():
    color = FILES[label][1]
    rows = sorted(record["rows"], key=lambda row: row["step_bohr"])
    steps = np.array([row["step_bohr"] for row in rows])
    for column, orbital in enumerate((14, 17, 18)):
        qp = [row["qp"][orbital] for row in rows]
        weight = [row["weight"][orbital] for row in rows]
        axes[0, column].plot(1000*steps, qp, ".-", color=color, label=label, linewidth=1.2)
        axes[1, column].plot(1000*steps, weight, ".-", color=color, linewidth=1.2)
        axes[0, column].set_title(f"MO index {orbital} (zero-based)")
        axes[0, column].set_ylabel("Selected root / Eh")
        axes[1, column].set_ylabel("Local QP residue Z")
        axes[1, column].set_xlabel(r"Displacement parameter / $10^{-3}$ bohr")
        for row in rows:
            table.append([label, row["step_bohr"], orbital, row["qp"][orbital],
                          row["weight"][orbital], row["residual"][orbital], row["converged"][orbital]])
    if record.get("anchor_path"):
        center = next(row for row in rows if row["step_bohr"] == 0.)
        for column, orbital in enumerate((14, 17, 18)):
            linear = center["qp"][orbital]+steps*record["derivative"][orbital]
            axes[0, column].plot(1000*steps, linear, "k--", linewidth=.9, label="Local AD tangent")
for ax in axes.ravel():
    ax.grid(alpha=.18)
    ax.ticklabel_format(axis="y", useOffset=False)
axes[0, 1].legend(fontsize=8)
fig.suptitle("Water / Cartesian 6-31G*: the selected QP root must be tracked\n"
             "Same HF, G/W windows and self-energy; continuation does not select the largest-Z peak")
for suffix in ("png", "pdf"):
    fig.savefig(FOLDER/f"qp_branches.{suffix}", dpi=180)
plt.close(fig)
with (FOLDER/"qp_branches.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["scheme", "step_bohr", "orbital_zero_based", "root_hartree", "qp_weight", "residual_hartree", "converged"])
    writer.writerows(table)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
rows = []
for label, record in records.items():
    color = FILES[label][1]
    steps = [check["step"] for check in record["checks"]]
    for ax, selected, title in zip(axes, (slice(None), slice(0, 13)),
                                   ("All 19 QP roots", "MO indices 0–12")):
        errors = [max(check["errors"][selected]) for check in record["checks"]]
        ax.loglog(steps, errors, "o-", color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel("Central-difference step / bohr")
        ax.set_ylabel("Maximum |AD − FD| / (Eh bohr⁻¹)")
        ax.set_xlim(max(steps)*1.1, min(steps)/1.1)
        ax.grid(alpha=.2, which="both")
    for check in record["checks"]:
        rows.extend([label, check["step"], i, record["derivative"][i], value, check["errors"][i]]
                    for i, value in enumerate(check["finite"]))
axes[0].legend(fontsize=8)
fig.suptitle("Reconverged finite differences: branch switching is not a local AD derivative")
for suffix in ("png", "pdf"):
    fig.savefig(FOLDER/f"qp_derivative_checks.{suffix}", dpi=180)
plt.close(fig)
with (FOLDER/"qp_derivative_checks.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["scheme", "step_bohr", "orbital_zero_based", "ad_hartree_per_bohr", "fd_hartree_per_bohr", "absolute_error"])
    writer.writerows(rows)
print("Saved QP branch figures and unscaled root/derivative tables.")
