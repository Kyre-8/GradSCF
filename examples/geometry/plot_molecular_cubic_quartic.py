"""Plot completed G08 derivative checks; no electronic calculation in this file."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


output = Path("reproducibility/ad_showcase/G08_molecular_cubic_quartic")
cases = ("h2", "water_sto3g", "water_631gstar")
titles = ("H₂ / 6-31G\nBond-length coordinate", "H₂O / STO-3G\nFixed A₁ direction",
          "H₂O / 6-31G*\nUnit-norm bend direction")
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
figure, axes = plt.subplots(2, 3, figsize=(12, 6.6), constrained_layout=True)
geometry_figure, geometry_axes = plt.subplots(1, 3, figsize=(10, 3.7), constrained_layout=True)
rows = []
for column, (case, title) in enumerate(zip(cases, titles)):
    data = json.loads((output/case/"gradscf.json").read_text())
    reference = json.loads((output/case/"pyscf.json").read_text())
    for row, (order, key) in enumerate(((3, "cubic_error"), (4, "quartic_error"))):
        ax = axes[row, column]
        for checks, label, marker, color in (
                (data["native_curvature_differences"], "GradSCF curvature FD", "o", "#276A9D"),
                (reference["checks"], "Independent PySCF Hessian FD", "x", "#C16A32")):
            ax.loglog([item["step_bohr"] for item in checks], [item[key] for item in checks],
                      marker=marker, color=color, label=label)
            for item in checks:
                rows.append(dict(case=case, order=order, reference=label, step_bohr=item["step_bohr"],
                                 ad_derivative=data["derivatives"][order], absolute_error=item[key]))
        ax.set_xlabel("Central step / bohr")
        ax.set_ylabel(f"Absolute error / Eh bohr$^{{-{order}}}$")
        ax.grid(alpha=.18, which="both")
        ax.text(.04, .94, f"AD derivative = {data['derivatives'][order]:.9g}",
                transform=ax.transAxes, va="top", fontsize=9)
        if row == 0:
            ax.set_title(title)
    atoms = np.array(data["coordinates_bohr"])
    vectors = np.array(data["direction"])
    ax = geometry_axes[column]
    for atom in range(1, len(atoms)):
        ax.plot(atoms[[0, atom], 1], atoms[[0, atom], 2], color="#A7ADB4", lw=2, zorder=1)
    for symbol, position, vector in zip(data["symbols"], atoms, vectors):
        ax.scatter(position[1], position[2], s=160 if symbol == "O" else 65,
                   color="#BD4339" if symbol == "O" else "#808894", zorder=3)
        ax.annotate("", position[1:]+.3*vector[1:], position[1:],
                    arrowprops=dict(arrowstyle="->", color="#276A9D", lw=2), zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-1.2, 1.6)
    ax.set_xlabel("y / bohr")
    ax.set_ylabel("z / bohr")
    ax.set_title(title)
axes[1, 0].legend(fontsize=8, loc="lower right")
figure.suptitle("Relaxed RHF cubic and quartic derivatives: multi-step validation", fontsize=14)
geometry_figure.suptitle("Fixed displacement paths: arrows show q = +0.3 bohr", fontsize=13)
for fig, name in ((figure, "derivative_checks"), (geometry_figure, "displacement_paths")):
    fig.savefig(output/f"{name}.png", dpi=180)
    fig.savefig(output/f"{name}.pdf")
    plt.close(fig)
with (output/"derivative_checks.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
