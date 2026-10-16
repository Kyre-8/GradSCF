"""Render saved HF/PBE electric response and export its numerical CSVs."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("reproducibility/ad_showcase")
CASES = (RESULTS / "R01_R02_water_electric", RESULTS / "R01_R02_water_pbe_electric")


def run(case_path, output=None):
    case_path = Path(case_path)
    data = json.loads((case_path / "gradscf.json").read_text())
    reference = json.loads((case_path / "pyscf.json").read_text())
    output = case_path if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    alpha = np.asarray(data["polarizability_bohr3"])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout="constrained")
    im = axes[0].imshow(alpha, cmap="Blues", vmin=0)
    axes[0].set(xticks=range(3), xticklabels=list("xyz"), yticks=range(3), yticklabels=list("xyz"),
                title="Static polarizability / Bohr³")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{alpha[i,j]:.5f}", ha="center", va="center",
                         color="white" if alpha[i,j] > .6*alpha.max() else "black")
    for rows, label, color in ((data["finite_field_checks"], "GradSCF", "#337a9b"),
                               (reference["finite_field_checks"], "PySCF", "#a45d20")):
        axes[1].loglog([x["field_step_au"] for x in rows], [x["max_alpha_error"] for x in rows],
                       "o-", color=color, label=label)
    axes[1].set(xlabel="Finite-field step / au", ylabel="Max difference from AD / Bohr³",
                title="Relaxed finite-field validation")
    axes[1].legend(frameon=False)
    label = data.get("xc", data["method"])
    fig.suptitle(f"Water · {label} / Cartesian {data['basis']} · its optimized geometry")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"electric_response.{suffix}", dpi=220)
    plt.close(fig)
    with (output / "dipole.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["component", "dipole_e_bohr"])
        writer.writerows(zip("xyz", data["dipole_e_bohr"]))
    with (output / "polarizability.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["response_component", "field_component", "polarizability_bohr3"])
        writer.writerows((axis, field, alpha[i, j]) for i, axis in enumerate("xyz")
                         for j, field in enumerate("xyz"))
    with (output / "finite_field_checks.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "field_step_au", "max_alpha_error_bohr3"])
        for label, payload in (("GradSCF", data), ("PySCF", reference)):
            writer.writerows((label, row["field_step_au"], row["max_alpha_error"])
                             for row in payload["finite_field_checks"])


if __name__ == "__main__":
    for case_path in CASES:
        run(case_path)

# PBE figure generated on jy_gpu_1, 2026-10-15:
# R01_R02_water_pbe_electric/electric_response.png and electric_response.pdf
