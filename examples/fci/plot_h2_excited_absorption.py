"""Weighted ground/ESA/emission sticks and the checked FCI bond response."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants


CASE = Path("reproducibility/ad_showcase/X09_h2_fci_esa")


def run():
    data = json.loads((CASE/"gradscf.json").read_text())
    reference = json.loads((CASE/"pyscf.json").read_text())
    energies = np.asarray(data["energies_hartree"])
    strengths = np.asarray(data["signed_oscillator_strengths"])
    derivative = np.asarray(data["signed_f_derivative_per_bohr"])
    hartree_ev = physical_constants["Hartree energy in eV"][0]
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    panels = ((0, 0, "Ground-state absorption · initial root 0", "#287fa3"),
              (1, 2, "Excited-state absorption · initial singlet root 2", "#ba5f24"),
              (1, 2, "Stimulated emission · initial singlet root 2", "#7363aa"))
    fig, axes = plt.subplots(3, 1, figsize=(10.6, 8.7), sharex=True, layout="constrained")
    for panel, (row, initial, title, color) in enumerate(panels):
        gaps = energies-energies[initial]
        mask = (strengths[row] > 1e-10) if panel < 2 else (strengths[row] < -1e-10)
        roots = np.flatnonzero(mask)
        photons, weights = np.abs(gaps[mask])*hartree_ev, np.abs(strengths[row, mask])
        axes[panel].vlines(photons, 0, weights, color=color, linewidth=2.3)
        axes[panel].scatter(photons, weights, color=color, s=23, zorder=3)
        for root, photon, weight in zip(roots, photons, weights):
            if weight > .015:
                axes[panel].annotate(f"{initial}→{root}", (photon, weight),
                                     xytext=(5, 7), textcoords="offset points", color=color)
        axes[panel].set(title=title, ylabel="Oscillator strength |f|", ylim=(0, 1.0))
    axes[-1].set(xlabel="Photon energy / eV", xlim=(0, 77))
    fig.suptitle("H₂ · full FCI / 6-31G · R = 1.4 Bohr\n"
                 "Actual transition weights; discrete finite-basis states; |f| > 10⁻¹⁰ shown")
    for extension in ("png", "pdf"):
        fig.savefig(CASE/f"excited_absorption.{extension}", dpi=200)
    plt.close(fig)

    targets = data["derivative_targets"]
    numerical = np.asarray(reference["finite_difference_checks"][-1]["signed_f_derivative_per_bohr"])
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.3), layout="constrained")
    axes[0].bar(np.arange(len(targets)), derivative[1, targets], width=.55,
                 color=["#287fa3", "#ba5f24"], label="Complete geometry AD")
    axes[0].scatter(np.arange(len(targets)), numerical[1, targets], s=70, marker="s",
                     facecolors="none", edgecolors="black", label="PySCF central difference")
    axes[0].axhline(0, color=".6", linewidth=.8)
    axes[0].set(xticks=np.arange(len(targets)), xticklabels=[f"2→{root}" for root in targets],
                ylabel="df / dR  /  Bohr⁻¹", title="Excited-state absorption sensitivity")
    axes[0].legend(fontsize=9)
    for label, checks, style in (("Native FCI differences", data["finite_difference_checks"], "o-"),
                                 ("Independent PySCF differences", reference["finite_difference_checks"], "s--")):
        axes[1].loglog([row["step_bohr"] for row in checks],
                       [row["max_derivative_error"] for row in checks], style, label=label)
    axes[1].axhline(3e-6, color=".5", linestyle=":", label="Small-step acceptance")
    axes[1].set(xlabel="Bond-length difference step / Bohr", ylabel="Maximum df/dR difference / Bohr⁻¹",
                title="All ground/excited rows, 16 final states")
    axes[1].legend(fontsize=8)
    fig.suptitle("Complete moving integrals, overlap/Cholesky frame and isolated FCI eigenpair response")
    for extension in ("png", "pdf"):
        fig.savefig(CASE/f"geometry_response.{extension}", dpi=200)
    plt.close(fig)

    with (CASE/"states.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["root", "energy_hartree", "spin_square", "inversion_parity", "nearest_gap_hartree"])
        writer.writerows(zip(range(len(energies)), energies, data["spin_square"], data["parity"], data["nearest_gaps_hartree"]))
    with (CASE/"transitions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["initial_root", "final_root", "signed_gap_hartree", "photon_energy_eV", "signed_f", "direction", "df_dR_per_bohr"])
        for row, initial in enumerate(data["initial_roots"]):
            for final in range(len(energies)):
                gap = energies[final]-energies[initial]
                direction = "up" if gap > 0 else "down" if gap < 0 else "diagonal"
                writer.writerow([initial, final, gap, abs(gap)*hartree_ev, strengths[row, final], direction, derivative[row, final]])
    with (CASE/"derivative_checks.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference", "step_bohr", "max_df_dR_error_per_bohr"])
        for label, checks in (("native", data["finite_difference_checks"]), ("PySCF", reference["finite_difference_checks"])):
            writer.writerows((label, row["step_bohr"], row["max_derivative_error"]) for row in checks)


if __name__ == "__main__":
    run()

# Rendered on jy_gpu_1, 2026-10-16: two PNG/PDF pairs and three CSV tables.
# Ground absorption, upward ESA and downward emission have physical f weights.
# PDF font type 42 was applied after review; scientific data were not rerun.
