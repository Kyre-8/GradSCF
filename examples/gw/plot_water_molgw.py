"""Plot measured water G0W0/BSE reference and geometry-response comparisons."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("reproducibility/ad_showcase/X02_water_gw_bse_response/primary")


def plot(*, nw=200, comparison_name="refined"):
    folder = OUTPUT/f"nw{nw}"
    comparison = json.loads((folder/f"comparison_{comparison_name}.json").read_text())
    # Rejected reference grids are useful diagnostics and remain visibly marked.
    reference_note = ("\nQP reference refined separately; optical reference unchanged"
                      if comparison["validation_passed"] else
                      "\nReference comparison not fully accepted; see recorded errors")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    for column, (label, title) in enumerate((("tda", "G₀W₀ + BSE / TDA"), ("full", "G₀W₀ + full BSE"))):
        data = np.loadtxt(folder/f"{label}_molgw_{comparison_name}_comparison.csv", delimiter=",", skiprows=1)
        axes[0, column].plot(data[:, 0], data[:, 1], color="#137c9a", label="GradSCF")
        axes[0, column].plot(data[:, 0], data[:, 2], color="#bb4f36", linestyle="--", label="MolGW")
        axes[0, column].set(title=title, ylabel="Cross section / bohr²")
        axes[0, column].legend(frameon=False)
        axes[1, column].plot(data[:, 0], data[:, 3], color="#6b4e93", linewidth=1.)
        axes[1, column].axhline(0., color=".7", linewidth=.6)
        axes[1, column].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[1, column].set(xlabel="Photon energy / eV", ylabel="Difference / bohr²", xlim=(0, 50))
    fig.suptitle("Water / Cartesian 6-31G*: 19 QP orbitals and 70 optical transitions\n"
                 "Native main-peak seeds; imposed Lorentzian half-width 0.01 Eh"+reference_note)
    for suffix in ("png", "pdf"):
        fig.savefig(folder/f"water_molgw_spectra_{comparison_name}.{suffix}")
    plt.close(fig)

    paths = [OUTPUT/f"g0w0_{label}_nw{nw}_continued.json" for label in ("tda", "full")]
    if not all(path.exists() for path in paths):
        return
    records = [json.loads(path.read_text()) for path in paths]
    assert all(record["validation_passed"] for record in records)
    panels = (("qp_energy", "QP energy", "Eh/bohr"), ("qp_weight", "QP residue", "1/bohr"),
              ("screening_energy", "Screening energies", "Eh/bohr"),
              ("excitation_energy", "Excitation energy", "Eh/bohr"),
              ("oscillator_strength", "Oscillator strength", "1/bohr"))
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.3), layout="constrained")
    for axis, (key, title, unit) in zip(axes, panels):
        for record, label, color in zip(records, ("TDA", "Full BSE"), ("#137c9a", "#bb4f36")):
            checks = record["finite_difference_checks"]
            axis.loglog([row["step_bohr"] for row in checks],
                        [row["errors"][key] for row in checks], "o-", label=label, color=color)
        axis.set(title=title, xlabel="Central step / bohr", ylabel=f"Max |AD − FD| / {unit}")
        axis.grid(alpha=.18, which="both")
    axes[-1].legend(frameon=False)
    fig.suptitle("Water: complete geometry response along one fixed Cartesian direction\n"
                 "Local continuation of native main-peak branches; six lowest optical states")
    for suffix in ("png", "pdf"):
        fig.savefig(folder/f"water_geometry_response.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
