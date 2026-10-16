"""Plot saved AD Hessian checks and H2O modes; the default case is PBE.

The independent reference is a Hessian from differences of full-grid PySCF
analytic gradients. Mode-marker heights are not IR intensities. Animation
displacements are scaled for visibility and are not thermal amplitudes.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

def run(data_path=Path("reproducibility/ad_showcase/G05_water_pbe_hessian/gradscf.json"),
        reference_path=None, output=None):
    data_path = Path(data_path)
    output = data_path.parent if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads(data_path.read_text())
    method = data.get("label", data["xc"])
    response = (f"level-{data['grid_level']} moving grid + implicit response"
                if data.get("grid_level") is not None else "full electronic response")
    reference_path = data_path.parent / "pyscf.json" if reference_path is None else Path(reference_path)
    reference = json.loads(reference_path.read_text())
    hessian = np.asarray(data["hessian_hartree_per_bohr2"])
    reference_hessian = (reference["hessian_hartree_per_bohr2"] if "hessian_hartree_per_bohr2" in reference
                         else reference["hessian_checks"][-1]["hessian_hartree_per_bohr2"])
    error = hessian - reference_hessian
    labels = [f"{atom}{axis}" for atom in ("O", "H1", "H2") for axis in ("x", "y", "z")]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for ax, matrix, title in ((axes[0, 0], hessian, "Complete AD Hessian"),
                              (axes[0, 1], error, "AD minus independent reference Hessian")):
        scale = max(np.max(np.abs(matrix)), 1e-15)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-scale, vmax=scale)
        ax.set(xticks=np.arange(9), yticks=np.arange(9), xticklabels=labels,
               yticklabels=labels, title=title)
        ax.tick_params(axis="x", labelrotation=45)
        fig.colorbar(im, ax=ax, label="Eh/Bohr²", shrink=.8)
    checks = data["finite_difference_checks"]
    axes[1, 0].loglog([r["step_bohr"] for r in checks], [r["max_hvp_error"] for r in checks],
                      "o-", color="#337a9b")
    axes[1, 0].axhline(2e-6, ls="--", color=".5", label="acceptance")
    axes[1, 0].set(xlabel="Gradient-difference step / Bohr", ylabel="Max HVP error / (Eh/Bohr²)",
                   title="A mixed Cartesian direction; fixed seed 13")
    axes[1, 0].legend(frameon=False)
    for label, height, color in (("H2O", 1., "#337a9b"), ("D2O", .55, "#c07839")):
        frequencies = data["isotopes"][label]["frequencies_cm1"]
        axes[1, 1].vlines(frequencies, 0., height, color=color, lw=2, label=label)
    axes[1, 1].set(xlabel="Harmonic wavenumber / cm⁻¹", ylabel="Mode markers (arbitrary height)",
                   title="Isotope shifts with the same electronic Hessian", ylim=(0, 1.2))
    axes[1, 1].legend(frameon=False)
    fig.suptitle(f"Water · {method} / Cartesian {data['basis']} · {response}")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"harmonic_checks.{suffix}", dpi=220)
    plt.close(fig)

    coordinates = np.asarray(data["coordinates_bohr"])
    displacements = np.asarray(data["isotopes"]["H2O"]["displacements"])
    frequencies = data["isotopes"]["H2O"]["frequencies_cm1"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    artists = []
    for index, ax in enumerate(axes):
        bonds = [ax.plot([], [], color=".5", lw=3)[0] for _ in (1, 2)]
        atoms = ax.scatter(coordinates[:, 1], coordinates[:, 2], s=[180, 65, 65],
                          c=["#b83737", "#6e91ab", "#6e91ab"], zorder=3)
        ax.set(xlim=(-2., 2.), ylim=(-.7, 1.9), aspect="equal", xlabel="y / Bohr", ylabel="z / Bohr",
               title=f"Mode {index+1}: {frequencies[index]:.1f} cm⁻¹")
        artists.append((bonds, atoms))
    fig.suptitle(f"H₂O / {method} normal modes · displacements scaled to 0.25 Bohr for visibility")


    def frame(number):
        updated = []
        for index, (bonds, atoms) in enumerate(artists):
            vector = displacements[:, :, index]
            moving = coordinates + .25*np.sin(2*np.pi*number/40)*vector/np.max(np.abs(vector))
            atoms.set_offsets(moving[:, 1:])
            for bond, hydrogen in zip(bonds, (1, 2)):
                bond.set_data(moving[[0, hydrogen], 1], moving[[0, hydrogen], 2])
            updated.extend([atoms, *bonds])
        return updated


    animation = FuncAnimation(fig, frame, frames=40, interval=60, blit=False)
    animation.save(output / "water_modes.gif", writer=PillowWriter(fps=16))
    plt.close(fig)


if __name__ == "__main__":
    run()


# Generated on jy_gpu_1, 2026-10-15, from saved PBE AD/reference data:
# G05_water_pbe_hessian/harmonic_checks.png and harmonic_checks.pdf
# G05_water_pbe_hessian/water_modes.gif: 40 frames, 1000 x 320 pixels.
