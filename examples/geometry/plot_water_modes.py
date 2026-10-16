"""Plot saved optimization, Hessian checks and isotope frequencies; animate modes.

No SCF is run. Mode displacements are scaled for visibility, not thermal
amplitudes. Frequency markers have unit height and are not IR intensities.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

root = Path("reproducibility/ad_showcase")
optimization = json.loads((root / "G04_water_rhf_optimization/gradscf.json").read_text())
output = root / "G05_water_rhf_hessian"
data = json.loads((output / "gradscf.json").read_text())
reference = json.loads((output / "pyscf.json").read_text())
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
history = optimization["history"]
axes[0].semilogy(np.arange(1, len(history)+1), [row["max_gradient"] for row in history],
                 "o-", color="#337a9b")
axes[0].axhline(1e-7, color=".5", ls="--")
axes[0].set(xlabel="SCF evaluation (includes line search)", ylabel="Max |gradient| / (Eh/Bohr)",
            title="Water geometry optimization")
error = np.array(data["hessian_hartree_per_bohr2"]) - reference["hessian_hartree_per_bohr2"]
scale = max(np.max(np.abs(error)), 1e-15)
im = axes[1].imshow(error, cmap="RdBu_r", vmin=-scale, vmax=scale)
axes[1].set(xlabel="Cartesian component", ylabel="Cartesian component", title="AD minus PySCF Hessian")
fig.colorbar(im, ax=axes[1], label="Eh/Bohr²", shrink=.75)
for label, offset, color in (("H2O", 1., "#337a9b"), ("D2O", .55, "#c07839")):
    frequencies = data["isotopes"][label]["frequencies_cm1"]
    axes[2].vlines(frequencies, 0., offset, color=color, linewidth=2, label=label)
axes[2].set(xlabel="Harmonic wavenumber / cm⁻¹", ylabel="Unit-height markers (offset)",
            title="Isotope shifts; fixed electronic Hessian", ylim=(0, 1.2))
axes[2].legend(frameon=False)
fig.suptitle("Water · RHF / Cartesian 6-31G* · analytic stationary response")
for extension in ("png", "pdf"):
    fig.savefig(output / f"harmonic_checks.{extension}", dpi=220)
plt.close(fig)

coordinates = np.array(data["coordinates_bohr"])
displacements = np.array(data["isotopes"]["H2O"]["displacements"])
frequencies = data["isotopes"]["H2O"]["frequencies_cm1"]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
artists = []
for index, ax in enumerate(axes):
    bonds = [ax.plot([], [], color=".5", lw=3)[0] for _ in (1, 2)]
    atoms = ax.scatter(coordinates[:, 1], coordinates[:, 2], s=[180, 65, 65],
                      c=["#b83737", "#6e91ab", "#6e91ab"], zorder=3)
    ax.set(xlim=(-2.0, 2.0), ylim=(-.7, 1.9), aspect="equal",
           xlabel="y / Bohr", ylabel="z / Bohr", title=f"Mode {index+1}: {frequencies[index]:.1f} cm⁻¹")
    artists.append((bonds, atoms))
fig.suptitle("H₂O normal modes · displacements scaled to 0.25 Bohr for visibility")


def frame(number):
    updated = []
    for index, (bonds, atoms) in enumerate(artists):
        vector = displacements[:, :, index]
        moving = coordinates + .25 * np.sin(2*np.pi*number/40) * vector / np.max(np.abs(vector))
        atoms.set_offsets(moving[:, 1:])
        for bond, hydrogen in zip(bonds, (1, 2)):
            bond.set_data(moving[[0, hydrogen], 1], moving[[0, hydrogen], 2])
        updated.extend([atoms, *bonds])
    return updated


animation = FuncAnimation(fig, frame, frames=40, interval=60, blit=False)
animation.save(output / "water_modes.gif", writer=PillowWriter(fps=16))
plt.close(fig)
