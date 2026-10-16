"""Validated ground/excited polarizability tensors and excitation Stark shifts."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants


output = Path("reproducibility/ad_showcase/X08_formaldehyde_cis_stark")
native = json.loads((output/"gradscf.json").read_text())
reference = json.loads((output/"pyscf.json").read_text())
digest = hashlib.sha256((output/"gradscf.json").read_bytes()).hexdigest()
if not native["complete"] or not reference["passed"] or reference["native_sha256"] != digest:
    raise RuntimeError("Only current independently validated X08 records are plotted.")
alpha = np.asarray(native["polarizabilities_au"])
difference = alpha[1]-alpha[0]
hartree_to_mev = 1000*physical_constants["Hartree energy in eV"][0]
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), layout="constrained")
for axis, matrix, labels, title in (
    (axes[0, 0], alpha.reshape(6, 3), ["Ground x", "Ground y", "Ground z", "S1 x", "S1 y", "S1 z"], "Total-energy polarizabilities"),
    (axes[0, 1], difference, ["x", "y", "z"], "Excited minus ground polarizability")):
    limit = max(float(np.abs(matrix).max()), 1e-10)
    artist = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set(xticks=range(3), xticklabels=list("xyz"), yticks=range(len(labels)), yticklabels=labels, title=title)
    for (i, j), value in np.ndenumerate(matrix):
        label = f"{value:.3f}"
        axis.text(j, i, "0.000" if label == "-0.000" else label, ha="center", va="center",
                  color="white" if abs(value) > .65*limit else "black", fontsize=10)
    fig.colorbar(artist, ax=axis, label="Static polarizability / au", shrink=.9)
for axis, component in zip(axes[1], ("x", "z")):
    points = [point for point in native["curves"] if point["axis"] == component]
    refs = [point for point in reference["curves"] if point["axis"] == component]
    field = [point["field_au"] for point in points]
    axis.plot(field, [point["excitation_shift_hartree"]*hartree_to_mev for point in points], "o-", color="#0072B2", label="Native tracked CIS")
    axis.scatter(field, [point["excitation_shift_hartree"]*hartree_to_mev for point in refs], marker="x", color="black", label="Independent PySCF")
    axis.plot(field, [point["quadratic_excitation_shift_hartree"]*hartree_to_mev for point in points], "--", color="#D55E00", label="Local dipole + alpha prediction")
    axis.axhline(0., color=".5", linewidth=.8)
    axis.set(xlabel=f"Electric field F_{component} / au", ylabel="Excitation-energy shift / meV",
             title="Out-of-plane field" if component == "x" else "C=O-axis field")
    axis.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    axis.legend(frameon=False, fontsize=9)
fig.suptitle("Formaldehyde S1 · RHF/CIS · Cartesian 6-31+G*\n"
             "Clamped nuclei at the RHF minimum; ground and total excited energies both respond", fontsize=13)
for suffix in ("png", "pdf"):
    fig.savefig(output/f"stark_polarizability.{suffix}", dpi=220)
plt.close(fig)
with (output/"electric_tensors.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["state", "component", "dipole_au", "alpha_x_au", "alpha_y_au", "alpha_z_au"])
    for label, mu, tensor in zip(("ground", "S1_total", "S1_minus_ground"),
        [*native["dipoles_au"], native["dipole_difference_au"]], [alpha[0], alpha[1], difference]):
        for i in range(3):
            writer.writerow([label, "xyz"[i], mu[i], *tensor[i]])
with (output/"stark_curves.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["field_axis", "field_au", "ground_energy_hartree", "total_excited_energy_hartree",
                     "ground_quadratic_hartree", "excited_quadratic_hartree", "excitation_shift_hartree", "quadratic_excitation_shift_hartree"])
    for point in native["curves"]:
        writer.writerow([point["axis"], point["field_au"], *point["energies_hartree"], *point["quadratic_energies_hartree"],
                         point["excitation_shift_hartree"], point["quadratic_excitation_shift_hartree"]])
print("Saved X08 stark_polarizability.png/.pdf and tensor/curve CSV files", flush=True)
