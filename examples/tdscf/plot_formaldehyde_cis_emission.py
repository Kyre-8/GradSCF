"""Plot validated X10 energy endpoints, gap definitions and harmonic checks."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


output = Path("reproducibility/ad_showcase/X10_formaldehyde_cis_emission")
native = json.loads((output/"emission.json").read_text())
reference = json.loads((output/"emission_pyscf.json").read_text())
excited = json.loads((output/"excited_ad.json").read_text())
excited_reference = json.loads((output/"excited_pyscf.json").read_text())
for native_file, result in (("emission.json", reference), ("excited_ad.json", excited_reference)):
    if not result["passed"] or result["native_sha256"] != hashlib.sha256((output/native_file).read_bytes()).hexdigest():
        raise RuntimeError("Only current independently validated results are plotted.")
if not native["complete"] or not excited["complete"]:
    raise RuntimeError("Complete both native stages first.")
if (native["excited_native_sha256"] != hashlib.sha256((output/"excited_ad.json").read_bytes()).hexdigest()
        or reference["excited_reference_sha256"] != hashlib.sha256((output/"excited_pyscf.json").read_bytes()).hexdigest()):
    raise RuntimeError("The paired-minimum calculation is linked to a different excited Hessian.")
hartree_to_ev = 27.211386245988
values = native["electronic_energies_hartree"]
origin = values["ground_at_ground"]
relative = {key: (value-origin)*hartree_to_ev for key, value in values.items()}
# Align the endpoint body frames before comparing mass-normalized mode vectors.
# This is an endpoint displacement-character assignment, not continuous mode
# tracking or a calculation of vibronic intensities.
masses = np.asarray(native["masses_amu"])
ground_coordinates = np.asarray(native["ground_coordinates_bohr"])
excited_coordinates = np.asarray(native["excited_coordinates_bohr"])
ground_centered = ground_coordinates-np.average(ground_coordinates, axis=0, weights=masses)
excited_centered = excited_coordinates-np.average(excited_coordinates, axis=0, weights=masses)
left, _, right = np.linalg.svd((excited_centered*masses[:, None]).T@ground_centered)
rotation = left@np.diag([1., 1., np.linalg.det(left@right)])@right
excited_modes = np.einsum("aim,ij->ajm", np.asarray(excited["displacements"]), rotation)
ground_modes = np.asarray(native["ground_displacements"])
overlaps = np.einsum("aim,ain,a->mn", ground_modes, excited_modes,
                     masses*excited["amu_to_electron_mass"])
rows, columns = linear_sum_assignment(-np.abs(overlaps)**2)
permutation = columns[np.argsort(rows)]
scores = np.abs(overlaps[np.arange(6), permutation])
competitors = np.array([np.max(np.abs(np.delete(overlaps[i], permutation[i]))) for i in range(6)])
ambiguous = (scores < .75) | (scores-competitors < .10)
assignment = dict(metric="Absolute mass-weighted endpoint displacement overlap after proper rigid alignment",
                  ground_to_excited_zero_based=permutation.tolist(), overlap_matrix=overlaps.tolist(),
                  assigned_absolute_overlaps=scores.tolist(), runner_up_overlaps=competitors.tolist(),
                  ambiguous=ambiguous.tolist(), ambiguity_overlap_threshold=.75, ambiguity_margin=.10,
                  excited_to_ground_rotation=rotation.tolist())
(output/"mode_assignment.json").write_text(json.dumps(assignment, indent=2)+"\n")
summary = dict(minimum_excited_displacement_overlap=min(item["overlap"] for item in excited["tracking"]),
               minimum_excited_displacement_gap_hartree=min(item["gap_hartree"] for item in excited["tracking"]),
               minimum_continuation_overlap=min(item["overlap"] for item in native["continuation"]),
               minimum_continuation_gap_hartree=min(item["gap_hartree"] for item in native["continuation"]),
               mode_assignment=assignment)
(output/"validation_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
colors = ("#0072B2", "#D55E00")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.1), layout="constrained")
ax = axes[0, 0]
for x, suffix in ((0., "ground"), (1., "excited")):
    for prefix, color in zip(("ground", "excited"), colors):
        value = relative[f"{prefix}_at_{suffix}"]
        ax.plot([x-.16, x+.16], [value, value], color=color, linewidth=3)
for prefix, color in zip(("ground", "excited"), colors):
    ax.plot([.16, .84], [relative[f"{prefix}_at_ground"], relative[f"{prefix}_at_excited"]],
            linestyle=":", linewidth=1.2, color=color)
for x, label, lower, upper in ((-.22, "Absorption", 0., relative["excited_at_ground"]),
                               (1.22, "Emission", relative["ground_at_excited"], relative["excited_at_excited"])):
    start, end = (lower, upper) if label == "Absorption" else (upper, lower)
    ax.annotate("", xy=(x, end), xytext=(x, start), arrowprops=dict(arrowstyle="->", color=".15", lw=1.5))
    ax.text(x+(-.06 if label == "Absorption" else .06), (lower+upper)/2, label,
            rotation=90, va="center", ha="right" if label == "Absorption" else "left")
zpe_levels = ((0., native["ground_harmonic_zpe_hartree"]*hartree_to_ev),
              (1., relative["excited_at_excited"]+native["excited_harmonic_zpe_hartree"]*hartree_to_ev))
for x, level in zpe_levels:
    ax.plot([x-.16, x+.16], [level, level], "--", color=".4", linewidth=1.5)
ax.set(xlim=(-.5, 1.5), ylim=(-.2, max(y for _, y in zpe_levels)+.35),
       xticks=[0., 1.], xticklabels=["Ground minimum $R_g$", "Excited minimum $R_e$"],
       ylabel="Energy relative to $E_0(R_g)$ / eV", title="Electronic endpoints and harmonic ZPE")
ax.text(.5, .24, "Dotted lines connect electronic endpoints", transform=ax.transAxes,
        fontsize=8.5, ha="center")
ax.plot([], [], color=colors[0], label="RHF ground")
ax.plot([], [], color=colors[1], label="CIS excited")
ax.plot([], [], "--", color=".4", label="Minimum + ZPE")
ax.legend(frameon=False, fontsize=8.5, loc="center", bbox_to_anchor=(.56, .54))

keys = ("vertical_absorption", "vertical_emission", "adiabatic_electronic", "harmonic_zero_zero")
labels = ("Vertical\nabsorption", "Vertical\nemission", "Electronic\nadiabatic", "Harmonic\n0–0")
gaps = np.array([native["gaps_hartree"][key]*hartree_to_ev for key in keys])
ax = axes[0, 1]
ax.bar(np.arange(4), gaps, color=["#0072B2", "#D55E00", "#009E73", "#7B6AA8"], width=.65)
for index, value in enumerate(gaps):
    ax.text(index, value+.045, f"{value:.4f}", ha="center", fontsize=10)
ax.set(xticks=np.arange(4), xticklabels=labels, ylabel="Gap / eV", ylim=(0., max(gaps)*1.15),
       title="Four distinct energy differences")

ax = axes[1, 0]
positions = np.arange(1, 7)
ax.bar(positions-.17, native["ground_frequencies_cm1"], width=.32, color=colors[0], label="Ground minimum")
bars = ax.bar(positions+.17, np.asarray(excited["frequencies_cm1"])[permutation], width=.32,
              color=colors[1], label="Excited: endpoint-overlap match")
for bar, uncertain in zip(bars, ambiguous):
    if uncertain:
        bar.set_hatch("///")
ax.set(xticks=positions, xlabel="Ground mode, ordered by frequency", ylabel="Harmonic wavenumber / cm$^{-1}$",
       title="Positive internal modes at both minima")
ax.legend(frameon=False, fontsize=9)

ax = axes[1, 1]
error = np.asarray(excited["hessian_hartree_per_bohr2"])-np.asarray(excited_reference["hessian_hartree_per_bohr2"])
limit = np.max(np.abs(error))*1e6
shown = ax.imshow(error*1e6, cmap="RdBu_r", vmin=-limit, vmax=limit)
atom_axes = [f"{symbol}{axis}" for symbol in ("C", "O", "H1", "H2") for axis in "xyz"]
ax.set(xticks=np.arange(12), yticks=np.arange(12), xticklabels=atom_axes, yticklabels=atom_axes,
       title=f"Excited AD − independent FD Hessian\nMax error {np.max(np.abs(error)):.2e} Eh/Bohr²")
ax.tick_params(labelsize=8)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.colorbar(shown, ax=ax, shrink=.8, label="Difference / (µEh/Bohr²)")
fig.suptitle("Formaldehyde · RHF/CIS · Cartesian 6-31+G*\n"
             "Paired local minima and harmonic 0–0 estimate; no anharmonic correction", fontsize=13)
for suffix in ("png", "pdf"):
    fig.savefig(output/f"energy_bookkeeping.{suffix}", dpi=220)
plt.close(fig)
with (output/"gaps.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["quantity", "native_hartree", "independent_hartree", "native_ev"])
    for key, value in native["gaps_hartree"].items():
        writer.writerow([key, value, reference["gaps_hartree"][key], value*hartree_to_ev])
with (output/"modes.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["ground_frequency_order", "excited_frequency_order", "ground_cm1", "matched_excited_cm1",
                     "shift_cm1", "absolute_overlap", "ambiguous", "independent_ground_cm1", "independent_excited_cm1"])
    for index in range(6):
        matched = permutation[index]
        writer.writerow([index+1, int(matched)+1, native["ground_frequencies_cm1"][index],
                         excited["frequencies_cm1"][matched],
                         excited["frequencies_cm1"][matched]-native["ground_frequencies_cm1"][index],
                         scores[index], bool(ambiguous[index]), reference["ground_frequencies_cm1"][index],
                         excited_reference["frequencies_cm1"][matched]])
print("Saved energy_bookkeeping.png/.pdf, gaps.csv, modes.csv, mode_assignment.json and validation_summary.json")

# Measured 2026-10-15: all six endpoint mode matches have |overlap| >= 0.980904;
# no assignment crosses the documented ambiguity thresholds. Figures use only
# the linked native and independently validated X10 records, generated on jy_gpu_1.
