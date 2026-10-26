"""Plot the saved X11 unit-Condon model, projection diagnostics and convergence."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("reproducibility/ad_showcase/X11_formaldehyde_franck_condon")
data = json.loads((OUTPUT/"gradscf.json").read_text())
assert data["complete"]
model = data["harmonic_model"]
spectrum = np.loadtxt(OUTPUT/"spectrum.csv", delimiter=",", skiprows=1)
hartree_to_ev = 27.211386245988
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4), layout="constrained")
for ax, key, title in ((axes[0, 0], "raw_duschinsky_overlap", "Endpoint mode overlap: diagnostic only"),
                       (axes[0, 1], "canonical_rotation", "Canonical rotation of the common-slice model")):
    values = np.asarray(model[key])**2
    shown = ax.imshow(values, vmin=0, vmax=1, cmap="Blues")
    ax.set(title=title, xticks=range(6), xticklabels=range(1, 7),
           yticks=range(6), yticklabels=range(1, 7), xlabel="Ground mode index", ylabel="Final mode index")
    fig.colorbar(shown, ax=ax, shrink=.78, label="Squared matrix element")
    for row in range(6):
        for column in range(6):
            if values[row, column] > .01:
                ax.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center",
                        color="white" if values[row, column] > .6 else ".1", fontsize=8)
axes[0, 0].text(.02, .02, f"max |JᵀJ − I| = {model['raw_J_orthogonality_max']:.5f}",
                transform=axes[0, 0].transAxes, fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
changes = np.asarray(model["projection_frequency_changes_cm1"])
axes[0, 2].bar(range(1, 7), changes, color="#0072B2")
axes[0, 2].axhline(0, color=".4", lw=.8)
axes[0, 2].set(title="The projection changes the harmonic model", xlabel="Final mode index",
               ylabel="Model − original excited frequency / cm⁻¹", ylim=(-68, 7))
for index, value in enumerate(changes, 1):
    axes[0, 2].text(index, value-2.5, f"{value:.2f}", ha="center", va="top", fontsize=8)
zero_zero = model["model_zero_zero_hartree"]*hartree_to_ev
old_zero_zero = model["X10_zero_zero_hartree"]*hartree_to_ev
axes[0, 2].text(.02, .08, f"Model 0–0: {zero_zero:.6f} eV\nOriginal X10: {old_zero_zero:.6f} eV",
                transform=axes[0, 2].transAxes, fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.95))

for index, (sigma, color) in enumerate(((.02, "#0072B2"), (.01, "#D55E00")), 1):
    axes[1, 0].plot(spectrum[:, 0], spectrum[:, index], color=color, label=f"Gaussian σ = {sigma:.2f} eV")
axes[1, 0].axvline(zero_zero, color=".4", linestyle=":", label="Model 0–0")
axes[1, 0].set(title="Unit-Condon probability envelope", xlabel="Photon energy / eV",
               ylabel="Probability density / eV⁻¹")
axes[1, 0].legend(frameon=False, fontsize=8)
lines = data["dominant_lines"][:20]
x = np.array([row["photon_energy_ev"]-zero_zero for row in lines])
y = np.array([row["probability"] for row in lines])
axes[1, 1].vlines(x, 0, y, color="#0072B2", lw=1.5)
axes[1, 1].plot(x, y, "o", color="#0072B2", ms=3)
axes[1, 1].set(title="Twenty largest raw Franck–Condon factors", xlabel="Energy above model 0–0 / eV",
               ylabel="Transition probability", ylim=(0, max(y)*1.32))
for index in range(4):
    label = ",".join(map(str, lines[index]["occupations"]))
    axes[1, 1].annotate(f"({label})", (x[index], y[index]), xytext=(0, 6+index*5),
                        textcoords="offset points", ha="center", fontsize=7)
convergence = data["convergence"]
cutoffs = [row["max_quanta"] for row in convergence]
axes[1, 2].semilogy(cutoffs, [max(row["missing_probability"], 1e-16) for row in convergence],
                    "o-", color="#0072B2", label="Unresolved total probability")
axes[1, 2].semilogy(cutoffs[1:], [max(max(row["relative_profile_l1_change"]), 1e-16) for row in convergence[1:]],
                    "s--", color="#D55E00", label="Profile change (larger of two σ)")
axes[1, 2].axhline(1e-8, color=".5", linestyle=":", label="Probability-tail gate")
axes[1, 2].set(title="Raw weight and spectral convergence", xlabel="Maximum total vibrational quanta",
               ylabel="Missing probability / relative L1 change")
axes[1, 2].legend(frameon=True, facecolor="white", framealpha=1, fontsize=8)
fig.suptitle("Formaldehyde RHF/CIS · common-ground-Eckart-slice double-harmonic model\n"
             "Unit Condon probabilities; no absolute dark-state absorption or Herzberg–Teller intensity", fontsize=12)
for suffix in ("png", "pdf"):
    fig.savefig(OUTPUT/f"franck_condon_model.{suffix}", dpi=220)
plt.close(fig)
(OUTPUT/"plot_settings.json").write_text(json.dumps(dict(
    probabilities="Raw, not renormalized after state truncation or finite spectral window.",
    spectrum="Gaussian illustration with sigma=.02 and .01 eV; no fitted lifetime or experimental broadening.",
    matrix_display="Squared raw endpoint overlap and model canonical rotation are distinct matrices.",
    stick_selection="Twenty largest raw factors; the smooth curve uses all retained weight except the reported <=1e-11 plotting screen.",
    scope=data["scope"]), indent=2)+"\n")
print("Saved X11 model, probability spectrum and Duschinsky diagnostics")
