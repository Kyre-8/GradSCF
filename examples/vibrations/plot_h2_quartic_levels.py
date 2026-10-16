"""Plot the verified H2 quartic model and its separate locality limitations."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


output = Path("reproducibility/ad_showcase/V08_h2_quartic")
model = json.loads((output/"model.json").read_text())
spectrum = json.loads((output/"spectrum.json").read_text())
reference = json.loads((output/"reference.json").read_text())
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
conversion = spectrum["hartree_to_wavenumber"]
orders = {row["powers"][0]: row["coefficient"] for row in model["coefficients"]}
grid = np.linspace(-.6, .8, 401)
potential = sum(coefficient*grid**order for order, coefficient in orders.items())
samples = model["sampled_true_rhf_potential"]
figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
axes[0].plot(grid, potential, color="#28699C", label="AD-derived quartic model")
axes[0].plot([row["displacement_bohr"] for row in samples], [row["energy_relative"] for row in samples],
             "o", ms=4, color="#C16A32", label="Re-converged RHF samples")
for level in spectrum["levels_hartree"][:3]:
    axes[0].axhline(level, color="#8997A2", lw=.7, ls=":")
axes[0].set(xlabel="q = R − Rₑ / bohr", ylabel="Energy above the electronic minimum / Eh",
            title=f"Rₑ = {model['equilibrium_length_bohr']:.9f} bohr")
axes[0].legend(fontsize=9)
quanta = np.arange(1, 6)
harmonic = model["harmonic_frequency_hartree"]*quanta*conversion
anharmonic = np.array(spectrum["transitions_cm1"])[1:]
axes[1].plot(quanta, harmonic, "o--", color="#8997A2", label="Harmonic limit")
axes[1].plot(quanta, anharmonic, "s-", color="#28699C", label="Quartic-model levels")
axes[1].set(xlabel="Excited level rank", ylabel="Excitation energy / cm⁻¹", xticks=quanta,
            title="Model levels; no H₂ dipole IR intensity")
axes[1].legend(fontsize=9)
figure.suptitle("H₂ / RHF / 6-31G: one-coordinate quartic model", fontsize=14)
for suffix in ("png", "pdf"):
    figure.savefig(output/f"model_and_levels.{suffix}", dpi=180)
plt.close(figure)

figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
final = np.array(spectrum["levels_hartree"])
sizes = [row["size"] for row in spectrum["convergence"][:-1]]
errors = [np.max(np.abs(np.array(row["levels_hartree"])-final)) for row in spectrum["convergence"][:-1]]
axes[0].semilogy(sizes, errors, "o-", color="#28699C", label="HO basis, relative to N=48")
axes[0].axhline(reference["max_dvr_ho_level_error"], color="#C16A32", ls="--", label="Independent DVR vs HO")
axes[0].set(xlabel="Oscillator basis size", ylabel="Maximum six-level error / Eh", title="Numerical model convergence")
axes[0].legend(fontsize=8)
domains = reference["harmonic_limit_domain_convergence"]
axes[1].semilogy([abs(row["bounds_bohr"][0]) for row in domains],
                 [row["max_exact_error"] for row in domains], "o-", color="#28699C")
axes[1].axhline(2e-8, color="#C16A32", ls="--", label="Original tolerance")
axes[1].set(xlabel="Magnitude of left boundary / bohr", ylabel="Harmonic-limit error / Eh",
            title="Fixed spacing 0.005 bohr")
axes[1].legend(fontsize=8)
windows = spectrum["sampling_windows"]
tails = np.array([row["probability_outside"] for row in windows]).T
display = axes[2].imshow(100*tails, origin="lower", aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
for row in range(6):
    for column in range(len(windows)):
        percent = 100*tails[row, column]
        label = f"{percent:.1f}" if percent >= .05 else f"{percent:.1e}"
        axes[2].text(column, row, label, ha="center", va="center", fontsize=8)
axes[2].set(xticks=range(5), xticklabels=["±0.3", "±0.4", "±0.5", "±0.6", "−0.6,0.8"],
            yticks=range(6), ylabel="Level rank", xlabel="Sample window in q / bohr",
            title="Probability outside the local window / %")
figure.colorbar(display, ax=axes[2], shrink=.85, label="Probability / %")
figure.suptitle("Numerical convergence and the sampled Taylor validity window", fontsize=13)
for suffix in ("png", "pdf"):
    figure.savefig(output/f"convergence_and_locality.{suffix}", dpi=180)
plt.close(figure)
with (output/"levels.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["level", "energy_above_electronic_minimum_hartree", "excitation_cm-1", "harmonic_excitation_cm-1"])
    for level, value in enumerate(spectrum["levels_hartree"]):
        writer.writerow([level, value, spectrum["transitions_cm1"][level], level*model["harmonic_frequency_hartree"]*conversion])
with (output/"locality_windows.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["level", "q_left_bohr", "q_right_bohr", "probability_outside", "max_sampled_taylor_error_hartree"])
    for window in windows:
        for level, probability in enumerate(window["probability_outside"]):
            writer.writerow([level, *window["interval_bohr"], probability, window["max_sampled_taylor_error_hartree"]])
