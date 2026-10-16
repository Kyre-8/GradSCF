"""Plot completed two-stretch model checks, physical dipole strengths and limits."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants


output = Path("reproducibility/ad_showcase/V08_water_two_stretch")
model = json.loads((output/"model.json").read_text())
spectrum = json.loads((output/"spectrum.json").read_text())
electronic = json.loads((output/"electronic_reference.json").read_text())
reference = json.loads((output/"dvr_reference.json").read_text())
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
conversion = physical_constants["hartree-inverse meter relationship"][0]/100
frequencies = np.array(spectrum["frequencies_cm1"])
strengths = np.array(spectrum["dipole_strengths_au"])
dvr_strengths = np.array(reference["grid_convergence"][-1]["dipole_strengths_au"])


def save(figure, name):
    for suffix in ("png", "pdf"):
        figure.savefig(output/f"{name}.{suffix}", dpi=180)
    plt.close(figure)


figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
for ax, selected, title in zip(axes, (np.arange(2), np.arange(2, 5)),
                               ("Fundamental region", "Overtone / combination region")):
    ax.vlines(frequencies[selected], 0, strengths[selected], color="#28699C", lw=2)
    ax.plot(frequencies[selected], strengths[selected], "o", color="#28699C", label="Shared solver, HO 22 × 22")
    ax.plot(frequencies[selected], dvr_strengths[selected], "x", color="#C16A32", ms=8, label="Independent sinc DVR")
    for state in selected:
        component = spectrum["dominant_harmonic_components"][state+1][0]
        label = f"{tuple(component['occupation'])}, {100*component['weight']:.0f}%"
        ax.annotate(label, (frequencies[state], strengths[state]), (0, 8), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set(xlabel="Model transition / cm⁻¹", ylabel="Dipole strength / e² bohr²", title=title,
            xlim=(frequencies[selected].min()-150, frequencies[selected].max()+150),
            ylim=(0, strengths[selected].max()*1.35))
axes[0].legend(fontsize=8, loc="upper left")
figure.suptitle("Water RHF/6-31G*: two-stretch model, bend and rotor frozen", fontsize=13)
save(figure, "stretch_spectrum")

figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
grid = np.linspace(-25., 25., 101)
first, second = np.meshgrid(grid, grid, indexing="ij")
potential = sum(row["coefficient"]*first**row["powers"][0]*second**row["powers"][1] for row in model["coefficients"])
levels = axes[0].contourf(first, second, potential, levels=14, cmap="viridis")
axes[0].plot([-20, 20, 20, -20, -20], [-20, -20, 20, 20, -20], "w--", lw=1)
figure.colorbar(levels, ax=axes[0], label="Quartic-model energy / Eh")
axes[0].set(title="Certified polynomial; dashed sampling square",
            xlabel="Q₁ / bohr √mₑ", ylabel="Q₂ / bohr √mₑ")
coordinates = np.array([row["q"] for row in model["samples"]])
errors = np.array([row["energy_error"] for row in model["samples"]])
scale = max(abs(errors))
points = axes[1].scatter(coordinates[:, 0], coordinates[:, 1], c=errors, marker="s", s=130,
                         cmap="RdBu_r", vmin=-scale, vmax=scale)
figure.colorbar(points, ax=axes[1], label="Sampled quartic − RHF / Eh")
axes[1].set(title="Independent RHF-checked sample points",
            xlabel="Q₁ / bohr √mₑ", ylabel="Q₂ / bohr √mₑ", xlim=(-23, 23), ylim=(-23, 23))
selected_basis = ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
vectors = np.array(spectrum["coefficients_in_oscillator_basis"])
weights = np.array([vectors[22*a+b, 1:6]**2 for a, b in selected_basis]).T
axes[2].imshow(weights, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="Blues")
for row in range(5):
    for column in range(5):
        axes[2].text(column, row, f"{100*weights[row, column]:.1f}", ha="center", va="center",
                      color="white" if weights[row, column] > .55 else "black", fontsize=8)
axes[2].set(xticks=range(5), xticklabels=[str(x) for x in selected_basis],
            yticks=range(5), yticklabels=[f"{x:.1f}" for x in frequencies[:5]],
            xlabel="Selected harmonic occupation", ylabel="Model transition / cm⁻¹",
            title="Raw harmonic-basis weights / %")
figure.suptitle("Coupled polynomial structure, sampled locality and state mixing", fontsize=13)
save(figure, "coupling_and_surface")

figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
final = np.array(spectrum["levels_hartree"])
axes[0].semilogy([row["size_per_mode"] for row in spectrum["convergence"][:-1]],
    [np.max(np.abs(np.array(row["levels_hartree"])-final)) for row in spectrum["convergence"][:-1]],
    "o-", color="#28699C")
axes[0].axhline(reference["max_level_error"], color="#C16A32", ls="--", label="Independent DVR vs final HO")
axes[0].set(xlabel="HO basis size per coordinate", ylabel="Maximum 12-level error / Eh", title="Basis convergence")
axes[0].legend(fontsize=8)
axes[1].plot([row["spacing"] for row in reference["grid_convergence"]],
    [np.max(np.abs(np.array(row["levels_hartree"])-final)) for row in reference["grid_convergence"]],
    "o-", color="#C16A32")
axes[1].set(xlabel="DVR spacing / bohr √mₑ", ylabel="Maximum 12-level error / Eh",
            title="Fixed square |Qᵢ| ≤ 45")
windows = spectrum["sampling_windows"]
tails = np.array([row["probability_outside"][:6] for row in windows]).T
display = axes[2].imshow(100*tails, origin="lower", aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
for row in range(6):
    for column in range(len(windows)):
        percent = 100*tails[row, column]
        label = f"{percent:.1f}" if percent >= .05 else f"{percent:.1e}"
        axes[2].text(column, row, label, ha="center", va="center", fontsize=8)
axes[2].set(xticks=range(4), xticklabels=[str(row["half_width_mass_weighted_bohr"]) for row in windows],
            yticks=range(6), xlabel="Sampling-square half width / bohr √mₑ", ylabel="Level rank",
            title="Probability outside the sampled square / %")
figure.colorbar(display, ax=axes[2], shrink=.85, label="Probability / %")
figure.suptitle("Numerical model convergence and physical sampling-window diagnostics", fontsize=13)
save(figure, "convergence_and_locality")

with (output/"spectrum.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["level", "frequency_cm-1", "dipole_strength_au", "oscillator_strength", "DVR_dipole_strength_au",
                     "dominant_n1", "dominant_n2", "dominant_weight"])
    for index, (frequency, strength) in enumerate(zip(frequencies, strengths), 1):
        component = spectrum["dominant_harmonic_components"][index][0]
        writer.writerow([index, frequency, strength, spectrum["oscillator_strengths"][index-1],
                         dvr_strengths[index-1], *component["occupation"], component["weight"]])
with (output/"locality_windows.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["level", "half_width_Q", "probability_outside", "max_sampled_energy_error_Eh", "max_sampled_dipole_error_au"])
    for window in windows:
        for level, probability in enumerate(window["probability_outside"]):
            writer.writerow([level, window["half_width_mass_weighted_bohr"], probability,
                             window["max_sampled_taylor_energy_error"], window["max_sampled_dipole_error"]])
with (output/"convergence.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["method", "size_per_coordinate", "half_width_Q", "level", "energy_Eh", "error_to_final_HO_Eh"])
    for method, rows in (("HO", spectrum["convergence"]), ("DVR grid", reference["grid_convergence"]),
                          ("DVR domain", reference["domain_convergence"])):
        for row in rows:
            for level, energy in enumerate(row["levels_hartree"]):
                writer.writerow([method, row.get("size_per_mode", row.get("size_per_coordinate")),
                                 row.get("half_width", ""), level, energy, energy-final[level]])
