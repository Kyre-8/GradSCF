"""Read-only figures for the matched X15 diagonal water vertex comparison."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


folder = Path("reproducibility/ad_showcase/X15_water_matched_vertices")
vertex = json.loads((folder/"gradscf.json").read_text())
fd = json.loads((folder/"finite_differences.json").read_text())
fc = json.loads((folder/"diagonal_vibronic.json").read_text())
assert fd["validation_passed"] and fd["complete"] and fc["validation_passed"]
for name, expected in fc["input_sha256"].items():
    assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == expected
arrays = np.load(folder/"diagonal_vibronic.npz")
inputs = np.load(folder/"vertices.npz")
hartree_to_ev = 27.211386245988  # Same conversion as gradscf.tools.spectra.
colors = ("#4477aa", "#ee7733")
labels = ("Frozen GW correction/kernel", "Relaxed G0W0/TDA")
couplings = np.asarray([vertex["frozen_g_hartree"], vertex["relaxed_g_hartree"]])*hartree_to_ev
frequencies = vertex["frequencies_cm1"]
states = np.arange(vertex["nstates"])
mode_names = ("Bend", "Symmetric stretch", "Antisymmetric stretch")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(12.4, 7.2), constrained_layout=True)
layout = fig.add_gridspec(2, 3, height_ratios=(1, 1.35))
maximum = 1.2*np.max(np.abs(couplings))
for mode in range(3):
    ax = fig.add_subplot(layout[0, mode])
    for model in range(2):
        ax.bar(states+(model-.5)*.34, couplings[model, mode], .32,
               color=colors[model], label=labels[model])
    ax.axhline(0, color=".5", lw=.7)
    ax.set(xticks=states, xticklabels=[f"S{s+1}" for s in states], ylim=(-maximum, maximum),
           title=f"{mode_names[mode]} / {frequencies[mode]:.1f} cm$^{{-1}}$")
    if mode == 0:
        ax.set_ylabel("Diagonal vertex g / eV")
        ax.legend(frameon=False, fontsize=8)
    if mode == 2:
        ax.text(.5, .77, "Symmetry-forced diagonal zero\nOffdiagonal couplings omitted in both models",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)

ax = fig.add_subplot(layout[1, :2])
grid, curves = arrays["energy_ev"], arrays["curves"]
ax.plot(grid, arrays["bare_curves"][0], color=".65", lw=1., label="Common electronic reference")
for model in range(2):
    ax.plot(grid, curves[model, 0], color=colors[model], lw=1.4, label=labels[model])
energies = np.asarray(vertex["excitation_hartree"])*hartree_to_ev
low = np.min(fc["zero_zero_hartree"])*hartree_to_ev-.5
high = energies.max()+5*inputs["omega"].max()*hartree_to_ev
ax.set(xlabel="Photon energy / eV", ylabel="Oscillator-strength density / eV$^{-1}$",
       xlim=(low, high), title="0 K diagonal Condon progressions; common HWHM 0.05 eV")
ax.legend(frameon=False, fontsize=8)
ax = fig.add_subplot(layout[1, 2])
reorganization = np.asarray(fc["reorganization_hartree"])*hartree_to_ev
for model in range(2):
    ax.bar(states+(model-.5)*.34, reorganization[model], .32, color=colors[model])
ax.set(xticks=states, xticklabels=[f"S{s+1}" for s in states], ylabel="Reorganization energy / eV",
       title="Local linear-model displacement energy")
fig.suptitle("Water / Cartesian 6-31G*: matched frozen and relaxed exciton-mode vertices")
for extension in ("png", "pdf"):
    fig.savefig(folder/("matched_vertices."+extension), dpi=180)
plt.close(fig)

fig = plt.figure(figsize=(12.4, 7.), constrained_layout=True)
layout = fig.add_gridspec(2, 3)
keys = (("qp", "QP energies"), ("relaxed", "Relaxed TDA"),
        ("frozen", "Frozen projection"), ("transported_fock", "Transported HF Fock"))
for mode in range(3):
    ax = fig.add_subplot(layout[0, mode])
    rows = [entry for entry in fd["records"] if entry["mode"] == mode]
    steps = np.asarray([entry["step_Q"] for entry in rows])
    for key, label in keys:
        ax.loglog(steps, [entry["errors"][key] for entry in rows], "o-", ms=4, label=label)
    ax.invert_xaxis()
    ax.set(xlabel="Mass-weighted displacement dQ", title=mode_names[mode])
    if mode == 0:
        ax.set_ylabel("AD vs reconverged difference / Eh Q$^{-1}$")
        ax.legend(frameon=False, fontsize=8)
ax = fig.add_subplot(layout[1, :])
cutoffs = np.asarray([entry["max_quanta"] for entry in fc["cutoffs"]])
ax.semilogy(cutoffs, [entry["maximum_poisson_tail"] for entry in fc["cutoffs"]], "o-",
            color=colors[0], label="Largest raw Poisson tail")
changes = [np.max(entry["relative_l1_change"]) for entry in fc["cutoffs"][1:]]
ax.semilogy(cutoffs[1:], changes, "s-", color=colors[1], label="Largest spectrum change, both widths")
ax.axhline(1e-10, color=colors[0], ls="--", lw=.8, label="Raw-tail target 1e-10")
ax.axhline(1e-6, color=colors[1], ls="--", lw=.8, label="Spectrum-change target 1e-6")
ax.set(xlabel="Maximum total vibrational quanta", ylabel="Dimensionless error",
       title="Raw cutoff checks: probabilities are not renormalized")
ax.legend(frameon=False, fontsize=9, ncol=2)
fig.suptitle("Matched model verification: all modes, both transported frames, and FC cutoff")
for extension in ("png", "pdf"):
    fig.savefig(folder/("validation."+extension), dpi=180)

rows = [[mode, state, frequencies[mode], couplings[0, mode, state], couplings[1, mode, state]]
        for mode in range(3) for state in states]
np.savetxt(folder/"vertices.csv", rows, delimiter=",", comments="",
           header="mode,state,frequency_cm1,frozen_g_eV,relaxed_g_eV")
