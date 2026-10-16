"""Saved one-phonon thermal absorption; spectra retain their raw strengths."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("reproducibility/ad_showcase/X15_water_bse_temperature")
data = np.load(OUTPUT/"temperature.npz")
native = json.loads((OUTPUT/"gradscf.json").read_text())
reference = json.loads((OUTPUT/"scipy.json").read_text())
assert native["complete"] and reference["complete"]
energy = data["photon_energy_ev"]
curves = data["cross_sections_mb"]
colors = ("#315c86", "#35978f", "#d99b46", "#b34c49")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
for ti, (temperature, color) in enumerate(zip(data["temperatures_kelvin"], colors)):
    for mi, ax in enumerate(axes[0]):
        ax.plot(energy, curves[ti, mi, 0], color=color, label=f"{temperature:g} K", lw=1.1)
for ax, name in zip(axes[0], ("One-phonon Fan", "One-phonon Fan + static DW")):
    ax.set(title=name+" · η = 10 meV", ylabel="Absorption cross section / Mb")
axes[0, 0].legend(frameon=False, fontsize=9)
axes[1, 0].plot(energy, data["temperature_derivative_mb_per_kelvin"], color=colors[2], lw=1.)
axes[1, 0].axhline(0., color=".6", lw=.6)
axes[1, 0].set(title="Fan + DW · complete temperature JVP at 600 K",
               ylabel=r"$\partial\sigma/\partial T$ / Mb K$^{-1}$")
for wi, (width, style) in enumerate(zip(data["widths_ev"], ("-", "--"))):
    axes[1, 1].plot(energy, curves[2, 1, wi], style, color=colors[2-wi], lw=1.,
                    label=f"η = {1000*width:g} meV")
axes[1, 1].set(title="Same 600 K model · numerical width comparison", ylabel="Absorption cross section / Mb")
axes[1, 1].legend(frameon=False, fontsize=9)
for ax in axes.flat:
    ax.set_xlim(6., 15.); ax.grid(alpha=.15)
for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
    ax.set_xlim(10.85, 11.2)
axes[0, 1].set_title("Fan + DW · strongest line group · η = 10 meV")
axes[1, 0].set_title("Fan + DW · temperature JVP at 600 K · zoom")
axes[1, 1].set_title("Same 600 K model · numerical width · zoom")
for ax in axes[1]:
    ax.set_xlabel("Photon energy / eV")
fig.suptitle("Water · RHF/3-21G → G₀W₀ → three-state TDA-BSE\n"
             "Frozen electronic kernel and RHF vertices; temperature changes the Bose bath", fontsize=13)
fig.savefig(OUTPUT/"water_bse_temperature.png", dpi=210)
fig.savefig(OUTPUT/"water_bse_temperature.pdf")
plt.close(fig)

# Recorded outputs: water_bse_temperature.png and water_bse_temperature.pdf.
# Final full-range/zoom panels were inspected after native and reference passes.
