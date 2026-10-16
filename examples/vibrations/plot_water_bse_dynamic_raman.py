"""Water full-BSE response: the same electronic model, three optical widths."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants


OUTPUT = Path("reproducibility/ad_showcase/V06_water_bse_dynamic_raman")
data = np.load(OUTPUT/"response.npz")
report = json.loads((OUTPUT/"gradscf.json").read_text())
oracle = json.loads((OUTPUT/"scipy.json").read_text())
assert report["complete"] and oracle["complete"]
energy = data["frequencies_hartree"]*physical_constants["Hartree energy in eV"][0]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight"})
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3), sharex=True, constrained_layout=True)
colors = ("#8c9eae", "#376994", "#c25a3c")
for width, color, alpha, activities in zip(data["dampings_hartree"], colors,
                                         data["polarizability"], data["activities_angstrom4_amu"]):
    label = rf"$\eta={width:g}\ E_h$"
    axes[0, 0].plot(energy, np.trace(alpha.imag, axis1=-2, axis2=-1)/3,
                    color=color, label=label, lw=1.35)
    for mode, ax in enumerate(axes.flat[1:]):
        ax.semilogy(energy, activities[mode], color=color, lw=1.35)
axes[0, 0].set(title="Complete 70-state polarizability", ylabel=r"Im Tr($\alpha$)/3 / $a_0^3$")
axes[0, 0].legend(frameon=False, fontsize=9)
names = ("Bend", "Symmetric stretch", "Antisymmetric stretch")
for mode, (ax, name, frequency) in enumerate(zip(axes.flat[1:], names, report["mode_frequencies_cm1"])):
    ax.set(title=f"{name} · {frequency:.1f} cm$^{{-1}}$",
           ylabel=r"Dynamic activity / $\mathrm{\AA}^4\,\mathrm{amu}^{-1}$")
for ax in axes[1]:
    ax.set_xlabel("Incident photon energy / eV")
for ax in axes.flat:
    ax.set_xlim(0, energy[-1]); ax.grid(alpha=.15)
fig.suptitle("Water · RHF/G₀W₀/full static BSE · Cartesian 6-31G*\n"
             "Relaxed mode derivatives; imposed optical width, harmonic Placzek model", fontsize=13)
fig.savefig(OUTPUT/"water_bse_dynamic_raman.png", dpi=210)
fig.savefig(OUTPUT/"water_bse_dynamic_raman.pdf")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
for mode, color, name in zip((1, 2, 3), colors, names):
    checks = [check for check in report["geometry_fd_checks"] if check["mode"] == mode]
    ax.loglog([check["step_bohr_sqrt_me"] for check in checks],
              [check["max_scaled_error"] for check in checks], "o-", color=color, label=name)
ax.axhline(1., color="0.4", ls="--", lw=1., label="Acceptance limit")
ax.set(xlabel=r"Normal-coordinate step / $a_0\sqrt{m_e}$", ylabel="Maximum scaled AD–FD error",
       title="Complete spectra: 3 widths × 1761 energies × 9 tensor entries")
ax.legend(frameon=False); ax.grid(alpha=.15, which="both")
fig.savefig(OUTPUT/"water_bse_mode_fd.png", dpi=210)
fig.savefig(OUTPUT/"water_bse_mode_fd.pdf")
plt.close(fig)

# Recorded outputs: water_bse_dynamic_raman.png/.pdf and water_bse_mode_fd.png/.pdf.
# Both PNGs were inspected after the native and independent checks passed.
