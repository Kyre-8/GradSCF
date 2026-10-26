"""Plot raw FC/HT/interference components of the validated X12 linear model."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("reproducibility/ad_showcase/X12_formaldehyde_herzberg_teller")
data = json.loads((OUTPUT/"gradscf.json").read_text())
assert data["complete"]
curves = np.loadtxt(OUTPUT/"spectrum.csv", delimiter=",", skiprows=1)
energy = curves[:, 0]
final = data["convergence"][-1]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.formatter.use_mathtext": True, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4), layout="constrained")
axes[0, 0].plot(energy, curves[:, 4], color=".12", lw=1.7, label="Total, σ = 0.02 eV")
axes[0, 0].plot(energy, curves[:, 2], "--", color="#D55E00", lw=1.2, label="HT, σ = 0.02 eV")
axes[0, 0].plot(energy, curves[:, 1], ":", color="#0072B2", label="FC-only, raw")
axes[0, 0].plot(energy, curves[:, 8], color=".55", lw=.8, alpha=.75, label="Total, σ = 0.01 eV")
axes[0, 0].set(title="HT activation: total is nonnegative", xlabel="Photon energy / eV",
               ylabel="Oscillator-strength density / eV⁻¹")
axes[0, 0].legend(frameon=False, fontsize=7.5)
axes[0, 1].plot(energy, curves[:, 1], color="#0072B2")
axes[0, 1].set(title="FC-only numerical symmetry residual", xlabel="Photon energy / eV",
               ylabel="Raw FC oscillator-strength density / eV⁻¹")
axes[0, 1].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
axes[0, 1].text(.96, .92, f"Σ f_FC = {final['raw_oscillator_strength_sums']['fc']:.3e}",
                transform=axes[0, 1].transAxes, ha="right", va="top", fontsize=8)
interference = curves[:, 3]
axes[0, 2].plot(energy, interference, color=".25", lw=.8)
axes[0, 2].fill_between(energy, 0, interference, where=interference >= 0, color="#D55E00", alpha=.65, label="Positive")
axes[0, 2].fill_between(energy, 0, interference, where=interference < 0, color="#0072B2", alpha=.65, label="Negative")
axes[0, 2].axhline(0, color=".3", lw=.8)
axes[0, 2].set(title="Signed interference term, not absorption", xlabel="Photon energy / eV",
               ylabel="Interference contribution / eV⁻¹")
axes[0, 2].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
axes[0, 2].legend(frameon=False, fontsize=8)

contributions = np.asarray(data["mode_HT_closure_contributions"])
axes[1, 0].bar(np.arange(1, 7), contributions, color="#0072B2")
axes[1, 0].set(title="Initial-mode contributions to HT closure", xlabel="Ground normal mode index",
               ylabel="Integrated HT dipole strength / (e bohr)²")
axes[1, 0].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
axes[1, 0].text(.96, .92, "Σₓ |∂μₓ/∂Qₖ|² / (2ωg,k)\nIncludes zero-point coordinate variance",
                transform=axes[1, 0].transAxes, ha="right", va="top", fontsize=8)
rows = data["convergence"]
cutoffs = [row["max_quanta"] for row in rows]
axes[1, 1].semilogy(cutoffs, [max(abs(row["relative_closure_remainder"]), 1e-16) for row in rows],
                    "o-", color="#0072B2", label="Relative dipole-strength remainder")
axes[1, 1].semilogy(cutoffs[1:], [max(row["relative_total_f_change"], 1e-16) for row in rows[1:]],
                    "s-", color="#009E73", label="Change in integrated f")
axes[1, 1].semilogy(cutoffs[1:], [max(max(row["relative_total_profile_l1_changes"]), 1e-16) for row in rows[1:]],
                    "x--", color="#D55E00", label="Profile change, larger of two σ")
axes[1, 1].axhline(2e-8, color=".5", ls=":", label="Closure gate")
axes[1, 1].set(title="Closure and raw-tail convergence", xlabel="Maximum total vibrational quanta",
               ylabel="Relative remainder or change")
axes[1, 1].legend(frameon=True, framealpha=1, facecolor="white", fontsize=7)
lines = data["dominant_lines"][:20]
zero_zero_ev = data["harmonic_model"]["model_zero_zero_hartree"]*27.211386245988
x = np.array([line["photon_energy_ev"]-zero_zero_ev for line in lines])
y = np.array([line["oscillator_strengths"]["total"] for line in lines])
axes[1, 2].vlines(x, 0, y, color="#0072B2", lw=1.5)
axes[1, 2].plot(x, y, "o", color="#0072B2", ms=3)
axes[1, 2].set(title="Largest activated lines: full Cartesian sum", xlabel="Energy above model 0–0 / eV",
               ylabel="Total oscillator strength f", ylim=(0, max(y)*1.3))
axes[1, 2].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
for index in range(3):
    label = ",".join(map(str, lines[index]["occupations"]))
    axes[1, 2].annotate(f"({label})", (x[index], y[index]), xytext=(0, 6+index*5),
                        textcoords="offset points", ha="center", fontsize=7)
fig.suptitle("Formaldehyde S1 · frozen X11 common-slice surfaces + X07 full electronic dipole derivatives\n"
             "Zero-temperature, one-local-well double-harmonic/linear-HT model · no component normalization", fontsize=12)
for suffix in ("png", "pdf"):
    fig.savefig(OUTPUT/f"herzberg_teller_activation.{suffix}", dpi=220)
plt.close(fig)
(OUTPUT/"plot_settings.json").write_text(json.dumps(dict(
    components="FC, HT, signed interference and total use their original raw oscillator strengths; all three Cartesian moment components are summed.",
    near_dark="The dedicated FC-only and interference panels keep their own labeled raw scales. Their numerical-residual ratios are not interpreted as physical enhancement factors.",
    spectrum="Gaussian sigma=.02/.01 eV; no fitted lifetime or experimental broadening.",
    line_selection="Twenty largest total oscillator strengths in the stick panel; the curves use all retained states except the reported relative <=1e-12 L1 plot screen.",
    log_display_floor=1e-16, scope=data["scope"]), indent=2)+"\n")
print("Saved X12 raw FC/HT/interference, closure and activated-line figure")
