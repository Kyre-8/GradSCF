"""Plot V07's raw KHD strengths and full complex tensor; run after both checks."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("reproducibility/ad_showcase/V07_formaldehyde_khd")
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(figure, name):
    figure.savefig(ROOT/f"{name}.png", dpi=180, bbox_inches="tight")
    figure.savefig(ROOT/f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def main():
    data = json.loads((ROOT/"gradscf.json").read_text())
    reference = json.loads((ROOT/"resolvent_reference.json").read_text())
    assert data["complete"] and reference["passed"]
    with np.load(ROOT/"khd_tensors.npz") as raw:
        laser, gamma = raw["laser_ev"], raw["gamma_ev"]
        r, anti, total, intensity = [raw[key] for key in ("resonant", "antiresonant", "total", "parallel_perpendicular")]
    isotropic = intensity.sum(axis=-1)
    shifts = np.asarray(data["ground_fundamentals_cm1"])
    colors = plt.cm.tab10(np.arange(6))
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for mode, color in enumerate(colors):
        axes[0, 0].plot(laser, isotropic[1, :, mode], color=color, label=f"mode {mode+1}: {shifts[mode]:.0f} cm$^{{-1}}$")
    axes[0, 0].set(title="Ground fundamentals, Γ = 0.030 eV", xlabel="Incident photon energy / eV", ylabel="Parallel + perpendicular strength / a.u.²")
    axes[0, 0].legend(fontsize=8, ncol=2)
    for index, width in enumerate(gamma):
        axes[0, 1].plot(laser, isotropic[index].sum(axis=1), label=f"Γ = {width:.3f} eV")
    axes[0, 1].set(title="Linewidth sensitivity, sum of six final states", xlabel="Incident photon energy / eV", ylabel="Raw strength / a.u.²")
    axes[0, 1].legend(fontsize=9)
    chosen = int(np.argmax(isotropic[1].sum(axis=1)))
    for mode, color in enumerate(colors):
        axes[1, 0].vlines(shifts[mode]-8, 0, intensity[1, chosen, mode, 0], color=color, linewidth=2)
        axes[1, 0].vlines(shifts[mode]+8, 0, intensity[1, chosen, mode, 1], color=color, linewidth=2, linestyle="--")
    axes[1, 0].plot([], [], color="black", label="parallel")
    axes[1, 0].plot([], [], color="black", linestyle="--", label="perpendicular")
    axes[1, 0].set(title=f"Stokes lines at {laser[chosen]:.4f} eV; Γ = 0.030 eV", xlabel="Raman shift / cm$^{-1}$", ylabel="Raw strength / a.u.²")
    axes[1, 0].legend(fontsize=9)
    rows = data["convergence"]
    cutoff = [row["max_quanta"] for row in rows]
    axes[1, 1].semilogy(cutoff, [row["max_tail_bound_relative_to_global_peak"] for row in rows], "o-", label="uniform tensor tail bound / peak")
    axes[1, 1].semilogy(cutoff[1:], [row["maximum_tensor_change_relative_to_global_peak"] for row in rows[1:]], "s-", label="tensor change / peak")
    axes[1, 1].semilogy(cutoff, [max(row["emission_relative_remainders"]) for row in rows], "^-", label="largest emission closure remainder")
    axes[1, 1].axhline(1e-4, color=".5", linestyle=":", linewidth=1)
    axes[1, 1].set(title="Intermediate-state convergence", xlabel="Maximum total excited quanta", ylabel="Dimensionless error")
    axes[1, 1].legend(fontsize=8)
    figure.suptitle("HCHO one-state KHD model · 0 K · harmonic surfaces + linear HT\nRaw squared amplitudes; photon prefactors omitted", fontsize=13)
    for axis in axes.flat:
        axis.grid(alpha=.18)
    save(figure, "khd_excitation_profiles")

    g, l, mode = data["strongest_gamma_laser_mode_indices"]
    tensor = total[g, l, mode]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    scale = max(np.max(abs(tensor.real)), np.max(abs(tensor.imag)))
    for axis, values, title in zip(axes[:2], (tensor.real, tensor.imag), ("Real A", "Imaginary A")):
        plot = axis.imshow(values, cmap="RdBu_r", vmin=-scale, vmax=scale)
        axis.set(xticks=range(3), yticks=range(3), xticklabels=list("xyz"), yticklabels=list("xyz"), xlabel="Incident index j", ylabel="Scattered index i", title=title)
        for i in range(3):
            for j in range(3):
                axis.text(j, i, f"{values[i,j]:.3g}", ha="center", va="center", fontsize=8,
                          color="white" if abs(values[i,j]) > .55*scale else "black")
    figure.colorbar(plot, ax=axes[:2], shrink=.8, label="Amplitude / (e bohr)² Eh$^{-1}$")
    norm_r = np.sum(abs(r[g, :, mode])**2, axis=(-2, -1))
    norm_a = np.sum(abs(anti[g, :, mode])**2, axis=(-2, -1))
    cross = 2*np.real(np.sum(r[g, :, mode]*anti[g, :, mode].conj(), axis=(-2, -1)))
    axes[2].plot(laser, norm_a, label="Antiresonant |A|²")
    axes[2].plot(laser, cross, label="Signed time-order interference")
    axes[2].axhline(0, color=".5", linewidth=.7)
    axes[2].set(xlabel="Incident photon energy / eV", ylabel="Frobenius contribution / a.u.²", title=f"Mode {mode+1}, Γ = {gamma[g]:.3f} eV")
    axes[2].legend(fontsize=8)
    figure.suptitle(f"Full complex tensor at {laser[l]:.4f} eV · mode {mode+1} · no symmetry imposed", fontsize=12)
    save(figure, "khd_complex_tensor")


if __name__ == "__main__":
    main()
