"""Plot saved theoretical formaldehyde sticks and independent software errors.

Gaussian sigma=0.10 eV is a display choice, not a calculated lifetime.
Curves retain the oscillator-strength sum of the six included singlets and
are not normalized separately. No electronic-structure calculation is run.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

output = Path("reproducibility/ad_showcase/X01_formaldehyde_absorption")
native = json.loads((output / "gradscf.json").read_text())
reference = json.loads((output / "pyscf.json").read_text())
if not native["complete"] or not reference["passed"]:
    raise RuntimeError("Only complete, independently checked spectra are plotted.")
if reference["native_sha256"] != hashlib.sha256((output / "gradscf.json").read_bytes()).hexdigest():
    raise RuntimeError("The reference comparison belongs to a different native result file.")
sigma_ev = 0.10
colors = dict(zip(native["functional_names"], ("#30343b", "#0072B2", "#D55E00", "#009E73", "#CC79A7")))
styles = dict(zip(native["functional_names"], ("-", "--", "-.", ":", (0, (4, 1, 1, 1)))))
labels = {"hf": "HF", "pbe": "PBE", "pbe0": "PBE0", "b3lyp": "B3LYP", "wb97xd": r"$\omega$B97X-D"}
maximum = max(state["excitation_ev"] for case in native["cases"] for state in case["states"])+0.6
energy_grid = np.linspace(3., maximum, 2400)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(3, 2, figsize=(11, 9.2), layout="constrained")
saved_columns = [energy_grid]
saved_names = ["energy_ev"]
for column, response in enumerate(("TDA", "TDDFT")):
    sticks, ax = axes[0, column], axes[1, column]
    for functional in native["functional_names"]:
        case = next(case for case in native["cases"] if case["functional"] == functional and case["response"] == response)
        energies = np.array([state["excitation_ev"] for state in case["states"]])
        strengths = np.array([state["oscillator_strength"] for state in case["states"]])
        spectrum = np.sum(strengths*np.exp(-0.5*((energy_grid[:, None]-energies)/sigma_ev)**2)
                          /(np.sqrt(2*np.pi)*sigma_ev), axis=1)
        ax.plot(energy_grid, spectrum, color=colors[functional], linestyle=styles[functional],
                label=labels[functional], linewidth=1.5)
        sticks.vlines(energies, 0., strengths, colors=colors[functional], alpha=.8, linewidth=1.5,
                      linestyles=styles[functional], label=labels[functional])
        saved_columns.append(spectrum)
        saved_names.append(f"{functional}_{response}_strength_per_ev")
    ax.set(xlabel="Excitation energy / eV", ylabel="Broadened oscillator strength / eV$^{-1}$",
           title="Gaussian broadening: σ = 0.10 eV", xlim=(3., maximum), ylim=(0., None))
    sticks.set(xlabel="Excitation energy / eV", ylabel="Oscillator strength f",
               title=f"{response} · first {native['nstates']} singlets", xlim=(3., maximum), ylim=(0., None))
    sticks.legend(frameon=False, ncol=3, loc="upper left")

    ax = axes[2, column]
    for functional in native["functional_names"]:
        case = next(case for case in native["cases"] if case["functional"] == functional and case["response"] == response)
        other = next(case for case in reference["cases"] if case["functional"] == functional and case["response"] == response)
        errors = np.abs(np.array([state["excitation_hartree"] for state in case["states"]])
                        - np.array([state["excitation_hartree"] for state in other["states"]]))
        ax.semilogy(np.arange(1, native["nstates"]+1), np.maximum(errors, 1e-15), "o",
                    color=colors[functional], linestyle=styles[functional], markersize=4, linewidth=1)
    ax.axhline(reference["tolerances"]["excitation_hartree"], linestyle="--", color="0.45", linewidth=1)
    ax.set(xlabel="Root index within each method", ylabel="|GradSCF − PySCF/SciPy| / Hartree",
           title="Independent excitation-energy comparison", xticks=np.arange(1, native["nstates"]+1))
for row in range(3):
    lower = min(ax.get_ylim()[0] for ax in axes[row])
    upper = max(ax.get_ylim()[1] for ax in axes[row])
    for ax in axes[row]:
        ax.set_ylim(lower, upper)
fig.suptitle("Formaldehyde · fixed C₂v geometry · Cartesian 6-31+G*\n"
             "Native full integrals · matched level-3 DFT grids · no experimental overlay", fontsize=12)
for suffix in ("png", "pdf"):
    fig.savefig(output / f"absorption_comparison.{suffix}", dpi=220)
plt.close(fig)
np.savetxt(output / "broadened.csv", np.column_stack(saved_columns), delimiter=",",
           header=",".join(saved_names), comments="")
(output / "plot_settings.json").write_text(json.dumps(dict(
    lineshape="normalized Gaussian", sigma_ev=sigma_ev, fwhm_ev=float(2*np.sqrt(2*np.log(2))*sigma_ev),
    energy_grid_points=len(energy_grid), energy_range_ev=[float(energy_grid[0]), float(energy_grid[-1])],
    normalization="No per-method normalization; six-state oscillator strengths retained.",
    scope="Theoretical vertical spectra at one basis and grid; width is a plotting choice."), indent=2)+"\n")
print("Saved absorption_comparison.png/.pdf and broadened.csv")

# Measured on jy_gpu_1, 2026-10-15:
# Saved absorption_comparison.png/.pdf and broadened.csv
# Five methods include original omegaB97X-D; original four-method data archived.
# Gaussian sigma=0.10 eV; 2400 grid points; no per-method normalization.
