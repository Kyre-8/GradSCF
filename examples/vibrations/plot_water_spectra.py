"""Plot saved water/CO2 harmonic IR and static Raman cases; no SCF is run.

The 15 cm^-1 Gaussian FWHM is a display setting, not a lifetime. Raman
activities are not scattering intensities. Degenerate-mode areas add.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("reproducibility/ad_showcase")
CASES = (RESULTS / "V01_V02_water_rhf_spectra", RESULTS / "V01_V02_water_pbe_spectra",
         RESULTS / "V03_co2_selection_rules")


def run(case_path, output=None):
    case_path = Path(case_path)
    data = json.loads((case_path / "gradscf.json").read_text())
    output = case_path if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frequencies = np.asarray(data["frequencies_cm1"])
    if data["symbols"] == ["O", "H", "H"]:
        lower, upper = 1000., 4600.  # Retain the common HF/PBE water display range.
    else:
        lower = max(0., 100*np.floor((frequencies.min()-200)/100))
        upper = 100*np.ceil((frequencies.max()+200)/100)
    grid = np.linspace(lower, upper, 6000)
    fwhm = 15.
    sigma = fwhm / np.sqrt(8*np.log(2))
    profiles = np.exp(-.5*((grid[:, None]-frequencies)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
    groups = []
    for index in np.argsort(frequencies):
        if groups and abs(frequencies[index]-frequencies[groups[-1][0]]) < 1e-3:
            groups[-1].append(index)
        else:
            groups.append([index])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), layout="constrained", sharex=True)
    series = (("ir_intensities_km_mol", "IR intensity / (km/mol)", "#337a9b"),
              ("raman_activities_angstrom4_amu", "Raman activity / (Å⁴/amu)", "#a45d20"))
    for ax, (key, label, color) in zip(axes, series):
        sticks = np.asarray(data[key])
        ax.plot(grid, profiles@sticks, color=color, lw=1.8)
        ax.set(ylabel=label + " per cm⁻¹")
        for group in groups:
            strength = np.sum(sticks[group])
            if strength <= sticks.max()*1e-10:
                continue  # Do not annotate numerical leakage of forbidden peaks.
            height = strength / (sigma*np.sqrt(2*np.pi))
            ax.annotate(",".join(str(i+1) for i in group), (frequencies[group[0]], height),
                        xytext=(0, 7), textcoords="offset points", ha="center", color=color)
        ax.set_ylim(0, ax.get_ylim()[1]*1.18)
    axes[-1].set_xlabel("Harmonic wavenumber / cm⁻¹")
    molecule = "CO₂" if data["symbols"] == ["C", "O", "O"] else "Water"
    method = data.get("label", data.get("xc", data["method"]))
    fig.suptitle(f"{molecule} · {method} / Cartesian {data['basis']}\n"
                 "Harmonic IR and static Raman · Gaussian display FWHM 15 cm⁻¹")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"ir_raman.{suffix}", dpi=220)
    plt.close(fig)
    if "depolarization_ratios" in data:
        ratios = np.asarray(data["depolarization_ratios"])
    else:
        derivatives = np.asarray(data["alpha_derivatives"])
        average = np.trace(derivatives, axis1=1, axis2=2)/3
        gamma2 = 1.5*np.sum((derivatives-average[:,None,None]*np.eye(3))**2, axis=(1,2))
        denominator = 45*average**2+4*gamma2
        ratios = np.divide(3*gamma2, denominator, out=np.full_like(average, np.nan), where=denominator>0)
        if "inversion_parity" in data:
            ratios[np.asarray(data["inversion_parity"]) < -.99] = np.nan
        # The polarization ratio of a symmetry-forbidden Raman line is undefined.
    np.savetxt(output / "sticks.csv", np.column_stack((frequencies, data[series[0][0]],
        data[series[1][0]], ratios)), delimiter=",",
        header="wavenumber_cm-1,IR_km_mol,Raman_A4_amu,depolarization_ratio", comments="")
    np.savetxt(output / "broadened.csv", np.column_stack((grid, profiles@data[series[0][0]],
        profiles@data[series[1][0]])), delimiter=",",
        header="wavenumber_cm-1,IR_density,Raman_activity_density", comments="")


if __name__ == "__main__":
    for case_path in CASES:
        run(case_path)


# PBE figure and CSVs generated on jy_gpu_1, 2026-10-15:
# V01_V02_water_pbe_spectra/ir_raman.png, ir_raman.pdf, sticks.csv, broadened.csv.
# Unit-area Gaussian display FWHM 15 cm^-1; no frequency scaling.
