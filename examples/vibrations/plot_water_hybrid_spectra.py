"""Plot saved hybrid optimization, Hessian/isotope checks and IR/Raman data."""
import json
from pathlib import Path
import runpy

import numpy as np


XC = "pbe0"


def run(xc):
    if xc not in ("pbe0", "b3lyp", "wb97xd"):
        raise ValueError("Choose pbe0, b3lyp or wb97xd.")
    output = Path("reproducibility/ad_showcase")/f"M2_water_{xc}_spectra"
    examples = Path(__file__).resolve().parents[1]
    optimization = runpy.run_path(str(examples/"geometry/plot_water_pbe_optimization.py"))["run"]
    modes = runpy.run_path(str(examples/"geometry/plot_water_pbe_modes.py"))["run"]
    spectra = runpy.run_path(str(examples/"vibrations/plot_water_spectra.py"))["run"]
    optimization(output/"geometry.json", output=output)
    modes(output/"hessian.json", output/"pyscf_geometry.json", output=output)
    spectra(output)
    geometry = json.loads((output/"geometry.json").read_text())
    hessian = json.loads((output/"hessian.json").read_text())
    history = geometry["history"]
    np.savetxt(output/"optimization.csv", np.column_stack((
        np.arange(1, len(history)+1), [row["energy_hartree"] for row in history],
        [row["max_gradient"] for row in history])), delimiter=",",
        header="evaluation,energy_hartree,max_gradient_hartree_per_bohr", comments="")
    np.savetxt(output/"isotope_frequencies.csv", np.column_stack((
        np.arange(1, 4), hessian["isotopes"]["H2O"]["frequencies_cm1"],
        hessian["isotopes"]["D2O"]["frequencies_cm1"])), delimiter=",",
        header="mode,H2O_wavenumber_cm-1,D2O_wavenumber_cm-1", comments="")


if __name__ == "__main__":
    run(XC)

# Rendered on jy_gpu_1, 2026-10-15, for PBE0, B3LYP(VWN-RPA) and original WB97XD.
# Each case: optimization, Hessian/isotope and IR/Raman PNG/PDF pairs;
# 40-frame 1000x320 mode GIF; optimization, isotope, stick and broadened CSVs.
# The Gaussian width and animation amplitude are display choices.
