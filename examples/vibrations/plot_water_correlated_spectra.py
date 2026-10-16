"""Reuse the saved-data optimization, mode-animation and spectrum plots."""
import json
from pathlib import Path
import runpy

import numpy as np


METHOD = "ccsd"


def run(method):
    if method not in ("mp2", "ccsd"):
        raise ValueError("Choose mp2 or ccsd.")
    output = Path("reproducibility/ad_showcase")/f"V05_water_{method}_spectra"
    examples = Path(__file__).resolve().parents[1]
    optimization = runpy.run_path(str(examples/"geometry/plot_water_pbe_optimization.py"))["run"]
    modes = runpy.run_path(str(examples/"geometry/plot_water_pbe_modes.py"))["run"]
    spectrum = runpy.run_path(str(examples/"vibrations/plot_water_spectra.py"))["run"]
    optimization(output/"geometry.json", output=output)
    modes(output/"hessian.json", output/"pyscf_geometry.json", output=output)
    spectrum(output)
    data = json.loads((output/"hessian.json").read_text())
    np.savetxt(output/"isotope_frequencies.csv", np.column_stack((
        np.arange(1, 4), data["isotopes"]["H2O"]["frequencies_cm1"],
        data["isotopes"]["D2O"]["frequencies_cm1"])), delimiter=",",
        header="mode,H2O_wavenumber_cm-1,D2O_wavenumber_cm-1", comments="")


if __name__ == "__main__":
    run(METHOD)
