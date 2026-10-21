"""Plot the saved DABNA-2 TDDFT oscillator strengths without fitted broadening."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


def plot_spectrum(spectrum_path, output_path):
    """Draw a stick spectrum from a completed ``dft_response.py`` calculation.

    Excitation energies are in eV; oscillator strengths are dimensionless.
    The sticks do not model a physical linewidth or vibrational structure.
    """
    spectrum_path, output_path = Path(spectrum_path), Path(output_path)
    spectrum = json.loads(spectrum_path.read_text())
    energies = np.asarray(spectrum["excitation_energy_ev"], dtype=float)
    strengths = np.asarray(spectrum["oscillator_strength"], dtype=float)
    if energies.ndim != 1 or strengths.shape != energies.shape or not energies.size:
        raise ValueError("Expected matching, nonempty excitation-energy and strength arrays.")
    if not np.isfinite(energies).all() or not np.isfinite(strengths).all():
        raise ValueError("The saved spectrum contains non-finite values.")

    fig, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    axis.vlines(energies, 0., strengths, color="#245c91", linewidth=1.5)
    axis.scatter(energies, strengths, color="#245c91", s=14, zorder=3)
    axis.set(xlabel="Excitation energy (eV)", ylabel="Oscillator strength",
             title=f"DABNA-2: {spectrum['xc'].upper()} / Cartesian def2-SVP",
             xlim=(max(0., energies.min()-.2), energies.max()+.2),
             ylim=(0., None))
    axis.text(.98, .95, f"Level-3 grid; {energies.size} singlet states\nFull TDDFT; complete orbital space",
              transform=axis.transAxes, ha="right", va="top", fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    output = here.parents[2] / "artifacts" / "dabna2" / "monomer_pbe_level3"
    print(plot_spectrum(output / "spectrum.json", output / "spectrum.png"))
