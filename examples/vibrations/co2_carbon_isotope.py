"""V04: 12C/13C isotope shifts from the same electronic CO2 force constants.

No electronic calculation is repeated. For symmetric linear O-C-O, the
gerade stretch leaves carbon stationary; ungerade frequencies scale with
sqrt(1/M_O+2/M_C). This provides an independent mass-metric check.
"""
import json
from pathlib import Path

import jax
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from gradscf_tools.vibrations import normal_modes


source = Path("reproducibility/ad_showcase/V03_co2_selection_rules/gradscf.json")
data = json.loads(source.read_text())
oxygen_mass = 15.99491461957
carbon_masses = (12., 13.00335483507)
frequencies = []
for carbon in carbon_masses:
    modes = normal_modes(data["hessian_hartree_per_bohr2"], data["coordinates_bohr"],
                          [carbon, oxygen_mass, oxygen_mass])
    assert modes.rigid_rank == 5 and np.all(modes.frequencies_cm1 > 0)
    frequencies.append(modes.frequencies_cm1)
frequencies = np.array(frequencies)
parity = np.array(data["inversion_parity"])
expected_ratio = np.sqrt((1/oxygen_mass + 2/carbon_masses[1])
                         / (1/oxygen_mass + 2/carbon_masses[0]))
np.testing.assert_allclose(frequencies[1, parity > .99], frequencies[0, parity > .99],
                           atol=1e-3, rtol=0)
np.testing.assert_allclose(frequencies[1, parity < -.99],
                           frequencies[0, parity < -.99]*expected_ratio, atol=1e-3, rtol=0)
output = Path("reproducibility/ad_showcase/V04_co2_carbon_isotope")
output.mkdir(parents=True, exist_ok=True)
result = dict(carbon_masses_amu=carbon_masses, oxygen_mass_amu=oxygen_mass,
              frequencies_cm1=frequencies.tolist(), ungerade_expected_ratio=float(expected_ratio),
              source_hessian=str(source), approximation="fixed electronic force constants; isotope masses only")
(output / "gradscf.json").write_text(json.dumps(result, indent=2)+"\n")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
for index, (values, label, color) in enumerate(zip(frequencies, ("12C16O2", "13C16O2"),
                                                ("#337a9b", "#a45d20"))):
    ax.plot(np.arange(4)+(index-.5)*.12, values, "o", label=label, color=color)
ax.set(xticks=range(4), xticklabels=["Bend 1", "Bend 2", "Symmetric stretch", "Antisymmetric stretch"],
       ylabel="Harmonic wavenumber / cm⁻¹", title="CO₂ carbon-isotope shifts · fixed RHF / 6-31G* Hessian")
ax.legend(frameon=False)
for suffix in ("png", "pdf"):
    fig.savefig(output / f"carbon_isotope.{suffix}", dpi=220)
plt.close(fig)
print("12C/13C frequencies / cm^-1:\n", frequencies)
print("Ungerade mass-scaling ratio:", expected_ratio)

# Measured on jy_gpu_1, CPU float64, 2026-10-15:
# 12C16O2: [745.80627644, 745.80627646, 1518.49578170, 2585.03147902] cm^-1
# 13C16O2: [724.57972273, 724.57972275, 1518.49578170, 2511.45833911] cm^-1
# Ungerade isotope ratio: 0.9715387837589528; gerade stretch unchanged.
