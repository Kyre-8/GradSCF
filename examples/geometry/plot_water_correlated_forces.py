"""G02: matched-basis force differences and independent derivative errors."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


output = Path("reproducibility/ad_showcase/G02_water_correlated_forces")
data = json.loads((output / "gradscf.json").read_text())
reference = json.loads((output / "pyscf.json").read_text())
dft_data = json.loads((output / "dft_gradscf.json").read_text())
dft_reference = json.loads((output / "dft_pyscf.json").read_text())
np.testing.assert_allclose(data["coordinates_bohr"], dft_data["coordinates_bohr"], atol=0, rtol=0)
records = data["records"] + dft_data["records"]
oracles = {(row["basis"], row["method"]): row
           for row in reference["records"] + dft_reference["records"]}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), layout="constrained")
colors = ("#337a9b", "#b16529", "#608b55", "#8b658e", "#89752e", "#b35f76", "#487c7a")
order = ("hf", "pbe", "pbe0", "b3lyp", "mp2", "ccsd", "fci")
colors = dict(zip(order, colors))
for column, basis in enumerate(("sto-3g", "6-31g*")):
    rows = sorted((row for row in records if row["basis"] == basis), key=lambda r: order.index(r["method"]))
    hf_gradient = np.array(rows[0]["gradient_hartree_per_bohr"])
    width = .8 / len(rows)
    for index, row in enumerate(rows):
        # The force has the opposite sign to the tabulated energy gradient.
        change = hf_gradient - np.array(row["gradient_hartree_per_bohr"])
        axes[0, column].bar(np.arange(3)+(index-(len(rows)-1)/2)*width,
                            np.linalg.norm(change, axis=1)*1e3, width=.9*width,
                            label=row["method"].upper(), color=colors[row["method"]])
    axes[0, column].set(xticks=range(3), xticklabels=["O", "H₁", "H₂"],
                         ylabel="|F(method) − F(HF)| / mEh Bohr⁻¹",
                         title=f"Water · Cartesian {basis.upper()}")
    axes[0, column].legend(frameon=False, ncols=2, fontsize=8)
    names = [row["method"].upper() for row in rows]
    axes[1, column].semilogy(names, [oracles[(basis, row["method"])]["max_gradient_error"]
                                    for row in rows], "o-", label="Independent analytic gradient")
    axes[1, column].semilogy(names, [row["directional_checks"][-1]["error"] for row in rows],
                              "s--", label="Energy FD, h = 10⁻⁴ Bohr")
    axes[1, column].set(ylabel="Derivative error / Eh Bohr⁻¹")
axes[1, 0].legend(frameon=False, fontsize=8)
fig.suptitle("Fully relaxed molecular forces across electronic-structure levels")
for suffix in ("png", "pdf"):
    fig.savefig(output / f"correlated_forces.{suffix}", dpi=220)
plt.close(fig)
