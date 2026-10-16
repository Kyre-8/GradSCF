"""Independent PySCF analytic-gradient check of the saved GradSCF water result.

Run water_rhf_gradient.py first. No GradSCF calculation is imported or replaced.
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, scf

output = Path("reproducibility/ad_showcase/G01_water_rhf_gradient")
data = json.loads((output / "gradscf.json").read_text())
mol = gto.M(
    atom=list(zip(data["symbols"], data["coordinates_bohr"])),
    basis=data["basis"], unit="Bohr", cart=data["cart"], verbose=0,
)
start = perf_counter()
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-10
mf.max_cycle = 150
mf.kernel()
if not mf.converged:
    raise RuntimeError("The independent PySCF reference did not converge.")
gradient = mf.nuc_grad_method().kernel()
elapsed = perf_counter() - start
error = np.max(np.abs(gradient - np.asarray(data["gradient_hartree_per_bohr"])))
np.testing.assert_allclose(mf.e_tot, data["energy_hartree"], atol=1e-8, rtol=0)
np.testing.assert_allclose(gradient, data["gradient_hartree_per_bohr"],
                           atol=1e-6, rtol=0)
reference = dict(
    energy_hartree=mf.e_tot, gradient_hartree_per_bohr=gradient.tolist(),
    max_gradient_error=float(error), elapsed_seconds=elapsed,
    pyscf_version=pyscf.__version__, conv_tol=mf.conv_tol,
    conv_tol_grad=mf.conv_tol_grad,
)
(output / "pyscf.json").write_text(json.dumps(reference, indent=2) + "\n")
print(f"PySCF E = {mf.e_tot:.12f} Eh; max gradient error = {error:.3e} Eh/Bohr")

# CPU float64, 2026-10-15:
# PySCF E = -76.010492818245 Eh; max gradient error = 4.521e-12 Eh/Bohr
