"""Independent PBE analytic gradients, including molecular-grid response.

Run water_pbe_gradient.py first. PySCF builds its own AO integrals and grids;
no GradSCF calculation is imported. A level-5 comparison estimates the grid
error separately from the matched level-3 AD comparison.
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import dft, gto
from pyscf.dft import gen_grid, radi

output = Path("reproducibility/ad_showcase/G01_water_pbe_gradient")
data = json.loads((output / "gradscf.json").read_text())
mol = gto.M(
    atom=list(zip(data["symbols"], data["coordinates_bohr"])),
    basis=data["basis"], unit="Bohr", cart=data["cart"], verbose=0,
)
references = []
for level in (data["grid_level"], 5):
    start = perf_counter()
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 150
    mf.small_rho_cutoff = 0
    mf.grids.level = level
    mf.grids.alignment = 1  # Count physical quadrature points without padding.
    mf.grids.radi_method = radi.treutler_ahlrichs
    mf.grids.prune = gen_grid.nwchem_prune
    mf.grids.radii_adjust = radi.treutler_atomic_radii_adjust
    mf.grids.atomic_radii = radi.BRAGG_RADII
    mf.grids.becke_scheme = gen_grid.original_becke
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"Independent PySCF level-{level} SCF did not converge.")
    derivative = mf.nuc_grad_method()
    derivative.grid_response = True
    gradient = derivative.kernel()
    elapsed = perf_counter() - start
    references.append(dict(
        grid_level=level, grid_points=len(mf.grids.weights),
        energy_hartree=mf.e_tot, gradient_hartree_per_bohr=gradient.tolist(),
        elapsed_seconds=elapsed, grid_response=True,
    ))

reference, fine = references
error = np.max(np.abs(np.asarray(reference["gradient_hartree_per_bohr"])
                      - np.asarray(data["gradient_hartree_per_bohr"])))
grid_error = np.max(np.abs(np.asarray(fine["gradient_hartree_per_bohr"])
                           - np.asarray(reference["gradient_hartree_per_bohr"])))
np.testing.assert_allclose(reference["energy_hartree"], data["energy_hartree"],
                           atol=1e-8, rtol=0)
np.testing.assert_allclose(reference["gradient_hartree_per_bohr"],
                           data["gradient_hartree_per_bohr"], atol=1e-6, rtol=0)
reference.update(max_gradient_error=float(error), pyscf_version=pyscf.__version__,
                 conv_tol=1e-12, conv_tol_grad=1e-10, finer_grid=fine,
                 max_grid_level_3_to_5_gradient_change=float(grid_error))
(output / "pyscf.json").write_text(json.dumps(reference, indent=2) + "\n")
print(f"PySCF E = {reference['energy_hartree']:.12f} Eh")
print(f"max AD-PySCF gradient error = {error:.3e} Eh/Bohr")
print(f"max level-3 to level-5 gradient change = {grid_error:.3e} Eh/Bohr")

# Measured on jy_gpu_1 (CPU, float64, 2026-10-15):
# PySCF E = -76.321861744207 Eh
# max AD-PySCF gradient error = 4.833e-12 Eh/Bohr
# max level-3 to level-5 gradient change = 1.652e-06 Eh/Bohr
# The grid-level difference is a quadrature effect, not an AD error.
