"""Independent PBE energy and full-grid analytic gradient at the AD minimum."""
import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import dft, gto
from pyscf.dft import gen_grid, radi

output = Path("reproducibility/ad_showcase/G04_water_pbe_optimization")
data = json.loads((output / "gradscf.json").read_text())
mol = gto.M(atom=list(zip(data["symbols"], data["coordinates_bohr"])),
            basis=data["basis"], unit="Bohr", cart=data["cart"], verbose=0)
mf = dft.RKS(mol)
mf.xc = "pbe"
mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 150
mf.small_rho_cutoff = 0
mf.grids.level, mf.grids.alignment = data["grid_level"], 1
mf.grids.radi_method, mf.grids.prune = radi.treutler_ahlrichs, gen_grid.nwchem_prune
mf.grids.radii_adjust, mf.grids.atomic_radii = radi.treutler_atomic_radii_adjust, radi.BRAGG_RADII
mf.grids.becke_scheme = gen_grid.original_becke
mf.kernel()
if not mf.converged:
    raise RuntimeError("Independent PySCF PBE SCF did not converge.")
derivative = mf.nuc_grad_method()
derivative.grid_response = True
gradient = derivative.kernel()
error = float(np.max(np.abs(gradient - np.asarray(data["gradient_hartree_per_bohr"]))))
np.testing.assert_allclose(mf.e_tot, data["energy_hartree"], atol=1e-8, rtol=0)
np.testing.assert_allclose(gradient, data["gradient_hartree_per_bohr"], atol=1e-7, rtol=0)
assert np.max(np.abs(gradient)) < 1e-7
reference = dict(pyscf_version=pyscf.__version__, energy_hartree=mf.e_tot,
                 gradient_hartree_per_bohr=gradient.tolist(), max_gradient_error=error,
                 grid_level=data["grid_level"], grid_points=len(mf.grids.weights),
                 grid_response=True, conv_tol=1e-13, conv_tol_grad=1e-11)
(output / "pyscf.json").write_text(json.dumps(reference, indent=2) + "\n")
print(f"PySCF E={mf.e_tot:.12f} Eh; max gradient={np.max(np.abs(gradient)):.3e} Eh/Bohr")
print(f"Maximum AD-PySCF gradient difference: {error:.3e} Eh/Bohr")

# Measured on jy_gpu_1 (CPU, float64, 2026-10-15):
# PySCF E=-76.322499095449 Eh; max gradient=4.778e-08 Eh/Bohr
# Maximum AD-PySCF gradient difference: 5.121e-13 Eh/Bohr
