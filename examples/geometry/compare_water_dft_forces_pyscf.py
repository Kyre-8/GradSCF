"""Independent full-grid PySCF analytic PBE/PBE0/B3LYP forces for G02."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import dft, gto
from pyscf.dft import gen_grid, radi

output = Path("reproducibility/ad_showcase/G02_water_correlated_forces")
data = json.loads((output / "dft_gradscf.json").read_text())
if not data["complete"]:
    raise RuntimeError("The requested GradSCF DFT method collection is incomplete.")
records = []
for row in data["records"]:
    mol = gto.M(atom=list(zip(data["symbols"], data["coordinates_bohr"])),
                basis=row["basis"], unit="Bohr", cart=data["cart"], verbose=0)
    start = perf_counter()
    mf = dft.RKS(mol)
    # B3LYPG fixes the Gaussian/VWN-RPA convention independently of a site's
    # configurable B3LYP_WITH_VWN5 default. GradSCF uses this same definition.
    mf.xc = "b3lypg" if row["method"] == "b3lyp" else row["method"]
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 150
    mf.small_rho_cutoff = 0
    mf.grids.level, mf.grids.alignment = data["grid_level"], 1
    mf.grids.radi_method, mf.grids.prune = radi.treutler_ahlrichs, gen_grid.nwchem_prune
    mf.grids.radii_adjust, mf.grids.atomic_radii = radi.treutler_atomic_radii_adjust, radi.BRAGG_RADII
    mf.grids.becke_scheme = gen_grid.original_becke
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"Independent {row['method']} SCF did not converge.")
    grad = mf.nuc_grad_method()
    grad.grid_response = True
    derivative = grad.kernel()
    error = float(np.max(np.abs(derivative-row["gradient_hartree_per_bohr"])))
    np.testing.assert_allclose(mf.e_tot, row["energy_hartree"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(derivative, row["gradient_hartree_per_bohr"], atol=1e-6, rtol=0)
    records.append(dict(basis=row["basis"], method=row["method"], pyscf_xc=mf.xc,
                         energy_hartree=float(mf.e_tot), gradient_hartree_per_bohr=derivative.tolist(),
                         max_gradient_error=error, grid_level=mf.grids.level,
                         grid_points=len(mf.grids.weights), grid_response=True,
                         elapsed_seconds=perf_counter()-start))
    print(f"{row['method']}: PySCF E={mf.e_tot:.12f} Eh; gradient error={error:.3e} Eh/Bohr", flush=True)
(output / "dft_pyscf.json").write_text(json.dumps(dict(
    pyscf_version=pyscf.__version__, complete=True, records=records,
    conv_tol=1e-12, conv_tol_grad=1e-10), indent=2) + "\n")


# Measured with PySCF 2.13.0 on jy_gpu_1, CPU float64, 2026-10-15:
# Full-grid analytic gradient maximum difference from GradSCF / Hartree/Bohr:
# PBE 4.818e-12; PBE0 4.739e-12; B3LYP (VWN-RPA) 3.868e-12.
# Each method used 33698 grid points, Cartesian 6-31G*, and grid_response=True.
