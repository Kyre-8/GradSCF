"""Independent HF/PBE PySCF finite fields; choose saved cases at the top."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import dft, gto, scf
from pyscf.dft import gen_grid, radi

RESULTS = Path("reproducibility/ad_showcase")
CASES = (RESULTS / "R01_R02_water_electric", RESULTS / "R01_R02_water_pbe_electric")


def run(case_path, output=None):
    """Compare one saved method/geometry without importing a GradSCF solver."""
    case_path = Path(case_path)
    data = json.loads((case_path / "gradscf.json").read_text())
    xc = data.get("xc", "HF").upper()
    if xc not in ("HF", "PBE"):
        raise ValueError("This independent reference supports HF and PBE.")
    mol = gto.M(atom=list(zip(data["symbols"], data["coordinates_bohr"])),
                basis=data["basis"], unit="Bohr", cart=data["cart"], verbose=0)
    position = mol.intor_symmetric("int1e_r", comp=3)
    hcore = scf.RHF(mol).get_hcore()
    nuclear_dipole = mol.atom_charges() @ mol.atom_coords()

    def dipole(field):
        mf = scf.RHF(mol) if xc == "HF" else dft.RKS(mol)
        mf.conv_tol, mf.conv_tol_grad = (1e-12, 1e-10) if xc == "HF" else (1e-13, 1e-11)
        mf.max_cycle = 150
        if xc == "PBE":
            mf.xc, mf.small_rho_cutoff = "pbe", 0
            mf.grids.level, mf.grids.alignment = data["grid_level"], 1
            mf.grids.radi_method, mf.grids.prune = radi.treutler_ahlrichs, gen_grid.nwchem_prune
            mf.grids.radii_adjust, mf.grids.atomic_radii = radi.treutler_atomic_radii_adjust, radi.BRAGG_RADII
            mf.grids.becke_scheme = gen_grid.original_becke
        mf.get_hcore = lambda *args: hcore + np.einsum("x,xpq->pq", field, position)
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("An independent finite-field SCF did not converge.")
        return nuclear_dipole - np.einsum("pq,xqp->x", mf.make_rdm1(), position)

    start = perf_counter()
    mu = dipole(np.zeros(3))
    np.testing.assert_allclose(mu, data["dipole_e_bohr"], atol=1e-7, rtol=0)
    checks = []
    for step in (3e-4, 1e-4):
        alpha = np.column_stack([(dipole(step*axis)-dipole(-step*axis))/(2*step)
                                 for axis in np.eye(3)])
        error = float(np.max(np.abs(alpha-data["polarizability_bohr3"])))
        checks.append(dict(field_step_au=step, polarizability_bohr3=alpha.tolist(), max_alpha_error=error))
        print(f"{xc}: PySCF field step={step:.1e} au; alpha error={error:.3e} Bohr^3", flush=True)
    np.testing.assert_allclose(alpha, data["polarizability_bohr3"], atol=2e-5, rtol=0)
    output = case_path if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reference = dict(pyscf_version=pyscf.__version__, xc=xc, grid_level=data.get("grid_level"),
                     dipole_e_bohr=mu.tolist(), finite_field_checks=checks, elapsed_seconds=perf_counter()-start)
    (output / "pyscf.json").write_text(json.dumps(reference, indent=2) + "\n")
    return reference


if __name__ == "__main__":
    for case_path in CASES:
        run(case_path)

# RHF measured on jy_gpu_1, CPU float64, 2026-10-15:
# PySCF field step=3.0e-04 au; alpha error=3.005e-06 Bohr^3
# PySCF field step=1.0e-04 au; alpha error=3.499e-07 Bohr^3

# PBE measured on jy_gpu_1, CPU float64, 2026-10-15:
# PySCF field step=3.0e-04 au; alpha error=3.121e-06 Bohr^3
# PySCF field step=1.0e-04 au; alpha error=3.530e-07 Bohr^3
