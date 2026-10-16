"""Independent RHF energy/gradient/Hessian and spin-broken UHF branch check."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, scf


output = Path("reproducibility/ad_showcase/G07_h2_surface")
data = json.loads((output / "gradscf.json").read_text())
rows = []
start = perf_counter()
for row in data["records"]:
    length = row["bond_bohr"]
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {length}", basis=data["basis"],
                unit="Bohr", cart=data["cart"], verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    mf.kernel()
    assert mf.converged
    first = float(mf.nuc_grad_method().kernel()[1, 2])
    second = float(mf.Hessian().kernel()[1, 1, 2, 2])
    np.testing.assert_allclose([mf.e_tot, first, second],
        [row["rhf_energy_hartree"], row["gradient_hartree_per_bohr"], row["curvature_hartree_per_bohr2"]],
        atol=2e-7, rtol=0)
    # Independent alpha/beta-opposite orbital rotation starts a UHF candidate.
    # It may return to RHF on the stable side; it does not reuse native states.
    c0, c1 = mf.mo_coeff[:, 0], mf.mo_coeff[:, 1]
    alpha = np.cos(.3)*c0+np.sin(.3)*c1
    beta = np.cos(.3)*c0-np.sin(.3)*c1
    unrestricted = scf.UHF(mol)
    unrestricted.conv_tol, unrestricted.conv_tol_grad, unrestricted.max_cycle = 1e-13, 1e-10, 300
    unrestricted.kernel(dm0=np.array([np.outer(alpha, alpha), np.outer(beta, beta)]))
    assert unrestricted.converged
    np.testing.assert_allclose(unrestricted.e_tot, row["selected_energy_hartree"], atol=2e-8, rtol=0)
    rows.append(dict(bond_bohr=length, rhf_energy_hartree=float(mf.e_tot),
        gradient_hartree_per_bohr=first, curvature_hartree_per_bohr2=second,
        selected_energy_hartree=float(unrestricted.e_tot), spin_square=float(unrestricted.spin_square()[0]),
        gradient_error=abs(first-row["gradient_hartree_per_bohr"]),
        curvature_error=abs(second-row["curvature_hartree_per_bohr2"])))
    print(f"R={length:.2f}: PySCF gradient error={rows[-1]['gradient_error']:.3e}, "
          f"curvature error={rows[-1]['curvature_error']:.3e}", flush=True)
(output / "pyscf.json").write_text(json.dumps(dict(pyscf_version=pyscf.__version__,
    records=rows, elapsed_seconds=perf_counter()-start), indent=2)+"\n")
