"""Independent PySCF analytic RHF/MP2/CCSD/full-CASCI nuclear gradients."""
import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, scf, mp, cc, mcscf

output = Path("reproducibility/ad_showcase/G02_water_correlated_forces")
data = json.loads((output / "gradscf.json").read_text())
records = []
for row in data["records"]:
    mol = gto.M(atom=list(zip(data["symbols"], data["coordinates_bohr"])),
                basis=row["basis"], unit="Bohr", cart=data["cart"], verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    assert mf.converged
    if row["method"] == "hf":
        model = mf
    elif row["method"] == "mp2":
        model = mp.MP2(mf).run()
    elif row["method"] == "ccsd":
        model = cc.CCSD(mf)
        model.conv_tol, model.conv_tol_normt = 1e-12, 1e-10
        model.kernel()
        assert model.converged
    else:
        # All orbitals/electrons active: this CASCI is exactly the molecular FCI.
        model = mcscf.CASCI(mf, mol.nao, mol.nelectron)
        model.fcisolver.conv_tol = 1e-12
        model.kernel()
        assert model.fcisolver.converged
    gradient = model.nuc_grad_method().kernel()
    error = float(np.max(np.abs(gradient-row["gradient_hartree_per_bohr"])))
    np.testing.assert_allclose(model.e_tot, row["energy_hartree"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(gradient, row["gradient_hartree_per_bohr"], atol=2e-6, rtol=0)
    records.append(dict(basis=row["basis"], method=row["method"], energy_hartree=float(model.e_tot),
                         gradient_hartree_per_bohr=gradient.tolist(), max_gradient_error=error))
    print(f"{row['basis']} {row['method']}: max PySCF gradient error {error:.3e} Eh/Bohr")
(output / "pyscf.json").write_text(json.dumps(dict(pyscf_version=pyscf.__version__,
    records=records), indent=2) + "\n")
