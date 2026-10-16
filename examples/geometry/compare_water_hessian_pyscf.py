"""Independent analytic PySCF Hessian at the saved GradSCF RHF minimum."""
import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, scf

output = Path("reproducibility/ad_showcase/G05_water_rhf_hessian")
data = json.loads((output / "gradscf.json").read_text())
mol = gto.M(atom=list(zip(data["symbols"], data["coordinates_bohr"])),
            basis=data["basis"], cart=data["cart"], unit="Bohr", verbose=0)
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-10
mf.kernel()
assert mf.converged
gradient = mf.nuc_grad_method().kernel()
hessian = mf.Hessian().kernel().transpose(0, 2, 1, 3).reshape(3*mol.natm, -1)
error = float(np.max(np.abs(hessian - np.array(data["hessian_hartree_per_bohr2"]))))
np.testing.assert_allclose(hessian, data["hessian_hartree_per_bohr2"], atol=2e-6, rtol=0)
assert np.max(np.abs(gradient)) < 1e-7
reference = dict(pyscf_version=pyscf.__version__, energy_hartree=float(mf.e_tot),
                 gradient_hartree_per_bohr=gradient.tolist(),
                 hessian_hartree_per_bohr2=hessian.tolist(), max_hessian_error=error)
(output / "pyscf.json").write_text(json.dumps(reference, indent=2) + "\n")
print(f"Maximum AD-PySCF Hessian error: {error:.3e} Eh/Bohr^2")

# jy_gpu_1, CPU float64, 2026-10-15:
# Maximum AD-PySCF Hessian error: 2.699e-08 Eh/Bohr^2
