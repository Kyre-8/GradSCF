"""Independent PySCF finite-field polynomial oracle for axial nonlinear response.

Fit dipoles as a polynomial in F/Fmax to avoid an ill-conditioned Vandermonde
matrix. Vary the field window, retain all fits, and report finite-field error.
"""
import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, scf

output = Path("reproducibility/ad_showcase/R03_hf_hyperpolarizability")
data = json.loads((output / "gradscf.json").read_text())
mol = gto.M(atom=list(zip(data["symbols"], data["coordinates_bohr"])),
            basis=data["basis"], cart=True, unit="Bohr", verbose=0)
z = mol.intor_symmetric("int1e_r", comp=3)[2]
hcore = scf.RHF(mol).get_hcore()
nuclear_z = mol.atom_charges() @ mol.atom_coords()[:, 2]
fits = []
for window in (1e-2, 5e-3, 2e-3):
    scaled = np.linspace(-1., 1., 21)
    dipoles = []
    for field in scaled*window:
        mf = scf.RHF(mol)
        mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-11
        mf.get_hcore = lambda *args: hcore+field*z
        mf.kernel()
        assert mf.converged
        dipoles.append(nuclear_z-np.einsum("pq,qp->", mf.make_rdm1(), z))
    coefficients = np.polynomial.polynomial.polyfit(scaled, dipoles, 5)
    beta = 2*coefficients[2]/window**2
    gamma = 6*coefficients[3]/window**3
    fits.append(dict(window_au=window, dipoles=dipoles, beta_zzz=float(beta), gamma_zzzz=float(gamma),
                     beta_error=float(abs(beta-data["beta_zzz"])), gamma_error=float(abs(gamma-data["gamma_zzzz"]))))
    print(f"PySCF window={window:.1e}: beta={beta:.8f}, gamma={gamma:.8f}")
np.testing.assert_allclose([beta, gamma], [data["beta_zzz"], data["gamma_zzzz"]],
                           atol=2e-2, rtol=2e-3)
(output / "pyscf.json").write_text(json.dumps(dict(pyscf_version=pyscf.__version__,
    degree=5, finite_field_fits=fits), indent=2) + "\n")

# jy_gpu_1, CPU float64, 2026-10-15:
# window 1e-2 au: beta=10.89022599, gamma=169.95014639
# window 5e-3 au: beta=10.89022339, gamma=169.94985376
# window 2e-3 au: beta=10.89022368, gamma=169.95270689
