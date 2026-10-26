"""Native GradSCF RHF -> CCSD -> singlet EE and doublet IP/EA energies.

Run on CPU with JAX float64. No PySCF solver is used. All energies are Hartree;
EA omega = E(N+1)-E(N), so the conventional electron affinity is -omega.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf, cc

mol = gto.M(
    atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587",
    basis="sto-3g",
    unit="Angstrom",
)
mf = scf.RHF(mol, conv_tol=1e-12).run()
mycc = cc.CCSD(mf, conv_tol=1e-12, residual_tol=1e-11).run()

for method in (cc.EOMEE, cc.EOMIP, cc.EOMEA):
    eom = method(mycc, nroots=3).run()
    if not np.all(eom.converged):
        raise RuntimeError(f"{eom.sector} failed to converge")
    print(f"{eom.sector.upper()} energies (Hartree):", np.asarray(eom.e))
    print("  right residuals:", np.asarray(eom.result.residual_norms))
    print("  left residuals: ", np.asarray(eom.result.left_residual_norms))
    print("  isolated energy response:", np.asarray(eom.result.response_valid))
    if eom.sector == "ea":
        print("  electron affinities:", -np.asarray(eom.e))
