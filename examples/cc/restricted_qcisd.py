"""Water/STO-3G RHF -> QCISD and QCISD(T), using GradSCF throughout.

Fixed geometry in Angstrom; energies in Hartree. No PySCF or CLI.
QCISD amplitudes are not interchangeable with converged CCSD amplitudes.
"""
import jax
import numpy as np
from gradscf import cc, gto, scf

jax.config.update("jax_enable_x64", True)
mol = gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="sto-3g")
mf = scf.RHF(mol, conv_tol=1e-12).run()
myqci = cc.QCISD(mf, conv_tol=1e-12, residual_tol=1e-11).run()
triples = myqci.triples()
dm1 = myqci.make_rdm1()
np.testing.assert_allclose(np.trace(dm1), 10., atol=1e-10)

print("RHF energy:", float(mf.e_tot))
print("QCISD energy:", float(myqci.e_tot))
print("QCISD(T) correction:", float(triples.energy))
print("QCISD(T) energy:", float(myqci.e_tot+triples.energy))
print("QCISD residual:", float(myqci.result.residual_norm))
print("QCISD model density trace:", float(np.trace(dm1)))
print("QCISD adjoint converged:", myqci.converged_lambda)
