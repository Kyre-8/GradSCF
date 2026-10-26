"""Water MP2--MP8: GradSCF Taylor series versus independent PySCF actions.

PySCF 2.9.0 has native MP2, not a native MP3/MP4 interface. Higher-order
reference coefficients below use its full-space FCI Hamiltonian action with
the textbook Rayleigh--Schrodinger recurrence, independently of GradSCF.
Both packages solve their own RHF references. All energies are Hartree.
"""
from time import perf_counter
from math import comb

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf, mp
from pyscf import gto as pyscf_gto, scf as pyscf_scf, mp as pyscf_mp, fci
from _pyscf_reference import mp_coefficients

atom = "O 0 0 0; H 0 .75 .58; H 0 -.75 .58"
order = 8

# GradSCF calculation; no PySCF data enter this path.
start = perf_counter()
mol = gto.M(atom=atom, basis="sto-3g", cart=True)
mf = scf.RHF(mol, conv_tol=1e-13, conv_tol_grad=1e-11).run()
if not mf.converged:
    raise RuntimeError("GradSCF RHF did not converge")
pt = mp.MP(mf, order=order, with_t2=False).run()
gradscf_seconds = perf_counter()-start

# Independent reference: PySCF integrals, orbitals and full determinant action.
start = perf_counter()
ref_mol = pyscf_gto.M(atom=atom, basis="sto-3g", cart=True, verbose=0)
ref_mf = pyscf_scf.RHF(ref_mol).run(conv_tol=1e-13, conv_tol_grad=1e-11)
if not ref_mf.converged:
    raise RuntimeError("PySCF RHF did not converge")
nmo = ref_mf.mo_coeff.shape[1]
nocc = ref_mol.nelectron//2
expected = mp_coefficients(ref_mf, order)
reference_seconds = perf_counter()-start

native_mp2 = pyscf_mp.MP2(ref_mf).run()
exact = fci.FCI(ref_mf)
exact.conv_tol = 1e-12
fci_energy = exact.kernel()[0]
actual = np.asarray(pt.corrections)
np.testing.assert_allclose(mf.e_tot, ref_mf.e_tot, atol=2e-10, rtol=0.)
np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0.)
np.testing.assert_allclose(pt.e2, native_mp2.e_corr, atol=2e-10, rtol=0.)

print("H2O / STO-3G, all electrons, Cartesian, CPU float64")
print("RHF: GradSCF = %.12f  PySCF = %.12f" % (mf.e_tot, ref_mf.e_tot))
print("Retained determinants = %d; reference determinants = %d" %
      (pt.space.size, comb(nmo, nocc)**2))
print("Order   GradSCF correction    PySCF-action correction   absolute difference")
for degree, value, reference in zip(range(2, order+1), actual, expected):
    print("%3d     % .12f         % .12f           %.3e" %
          (degree, value, reference, abs(value-reference)))
print("MP8 total = %.12f; PySCF FCI total = %.12f" % (pt.e_tot, fci_energy))
print("Cold elapsed: GradSCF RHF + Taylor = %.3f s; PySCF RHF + RS = %.3f s" %
      (gradscf_seconds, reference_seconds))

# Larger-basis native MP2 comparison. High-order full-space references above
# deliberately use a small basis; this does not establish large-system scaling.
mol = gto.M(atom=atom, basis="6-31g*", cart=True)
mf = scf.RHF(mol, conv_tol=1e-13, conv_tol_grad=1e-11).run()
pt2 = mp.MP2(mf, frozen=1, with_t2=False).run()
ref_mol = pyscf_gto.M(atom=atom, basis="6-31g*", cart=True, verbose=0)
ref_mf = pyscf_scf.RHF(ref_mol).run(conv_tol=1e-13, conv_tol_grad=1e-11)
ref_pt2 = pyscf_mp.MP2(ref_mf, frozen=1).run()
if not mf.converged or not ref_mf.converged:
    raise RuntimeError("Converge both larger-basis RHF references")
np.testing.assert_allclose(pt2.e_corr, ref_pt2.e_corr, atol=2e-10, rtol=0.)
np.testing.assert_allclose(pt2.e_tot, ref_pt2.e_tot, atol=3e-10, rtol=0.)
print("H2O / 6-31G*, frozen=1: native MP2 correlation")
print("GradSCF = %.12f  PySCF = %.12f  difference = %.3e" %
      (pt2.e_corr, ref_pt2.e_corr, abs(pt2.e_corr-ref_pt2.e_corr)))

# Measured output (macOS arm64 CPU, Python 3.12.2, JAX 0.8.1, PySCF 2.9.0):
# RHF: GradSCF = -74.961376619994  PySCF = -74.961376619994
# Retained determinants = 441; reference determinants = 441
# Order   GradSCF correction    PySCF-action correction   absolute difference
#   2     -0.034747934183         -0.034747934183           7.339e-14
#   3     -0.009340345100         -0.009340345100           4.046e-14
#   4     -0.002813547195         -0.002813547195           1.718e-14
#   5     -0.000916760464         -0.000916760464           7.240e-15
#   6     -0.000318643445         -0.000318643445           3.748e-15
#   7     -0.000116080027         -0.000116080027           1.902e-15
#   8     -0.000043500906         -0.000043500906           9.195e-16
# MP8 total = -75.009673431315; PySCF FCI total = -75.009700279727
# Cold elapsed: GradSCF RHF + Taylor = 5.600 s; PySCF RHF + RS = 0.043 s
# These regions have different AD/compilation work; this is not a speed benchmark.
# H2O / 6-31G*, frozen=1: native MP2 correlation
# GradSCF = -0.185314572684  PySCF = -0.185314572684  difference = 4.241e-14
