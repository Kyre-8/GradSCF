"""Explicit contraction data must reach the native SCF facade unchanged."""
import numpy as np
import pytest


@pytest.mark.parametrize("unrestricted", [False, True])
def test_he_dication_raw_basis_matches_pyscf(unrestricted):
    from gradscf import gto, scf
    from pyscf import gto as pgto, scf as pscf
    basis = {"He": [[0, [13.6267,.07285253393590319],
                    [1.99935,.3714689298559184], [.382993,.2503930524532253]]]}
    atom = "He 0 0 0; He 0 0 1.1"
    dm = np.array([np.diag([1.,0.]),np.diag([0.,1.])]) if unrestricted else None
    mol = gto.M(atom=atom,basis=basis,charge=2)
    mf = (scf.UHF if unrestricted else scf.RHF)(mol,init_guess=dm,
        conv_tol=1e-13,conv_tol_grad=1e-10,max_cycle=200).run()
    refmol = pgto.M(atom=atom,basis=basis,charge=2,cart=True,verbose=0)
    ref = (pscf.UHF if unrestricted else pscf.RHF)(refmol).run(
        dm,conv_tol=1e-13,conv_tol_grad=1e-10,max_cycle=200)
    assert mf.converged and ref.converged
    np.testing.assert_allclose(mf.e_tot,ref.e_tot,atol=2e-10,rtol=0.)
