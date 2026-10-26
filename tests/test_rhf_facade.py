"""The RHF object shares the restricted SCF and post-HF reference pipeline."""

import numpy as np
import pytest

from gradscf import bse, cc, ci, dft, fci, gto, gw, scf
from gradscf.model.neural_xc import training


@pytest.mark.parametrize('backend', ['full', 'df', 'direct'])
def test_rhf_matches_rks_hf_and_reuses_reference(backend):
    mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g')
    options = dict(conv_tol=1e-12, jk_backend=backend)
    if backend == 'df':
        options['auxbasis'] = 'def2-universal-jkfit'
    hf = scf.RHF(mol, **options).run()
    ks = dft.RKS(mol, xc='hf', **options).run()
    assert hf.converged and ks.converged
    np.testing.assert_allclose(hf.e_tot, ks.e_tot, rtol=0, atol=1e-12)
    np.testing.assert_allclose(hf.mo_energy, ks.mo_energy, rtol=0, atol=1e-12)
    np.testing.assert_allclose(hf.make_rdm1(), ks.make_rdm1(), rtol=0, atol=1e-12)
    ref = hf.to_reference()
    assert hf.to_reference() is ref
    assert training.Sample(hf, energy=hf.e_tot).molecule is ref
    assert hf.mo_coeff.ndim == 2
    hf.conv_tol = 1e-10
    with pytest.raises(RuntimeError, match='changed'):
        hf.to_reference()


def test_rhf_cannot_silently_switch_to_dft():
    mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g')
    with pytest.raises(TypeError):
        scf.RHF(mol, xc='pbe')
    mf = scf.RHF(mol)
    mf.xc = 'pbe'
    with pytest.raises(ValueError, match='RHF.*RKS'):
        mf.run()


def test_rhf_supplies_existing_post_hf_and_gw_bse_apis():
    mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g')
    mf = scf.RHF(mol, conv_tol=1e-12).run()
    exact = fci.FCI(mf, solver='dense').run()
    coupled = cc.CCSD(mf).run()
    assert exact.converged and coupled.converged
    np.testing.assert_allclose(coupled.e_tot, exact.e_tot, rtol=0, atol=1e-8)
    excited = ci.CIS(mf, nroots=1).run()
    assert np.isfinite(excited.e).all()
    qp = gw.GW(mf, nw=40).run()
    assert qp.converged
    response = bse.BSE(qp, nroots=1, tda=False).run()
    assert response.converged.all()
    assert np.isfinite(response.oscillator_strength()).all()
