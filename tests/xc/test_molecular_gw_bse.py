"""Actual KS references retain complete XC subtraction and post-GW frames.

H2/STO-3G, native integrals, grid level1, float64. Small frequency grids test
consumer assembly/coverage, not basis or frequency convergence of a spectrum.
"""
import numpy as np
import pytest

from gradscf import bse, dft, gto, gw


@pytest.fixture(scope='module', params=['pbe', 'pbe0', 'r2scan', 'lc_wpbe'])
def source(request):
    mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g')
    mf = dft.RKS(mol, xc=request.param, grids_level=1,
                 integral_backend='native', conv_tol=1e-11, max_cycle=100).run()
    assert mf.converged
    return mf


def test_zero_correlation_removes_entire_starting_xc_potential(source):
    calculation = gw.GW(source, nw=16, screening_occupied=()).run()
    assert calculation.converged
    # Independent AO Hartree-Fock contraction in the unchanged KS orbital
    # frame: suppressing correlation removes all starting semilocal, tau and
    # screened-exchange terms, not just a scalar hybrid coefficient.
    factors = np.asarray(calculation._df_factors())
    density = np.asarray(source.make_rdm1())
    coulomb = np.einsum('Pij,Pkl,kl->ij', factors, factors, density)
    exchange = np.einsum('Pik,Pjl,kl->ij', factors, factors, density)
    coeff = np.asarray(source.mo_coeff)
    hf_fock = np.asarray(source.scf_result.hcore_matrix)+coulomb-.5*exchange
    expected = np.diag(coeff.T @ hf_fock @ coeff)
    np.testing.assert_allclose(calculation.mo_energy, expected, atol=2e-9, rtol=0)


@pytest.mark.parametrize('method', ['g0w0', 'evgw', 'qsgw', 'scgw'])
def test_four_restricted_gw_drivers_accept_complete_ks_sources(source, method):
    calculation = gw.GW(source, method=method, nw=16, max_cycle=80,
                        conv_tol=1e-7, damp=.3, beta=20., mixing=.5).run()
    assert calculation.converged
    if method == 'scgw':
        assert calculation.mo_energy is None
        assert np.isfinite(calculation.result.self_energy_iw).all()
        with pytest.raises(NotImplementedError, match='QP|continuation'):
            bse.BSE(calculation, nroots=1).run()
    else:
        assert np.isfinite(calculation.mo_energy).all()
        response = bse.BSE(calculation, nroots=1).run()
        assert np.asarray(response.converged).all()
        assert np.isfinite(response.e).all()
        assert np.isfinite(response.oscillator_strength()).all()
