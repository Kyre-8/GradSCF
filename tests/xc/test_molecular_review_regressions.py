"""Review regressions at the custom-SCF and explicit-XC consumer boundaries."""
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import bse, dft, gto, gw, tdscf


def _custom_source(*, restricted=True):
    mol = gto.M(atom='H 0 0 0; H 0 0 .8', basis='sto-3g',
                charge=0 if restricted else 1, spin=0 if restricted else 1)
    functional = dft.custom_xc(
        input_fn=lambda state: (state.rho, state.tau, state.weights),
        energy_fn=lambda params, inputs: params['scale']*jnp.dot(
            inputs[2], .08*inputs[0]**2+.015*inputs[1]))
    mf = (dft.RKS if restricted else dft.UKS)(mol, xc=functional, params={'scale': jnp.asarray(.3)},
        grids_level=0, max_cycle=60, conv_tol=1e-11, conv_tol_grad=1e-8).run()
    assert mf.converged
    return mf


def test_custom_rks_retains_integral_inputs_for_actual_g0w0_and_bse():
    mf = _custom_source()
    solved, inputs, reference = mf.scf_result, mf._scf_inputs, mf.reference
    calculation = gw.GW(mf, method='g0w0', nw=16).run()
    assert calculation.converged
    assert np.isfinite(calculation.mo_energy).all()
    response = bse.BSE(calculation, nroots=1).run()
    assert np.asarray(response.converged).all()
    assert np.isfinite(response.e).all()
    assert np.isfinite(response.oscillator_strength()).all()
    # Consumers reuse the solved custom state and its original AO integrals.
    assert inputs is not None
    assert mf.scf_result is solved
    assert mf._scf_inputs is inputs
    assert mf.reference is reference
    assert inputs.dipole_integrals is not None
    np.testing.assert_allclose(inputs.overlap, solved.overlap_matrix, rtol=0, atol=0)


def test_custom_rks_response_refresh_preserves_the_solved_custom_state():
    mf = _custom_source()
    before, solved = mf.to_reference(), mf.scf_result
    mf.hfx_chunk_size += 1  # A response cache control, not a new SCF problem.
    after = mf.to_reference()
    assert after is not before
    assert mf.scf_result is solved
    assert after.exact_exchange_fraction == 0.
    assert after.xc_spec is None
    assert after.fock_matrix is not None
    assert after.ao_laplacian is not None
    np.testing.assert_allclose(after.rdm1, before.rdm1, rtol=0, atol=0)
    np.testing.assert_allclose(after.mf_energy, before.mf_energy, rtol=0, atol=0)
    np.testing.assert_allclose(after.fock_matrix, before.fock_matrix, rtol=0, atol=0)
    np.testing.assert_allclose(after.ao_laplacian, before.ao_laplacian, rtol=0, atol=0)


@pytest.fixture(scope='module', params=['pbe', 'cam-b3lyp'])
def traditional_source(request):
    mf = dft.RKS(gto.M(atom='H 0 0 0; H 0 0 .8', basis='sto-3g'),
        xc=request.param, grids_level=0, conv_tol=1e-11, conv_tol_grad=1e-8).run()
    assert mf.converged
    return mf, request.param


@pytest.mark.parametrize('method', ['TDA', 'TDDFT'])
def test_explicit_traditional_override_matches_explicit_string(traditional_source, method):
    mf, spec = traditional_source
    driver = getattr(tdscf, method)
    expected = driver(mf, xc_functional=spec, nstates=1).kernel().excitation_energies
    actual = driver(mf, xc_functional=dft.custom_xc(spec), nstates=1).kernel().excitation_energies
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-10)


@pytest.mark.parametrize('spin', ['restricted', 'unrestricted'])
@pytest.mark.parametrize('method', ['TDA', 'TDDFT'])
def test_traditional_override_does_not_inherit_custom_model_parameters(spin, method):
    mf = _custom_source(restricted=spin == 'restricted')
    driver = getattr(tdscf, method)
    expected = driver(mf.to_reference(), xc_functional='pbe',
                      nstates=1).kernel().excitation_energies
    for override in ('pbe', dft.custom_xc('pbe')):
        actual = driver(mf, xc_functional=override, nstates=1).kernel().excitation_energies
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-10)
    same_model = driver(mf, xc_functional=mf.xc, nstates=1).kernel().excitation_energies
    automatic = driver(mf, nstates=1).kernel().excitation_energies
    np.testing.assert_allclose(same_model, automatic, rtol=0, atol=2e-10)
    other_params = {'scale': jnp.asarray(.1)}
    explicit = driver(mf, xc_functional=mf.xc, xc_params=other_params,
                      nstates=1).kernel().excitation_energies
    reference_explicit = driver(mf.to_reference(), xc_functional=mf.xc,
        xc_params=other_params, nstates=1).kernel().excitation_energies
    np.testing.assert_allclose(explicit, reference_explicit, rtol=0, atol=2e-10)
