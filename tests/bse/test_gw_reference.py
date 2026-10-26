"""Recorded screening spectra distinguish G0W0/W0 from evGW/W reference data."""

from dataclasses import replace
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def inputs():
    from gradscf import gto, dft
    from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors

    mf = dft.RKS(gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g"), xc="hf").run()
    r = mf.scf_result
    factors = eri_pair_matrix_to_df_factors(
        mf._scf_inputs.eri_pair_matrix, nao=2, tol=1e-12
    )
    return dict(
        mo_energy=r.mo_energy,
        mo_coeff=r.mo_coeff,
        nocc=1,
        df_factors=factors,
        fock_matrix=r.fock_matrix,
        hcore_matrix=r.hcore_matrix,
        density_matrix=r.density_matrix,
    )


@pytest.mark.parametrize("method", ["g0w0", "evgw"])
def test_actual_gw_recorded_screening_and_bse_adapter(inputs, method):
    from gradscf import bse, gw
    from gradscf.gw.g0w0 import _mo_factors

    driver = gw.g0w0_cd_restricted if method == "g0w0" else gw.evgw_cd_restricted
    kwargs = {} if method == "g0w0" else dict(max_iter=50, tol=1e-9)
    result = driver(**inputs, nw=40, **kwargs)
    expected = inputs["mo_energy"] if method == "g0w0" else result.mo_energy
    np.testing.assert_array_equal(result.screening_energy, expected)
    if method == "g0w0":
        assert not np.allclose(result.screening_energy, result.mo_energy, atol=1e-6)
    factors = _mo_factors(inputs["df_factors"], inputs["mo_coeff"])
    ref = bse.BSEReference.from_gw_result(result, mo_factors=factors, nocc=1)
    space = bse.make_bse_space(2, 1)
    expected_bse = bse.run_bse(
        result.mo_energy,
        expected,
        factors,
        space,
        config=bse.BSEConfig(nroots=1, tda=False, solver="dense"),
    )
    actual = bse.BSE(ref, nroots=1, tda=False, solver="dense").run()
    np.testing.assert_allclose(actual.e, expected_bse.excitation_energies, atol=1e-12)
    with pytest.raises(ValueError, match="screening"):
        bse.BSEReference.from_gw_result(
            replace(result, screening_energy=None), mo_factors=factors, nocc=1
        )
    with pytest.raises(ValueError, match="coverage"):
        bse.BSEReference.from_gw_result(
            replace(result, qp_computed_mask=None), mo_factors=factors, nocc=1
        )


def test_custom_poles_are_recorded_and_evaluation_only_rejected(inputs):
    from gradscf import bse, gw
    from gradscf.gw.g0w0 import _mo_factors

    poles = jnp.asarray(inputs["mo_energy"]) + jnp.array([-0.03, 0.04])
    result = gw.g0w0_cd_restricted(
        **inputs, mo_energy_poles=poles, nw=40, evaluate_only=True
    )
    np.testing.assert_array_equal(result.screening_energy, poles)
    ref = bse.BSEReference.from_gw_result(
        result, mo_factors=_mo_factors(inputs["df_factors"], inputs["mo_coeff"]), nocc=1
    )
    with pytest.raises(ValueError, match="actually computed"):
        bse.BSE(ref, nroots=1).run()
