"""Native GW/BSE integration, provenance, partial QP coverage and limits."""

from gradscf.bse.reference import reference_from_source

from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def h2():
    from gradscf import dft, gto
    from gradscf.gw import GW

    mf = dft.RKS(
        gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g"), xc="hf", conv_tol=1e-12
    ).run()
    calculation = GW(mf, nw=24).run()
    assert calculation.converged
    return mf, calculation


def test_native_g0w0_bse_and_explicit_snapshot(h2):
    from gradscf import bse

    _, gw = h2
    np.testing.assert_array_equal(gw.result.qp_computed_mask, [True, True])
    reference = reference_from_source(gw)
    obj = bse.BSE(gw, nroots=1).run()
    explicit = bse.BSE(reference, nroots=1).run()
    assert obj.converged.all()
    np.testing.assert_allclose(obj.e, explicit.e, atol=1e-12)
    np.testing.assert_allclose(
        obj.oscillator_strength(), explicit.oscillator_strength(), atol=1e-12
    )
    assert np.isfinite(obj.transition_dipole()).all()
    assert obj.oscillator_strength()[0] > 0
    obj.nroots = 2
    with pytest.raises(RuntimeError, match="changed"):
        obj.oscillator_strength()


def test_partial_qp_coverage_is_not_convergence(h2):
    from gradscf import bse
    from gradscf.gw import GW

    mf, _ = h2
    partial = GW(mf, nw=24).run(orbs=[0])
    np.testing.assert_array_equal(partial.result.qp_computed_mask, [True, False])
    assert partial.result.converged_mask[1]  # legacy status does not mean QP coverage
    with pytest.raises(ValueError, match="QP"):
        bse.BSE(partial, nroots=1).run()


def test_stale_gw_and_bse_sources(h2):
    from gradscf import bse

    _, gw = h2
    obj = bse.BSE(gw, nroots=1).run()
    old = gw.eta
    try:
        gw.eta *= 2
        with pytest.raises(RuntimeError, match="changed"):
            bse.BSE(gw, nroots=1).run()
        with pytest.raises(RuntimeError, match="changed"):
            obj.oscillator_strength()
    finally:
        gw.eta = old
    ref = reference_from_source(gw)
    obj = bse.BSE(ref, nroots=1).run()
    obj.source = replace(ref, qp_energy=ref.qp_energy + 0.01)
    with pytest.raises(RuntimeError, match="changed"):
        obj.transition_dipole()


def test_full_chain_fixed_mo_factor_derivative(h2):
    from gradscf import bse
    from gradscf.gw import g0w0_cd_restricted

    mf, gw = h2
    ao = gw.get_bse_inputs()["ao_factors"]
    data = mf.scf_result
    space = bse.make_bse_space(2, 1)
    cfg = bse.BSEConfig(nroots=1, solver="dense")

    def value(x):
        factors = ao * x
        qp = g0w0_cd_restricted(
            mo_energy=data.mo_energy,
            mo_coeff=data.mo_coeff,
            nocc=1,
            df_factors=factors,
            fock_matrix=data.fock_matrix,
            hcore_matrix=data.hcore_matrix,
            density_matrix=data.density_matrix,
            nw=24,
        )
        l = jnp.einsum("Pmn,mi,nj->Pij", factors, data.mo_coeff, data.mo_coeff)
        return bse.run_bse(
            qp.mo_energy,
            data.mo_energy,
            l,
            space,
            qp_computed_mask=qp.qp_computed_mask,
            qp_converged_mask=qp.converged_mask,
            config=cfg,
        ).excitation_energies[0]

    actual = jax.jit(jax.grad(value))(1.0)
    np.testing.assert_allclose(
        actual, (value(1.0 + 1e-4) - value(1.0 - 1e-4)) / 2e-4, atol=3e-7, rtol=0
    )


def test_resource_checks_precede_gw_factor_transform(h2, monkeypatch):
    from gradscf import bse
    import gradscf.gw.rgw as rgw

    _, gw = h2

    def forbidden(*args):
        raise AssertionError("MO factor allocation must not start")

    monkeypatch.setattr(rgw, "_mo_factors", forbidden)
    with pytest.raises(ValueError, match="max_aux"):
        bse.BSE(gw, nroots=1, max_aux=1).run()
    with pytest.raises(ValueError, match="max_factor_elements"):
        bse.BSE(gw, nroots=1, max_factor_elements=1).run()


def test_failed_rerun_cannot_mix_old_result_and_new_space():
    from gradscf import bse

    rng = np.random.default_rng(89)
    e = jnp.array([-1.0, -0.5, 0.3, 0.8, 1.4])
    l = rng.normal(size=(3, 5, 5)) * 0.04
    l = (l + l.transpose(0, 2, 1)) / 2
    ref = bse.BSEReference(e, e, jnp.asarray(l), 2, jnp.ones((3, 5, 5)))
    obj = bse.BSE(ref, nroots=2, solver="dense").run()
    obj.occupied, obj.virtual = (1,), (2,)
    with pytest.raises(ValueError, match="root"):
        obj.kernel()
    obj.occupied = obj.virtual = None
    assert obj.converged is None
    with pytest.raises(RuntimeError, match="Run BSE"):
        obj.oscillator_strength()


@pytest.mark.parametrize("orbs", [[], [-1], [2], [0, 0], [0.5]])
def test_invalid_qp_index_requests(h2, orbs):
    from gradscf.gw import GW

    mf, _ = h2
    with pytest.raises(ValueError, match="orbs"):
        GW(mf, nw=24).run(orbs=orbs)


def test_evaluate_only_does_not_claim_computed_qp(h2):
    from gradscf.gw import g0w0_cd_restricted

    mf, gw = h2
    data = mf.scf_result
    out = g0w0_cd_restricted(
        mo_energy=data.mo_energy,
        mo_coeff=data.mo_coeff,
        nocc=1,
        df_factors=gw.get_bse_inputs()["ao_factors"],
        fock_matrix=data.fock_matrix,
        hcore_matrix=data.hcore_matrix,
        density_matrix=data.density_matrix,
        nw=24,
        evaluate_only=True,
    )
    np.testing.assert_array_equal(out.qp_computed_mask, [False, False])


def test_lazy_dipole_completion_preserves_gw_but_invalidates_changed_optics(h2):
    from gradscf import bse

    mf, gw = h2
    obj = bse.BSE(gw, nroots=1).run()
    strength = np.asarray(obj.oscillator_strength())
    inputs, cached = mf._scf_inputs, gw._dipole_ao
    try:
        mf._scf_inputs = replace(inputs, dipole_integrals=cached)
        np.testing.assert_allclose(obj.oscillator_strength(), strength, atol=1e-13)
        mf._scf_inputs = replace(inputs, dipole_integrals=2 * cached)
        with pytest.raises(RuntimeError, match="changed"):
            obj.oscillator_strength()
        updated = bse.BSE(gw, nroots=1).run()
        np.testing.assert_allclose(
            updated.oscillator_strength(), 4 * strength, atol=1e-13
        )
    finally:
        mf._scf_inputs, gw._dipole_ao = inputs, cached
