"""Independent G/W windows, fixed-W iteration, and QP diagnostics."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import gto, scf, gw, bse


@pytest.fixture(scope='module')
def mf():
    return scf.RHF(gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g'),conv_tol=1e-12).run()


def test_default_and_explicit_full_windows_match(mf):
    old=gw.GW(mf,nw=40).run()
    new=gw.GW(mf,nw=40,g_orbitals=(0,1),screening_occupied=(0,),screening_virtual=(1,)).run()
    np.testing.assert_allclose(new.mo_energy,old.mo_energy,atol=1e-12)
    assert new.result.g_orbitals==(0,1)
    assert new.result.screening_occupied==(0,)
    assert np.isfinite(new.result.qp_weight).all()


@pytest.mark.parametrize('window',[dict(g_orbitals=()),dict(screening_occupied=()),dict(screening_virtual=())])
def test_empty_correlation_windows_recover_hf(mf,window):
    obj=gw.GW(mf,nw=40,**window).run()
    np.testing.assert_allclose(obj.mo_energy,mf.mo_energy,atol=1e-10)
    np.testing.assert_allclose(obj.result.qp_weight,1.,atol=1e-10)
    np.testing.assert_allclose(obj.result.sigma_qp,0.,atol=1e-12)


def test_evgw0_keeps_w0_and_bse_uses_recorded_window(mf):
    obj=gw.GW(mf,nw=40,method='evgw0',max_cycle=80,conv_tol=1e-9).run()
    assert obj.converged
    np.testing.assert_array_equal(obj.result.screening_energy,mf.mo_energy)
    assert np.max(np.abs(obj.result.qp_residual))<1e-9
    response=bse.BSE(obj,nroots=1).run()
    assert response.converged.all()
    obj.method='evgw'
    with pytest.raises(RuntimeError,match='changed'):
        response.oscillator_strength()


def test_evgw_updates_both_spectra(mf):
    obj=gw.GW(mf,nw=40,method='evgw',max_cycle=80,conv_tol=1e-9).run()
    np.testing.assert_array_equal(obj.result.screening_energy,obj.mo_energy)
    assert obj.converged


def test_qp_coverage_and_failed_rerun(mf):
    obj=gw.GW(mf,nw=40).run(orbs=(0,))
    assert np.isfinite(obj.result.qp_weight[0])
    assert np.isnan(obj.result.qp_weight[1])
    obj.g_orbitals=(2,)
    with pytest.raises(ValueError):obj.kernel()
    assert obj.result is None and not obj.converged


def test_g_window_partitions_correlation_without_changing_w(mf):
    from gradscf.gw.g0w0 import g0w0_cd_restricted
    obj=gw.GW(mf)
    r=mf.scf_result
    args=dict(mo_energy=r.mo_energy,mo_coeff=r.mo_coeff,nocc=1,df_factors=obj._df_factors(),
              fock_matrix=r.fock_matrix,hcore_matrix=r.hcore_matrix,density_matrix=r.density_matrix,
              nw=40,evaluate_only=True)
    full=g0w0_cd_restricted(**args)
    parts=[g0w0_cd_restricted(**args,g_orbitals=(i,)) for i in (0,1)]
    np.testing.assert_allclose(full.sigma_qp,sum(p.sigma_qp for p in parts),atol=1e-12)
    for p in parts:np.testing.assert_array_equal(p.screening_energy,full.screening_energy)


def test_weight_factor_gradient_matches_reconverged_finite_difference(mf):
    from gradscf.gw.g0w0 import g0w0_cd_restricted
    r=mf.scf_result
    factors=gw.GW(mf)._df_factors()
    def loss(t):
        result=g0w0_cd_restricted(mo_energy=r.mo_energy,mo_coeff=r.mo_coeff,nocc=1,
            df_factors=factors*(1+.1*t),fock_matrix=r.fock_matrix,hcore_matrix=r.hcore_matrix,
            density_matrix=r.density_matrix,nw=40,g_orbitals=(0,))
        return result.qp_weight.sum()+result.mo_energy.sum()
    actual=jax.jit(jax.grad(loss))(0.)
    np.testing.assert_allclose(actual,(loss(1e-4)-loss(-1e-4))/2e-4,atol=2e-7)


def test_bse_default_inherits_empty_screening_and_explicit_override(mf):
    obj=gw.GW(mf,nw=40,screening_occupied=()).run()
    inherited=bse.BSE(obj,nroots=1,tda=False,solver='dense').run()
    ref=inherited.reference
    explicit=bse.BSE(ref,nroots=1,tda=False,solver='dense',screening_occupied=(),screening_virtual=(1,)).run()
    np.testing.assert_allclose(inherited.e,explicit.e,atol=1e-12)
    # Explicitly choosing a different screening model must have an effect.
    changed=bse.BSE(ref,nroots=1,tda=False,solver='dense',screening_occupied=(0,)).run()
    assert np.max(np.abs(changed.e-inherited.e))>1e-6
    omega=jnp.array([.2,.4])
    np.testing.assert_allclose(inherited.polarizability(omega,eta=.01),
        bse.polarizability(inherited.result,ref.dipole_mo,inherited.space,omega,eta=.01))


def test_mutable_window_input_does_not_mutate_bse_snapshot(mf):
    obj=gw.GW(mf,nw=40).run()
    ref=bse.BSE(obj,nroots=1).run().reference
    from dataclasses import replace
    window=[0]
    snapshot=replace(ref,screening_occupied=window)
    window.clear()
    assert snapshot.screening_occupied==(0,)
    obj.screening_occupied=[0]
    obj.run()
    obj.screening_occupied.clear()
    with pytest.raises(RuntimeError,match='changed'):
        obj.state_signature()


def test_noncontiguous_windows_and_independent_screening_response():
    from gradscf.integrals.molecular.jk import build_jk_from_df
    from gradscf.gw.g0w0 import g0w0_cd_restricted
    rng=np.random.default_rng(19)
    factors=jnp.asarray(rng.normal(size=(4,5,5))*.03)
    factors=(factors+factors.swapaxes(1,2))/2
    energy=jnp.array([-1.2,-.7,.2,.6,.9])
    density=jnp.diag(jnp.array([2.,2.,0.,0.,0.]))
    j,k=build_jk_from_df(factors,density)
    args=dict(mo_energy=energy,mo_coeff=jnp.eye(5),nocc=2,df_factors=factors,
              fock_matrix=jnp.diag(energy),hcore_matrix=jnp.diag(energy)-j+k/2,
              density_matrix=density,nw=32,orbs=(1,3),g_orbitals=(4,0,2),evaluate_only=True)
    direction=jnp.array([-.05,.03,.04,0.,.07])
    def loss(t,oi=(1,0),va=(4,2)):
        r=g0w0_cd_restricted(**args,screening_energy=energy+t*direction,
                            screening_occupied=oi,screening_virtual=va)
        return jnp.real(r.sigma_qp).sum()+r.qp_weight[jnp.array([1,3])].sum()
    np.testing.assert_allclose(loss(0.),loss(0.,(0,1),(2,4)),atol=1e-12)
    grad=jax.jit(jax.grad(loss))(0.)
    assert abs(float(grad))>1e-9
    np.testing.assert_allclose(jax.jvp(loss,(0.,),(1.,))[1],grad,atol=1e-10)
    np.testing.assert_allclose(grad,(loss(1e-4)-loss(-1e-4))/2e-4,atol=1e-9)
