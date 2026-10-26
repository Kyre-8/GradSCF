"""AD-forward QP methods keep root, implicit response, and BSE conventions."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import gto,scf,gw,bse
from test_gw_differentiability import _h2_inputs


@pytest.mark.parametrize('method',['newton','hybrid'])
def test_molecular_root_weight_and_factor_gradient(method):
    inputs=_h2_inputs()
    def solve(t,solver):
        return gw.g0w0_cd_restricted(**{**inputs,'df_factors':inputs['df_factors']*(1+.05*t)},
                                     nw=40,qp_solver=solver)
    reference=solve(0.,'secant')
    actual=solve(0.,method)
    assert actual.converged
    np.testing.assert_allclose(actual.mo_energy,reference.mo_energy,atol=2e-8)
    np.testing.assert_allclose(actual.qp_weight,reference.qp_weight,atol=2e-8)
    loss=lambda t:jnp.sum(solve(t,method).mo_energy*jnp.array([.7,.3]))
    grad=jax.jit(jax.grad(loss))(0.)
    np.testing.assert_allclose(grad,(loss(1e-4)-loss(-1e-4))/2e-4,atol=2e-7)
    baseline=jax.jit(jax.grad(lambda t:jnp.sum(solve(t,'secant').mo_energy*jnp.array([.7,.3]))))(0.)
    np.testing.assert_allclose(grad,baseline,atol=2e-8)


def test_gw_facade_and_screening_snapshot_track_qp_method():
    mf=scf.RHF(gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g')).run()
    obj=gw.GW(mf,nw=40,qp_solver='newton').run()
    optical=bse.BSE(obj,nroots=1,tda=False).run()
    assert optical.converged.all()
    obj.qp_solver='hybrid'
    with pytest.raises(RuntimeError,match='changed'):optical.oscillator_strength()
    with pytest.raises(ValueError,match='outer'):
        gw.GW(mf,method='evgw0',qp_solver='newton').run()


@pytest.mark.parametrize('method',['newton','hybrid'])
def test_explicit_qp_trajectory_matches_implicit(method):
    from gradscf.gw.qp import solve_qp_batch
    from test_gw_qp import _context
    shared,stacked=_context()
    def loss(t,mode):
        roots,done=solve_qp_batch(jnp.array([-.5]),jnp.zeros(1),shared,
            {**stacked,'wmn_p':stacked['wmn_p']*t},occupied=jnp.array([True]),
            method=method,diff_mode=mode,tol=1e-10,maxiter=25)
        return roots.sum()
    implicit=jax.jit(jax.grad(lambda t:loss(t,'implicit')))(1.)
    explicit=jax.jit(jax.grad(lambda t:loss(t,'explicit')))(1.)
    np.testing.assert_allclose(explicit,implicit,atol=1e-9)


def test_unrestricted_facade_uses_shared_ad_root_methods():
    mf=scf.UHF(gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g')).run()
    reference=gw.UGW(mf,nw=40).run()
    for method in ('newton','hybrid'):
        out=gw.UGW(mf,nw=40,qp_solver=method).run()
        assert out.converged
        np.testing.assert_allclose(out.mo_energy,reference.mo_energy,atol=1e-8)


def test_alternative_root_method_requires_an_actual_root_solve():
    with pytest.raises(ValueError,match='nonlinear QP solve'):
        gw.g0w0_cd_restricted(**_h2_inputs(),nw=40,qp_solver='newton',linearized=True)
