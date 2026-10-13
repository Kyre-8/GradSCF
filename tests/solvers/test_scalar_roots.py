"""Independent scalar roots: guarded AD updates, status, and explicit AD."""
import jax
import jax.numpy as jnp
import numpy as np


def test_bisection_preserves_brackets_and_reports_invalid_lane():
    from gradscf.solvers.nonlinear import bisect
    roots,converged=bisect(lambda x:x*x-jnp.array([2.,-1.]),jnp.zeros(2),jnp.ones(2)*2)
    np.testing.assert_allclose(roots[0],np.sqrt(2.),atol=1e-12)
    np.testing.assert_array_equal(converged,[True,False])
import pytest
from gradscf.solvers.nonlinear import ScalarRootConfig, solve_scalar_roots


@pytest.mark.parametrize('method',['secant','newton','hybrid'])
@pytest.mark.parametrize('scan',[False,True])
def test_roots_and_status(method,scan):
    cfg=ScalarRootConfig(method=method,ftol=1e-11,xtol=1e-11,step_cap=1.)
    out=jax.jit(lambda:solve_scalar_roots(lambda x:x*x-jnp.array([2.,3.]),jnp.ones(2),config=cfg,scan=scan))()
    assert out.converged.all()
    np.testing.assert_allclose(out.roots,np.sqrt([2.,3.]),atol=1e-10)
    assert np.max(np.abs(out.residual))<1e-11
    assert (out.iterations>0).all()
    assert (out.derivative_evaluations>0)==(method!='secant')


@pytest.mark.parametrize('method',['secant','newton','hybrid'])
def test_explicit_response(method):
    cfg=ScalarRootConfig(method=method,ftol=1e-12,xtol=1e-12,maxiter=25,step_cap=1.)
    def fun(a):return solve_scalar_roots(lambda x:x*x-a,jnp.array([1.]),config=cfg,scan=True).roots.sum()
    np.testing.assert_allclose(jax.jit(jax.grad(fun))(2.),1/(2*np.sqrt(2)),atol=1e-10)


@pytest.mark.parametrize('method',['secant','newton','hybrid'])
def test_no_false_success_or_nonfinite_best(method):
    out=solve_scalar_roots(lambda x:jnp.ones_like(x),jnp.array([0.]),config=ScalarRootConfig(method=method,maxiter=3))
    assert not out.converged.any()
    assert np.isfinite(out.roots).all()
    np.testing.assert_allclose(out.residual,1.)


def test_newton_backtracks_and_reports_actual_work():
    cfg=ScalarRootConfig(method='newton',step_cap=10.,max_backtrack=12,ftol=1e-10,xtol=1e-10)
    out=solve_scalar_roots(lambda x:jnp.arctan(x),jnp.array([2.]),config=cfg)
    assert out.converged.all() and out.backtracks>0
    assert out.derivative_evaluations>out.iterations.max()
    np.testing.assert_allclose(out.roots,0.,atol=1e-10)


def test_newton_recovers_from_nonfinite_trial():
    cfg=ScalarRootConfig(method='newton',step_cap=10.,max_backtrack=12,ftol=1e-11,xtol=1e-11)
    out=solve_scalar_roots(lambda x:jnp.log(x)+2,jnp.array([2.]),config=cfg)
    assert out.converged.all() and out.backtracks>0
    np.testing.assert_allclose(out.roots,np.exp(-2),atol=1e-10)


@pytest.mark.parametrize('method',['newton','hybrid'])
def test_discarded_domain_error_cannot_poison_explicit_gradient(method):
    cfg=ScalarRootConfig(method=method,step_cap=10.,max_backtrack=12,ftol=1e-11,xtol=1e-11,maxiter=50)
    def solve(a):return solve_scalar_roots(lambda x:jnp.sqrt(x)-a,jnp.array([2.]),config=cfg,scan=True)
    out=solve(.1)
    assert out.converged.all()
    np.testing.assert_allclose(out.roots,.01,atol=1e-10)
    np.testing.assert_allclose(jax.jit(jax.grad(lambda a:solve(a).roots.sum()))(.1),.2,atol=1e-9)
