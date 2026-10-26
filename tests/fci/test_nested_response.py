"""FCI validity guards remain explicit across nested JIT and first response."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import fci
from gradscf.solvers import EigenResponseConfig,EigenSolverConfig,LinearSolverConfig


def problem(target,invalid=False):
    space=fci.make_fci_space(3,(1,0))
    base=jnp.diag(jnp.array([0.,0. if target=='subspace' else .7,3.]))
    perturb=jnp.array([[.3,.1,.2],[.1,-.2,-.4],[.2,-.4,.1]])
    def solve(t):
        h=base+t*perturb
        if invalid:
            h=h.at[0,1].add(.1)
        return fci.solve_fci(h,jnp.zeros((3,)*4),space,ecore=.13*t,
            config=EigenSolverConfig(method='dense',nroots=2,atol=1e-12),
            response=EigenResponseConfig(target=target,linear_config=LinearSolverConfig(
                method='direct',max_dense=16)),
            probes=jnp.eye(3) if target=='subspace' else None)
    def observable(t):
        result=solve(t)
        if target=='subspace':
            return result.energy_sum,result.projection
        coefficients=result.coefficients.reshape(2,3)
        return result.total_energies,jnp.einsum('ip,iq->ipq',coefficients,coefficients)
    return solve,observable


@pytest.mark.parametrize('target',['eigenpairs','subspace'])
def test_nested_jit_eigenpair_and_subspace_jvp(target):
    _,function=problem(target)
    inner=jax.jit(function)
    value,tangent=jax.jit(lambda t:jax.jvp(inner,(t,),(jnp.ones_like(t),)))(jnp.asarray(0.))
    if target=='subspace':
        np.testing.assert_allclose(value[0],0.,atol=1e-13)
        np.testing.assert_allclose(tangent[0],.36,atol=1e-12)
        expected=np.array([[0.,0.,-.2/3],[0.,0.,.4/3],[-.2/3,.4/3,0.]])
    else:
        np.testing.assert_allclose(value[0],[0.,.7],atol=1e-13)
        np.testing.assert_allclose(tangent[0],[.43,-.07],atol=1e-12)
        expected=np.array([[[0.,-.1/.7,-.2/3],[-.1/.7,0.,0.],[-.2/3,0.,0.]],
                           [[0.,.1/.7,0.],[.1/.7,0.,.4/2.3],[0.,.4/2.3,0.]]])
    np.testing.assert_allclose(tangent[1],expected,atol=1e-12,rtol=0)
    loss=lambda t:sum(jnp.sum(x) for x in inner(t))
    reverse=jax.jit(jax.grad(loss))(jnp.asarray(0.))
    np.testing.assert_allclose(reverse,sum(np.sum(np.asarray(x)) for x in tangent),atol=1e-12)


@pytest.mark.parametrize('target',['eigenpairs','subspace'])
def test_nested_jit_invalid_fci_still_rejects_response(target):
    solve,function=problem(target,invalid=True)
    result=jax.jit(solve)(jnp.asarray(.1))
    assert not np.any(result.converged | result.response_valid)
    inner=jax.jit(function)
    value,tangent=jax.jit(lambda t:jax.jvp(inner,(t,),(jnp.ones_like(t),)))(jnp.asarray(.1))
    assert not all(np.isfinite(np.asarray(x)).all() for x in value)
    assert not all(np.isfinite(np.asarray(x)).all() for x in tangent)
