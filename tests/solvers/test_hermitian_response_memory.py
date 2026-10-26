"""The original spectral charts share the primal MB budget at response time."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import (EigenResponseConfig, EigenSolverConfig,
                             LinearSolverConfig, solve_hermitian)
from gradscf.solvers.eigen import api, response


def _matrix(t):
    return jnp.array([[-2.+.2*t, .3*t, .1],
                      [.3*t, -.5-.1*t, .2*t], [.1, .2*t, 3.+.1*t]])


def _policy(target='eigenpairs', method='direct', max_dense=6):
    return EigenResponseConfig(target=target, linear_config=LinearSolverConfig(
        method=method, max_dense=max_dense, restart=6, maxiter=20,
        rtol=1e-12, atol=1e-14))


def test_response_inherits_original_mb_and_visible_owners_once(monkeypatch):
    plans = []
    original = api.plan_hermitian

    def primal(*args, **kwargs):
        plans.append((args[3], kwargs['resident_bytes']))
        return original(*args, **kwargs)

    class Captured(Exception):
        pass

    def checked(action, rhs, **kwargs):
        assert rhs.shape == (3,)
        assert kwargs.get('max_memory') == 32.
        assert kwargs.get('reserved_bytes', 0) >= 4096+9*8+3*8
        assert kwargs.get('operator_min_bytes', 0) > 0
        raise Captured

    monkeypatch.setattr(api, 'plan_hermitian', primal)
    monkeypatch.setattr(response, 'checked_linear_solve', checked)
    function = lambda t: solve_hermitian(_matrix(t),
        config=EigenSolverConfig(method='dense', nroots=2), response=_policy(),
        max_memory=32., reserved_bytes=4096).vectors
    with pytest.raises(Captured):
        jax.jvp(function, (.13,), (1.,))
    assert len(plans) == 1 and plans[0][0] == 32.


def test_response_minimum_rejects_before_attaching_root(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('A response that cannot fit reached the shared root')

    monkeypatch.setattr(response, 'attach_root', forbidden)
    function = lambda t: solve_hermitian(_matrix(t),
        config=EigenSolverConfig(method='dense', nroots=2), response=_policy(),
        max_memory=.002).vectors
    assert jnp.all(jnp.isfinite(function(.13)))
    with pytest.raises(ValueError, match='memory|budget|minimum'):
        jax.jvp(function, (.13,), (1.,))


@pytest.mark.parametrize('target,size', [('eigenpairs', 3), ('subspace', 6)])
def test_response_uses_original_isolated_or_cluster_dimension(target, size, monkeypatch):
    original = getattr(response, 'plan_linear', None)
    assert callable(original), 'Spectral response has no shared linear budget'
    dimensions = []

    def plan(n, dtype, config, maximum, **kwargs):
        dimensions.append(n)
        assert maximum == 32.
        return original(n, dtype, config, maximum, **kwargs)

    monkeypatch.setattr(response, 'plan_linear', plan)
    def function(t):
        result = solve_hermitian(_matrix(t),
            config=EigenSolverConfig(method='dense', nroots=2),
            response=_policy(target), probes=jnp.ones(3) if target == 'subspace' else None,
            max_memory=32.)
        return result.vectors if target == 'eigenpairs' else result.projection
    jax.jvp(function, (.13,), (1.,))
    assert dimensions and set(dimensions) == {size}


def test_cluster_direct_bound_and_lazy_energy_full_span_are_preserved(monkeypatch):
    cfg = EigenSolverConfig(method='dense', nroots=2)
    function = lambda t: solve_hermitian(_matrix(t), config=cfg,
        response=_policy('subspace', max_dense=3), probes=jnp.ones(3), max_memory=8.).projection
    with pytest.raises(ValueError, match='max_dense'):
        jax.jvp(function, (.13,), (1.,))

    def forbidden(*args, **kwargs):
        pytest.fail('A stationary first energy derivative or full-span subspace requested response')
    monkeypatch.setattr(response, '_implicit_frame', forbidden)
    energy = lambda t: solve_hermitian(_matrix(t), config=cfg,
        response=_policy('eigenvalues', max_dense=1), max_memory=.002).values.sum()
    assert jnp.isfinite(jax.grad(energy)(.13))
    full = lambda t: solve_hermitian(_matrix(t),
        config=EigenSolverConfig(method='dense', nroots=3), response=_policy('subspace'),
        probes=jnp.ones(3), max_memory=.002).eigenvalue_sum
    np.testing.assert_allclose(jax.grad(jax.grad(full))(.13), 0., atol=1e-12)


def test_budget_keeps_existing_batched_third_response(monkeypatch):
    import test_spectral_higher_response as old
    original = old.solve_hermitian
    monkeypatch.setattr(old, 'solve_hermitian',
        lambda *args, **kwargs: original(*args, **kwargs, max_memory=8.))
    old.test_mixed_third_derivative_of_batched_eigenvectors(True, 4)


@pytest.mark.parametrize('cluster', [False, True])
def test_budget_retains_joint_jvp_vjp_hvp_and_gap_guards(cluster):
    base = jnp.diag(jnp.array([1., 1. if cluster else 2., 4.]))
    b = jnp.array([[.2, .3, .4], [.3, -.1, .2], [.4, .2, .1]])
    c = jnp.array([[.1, -.2, .2], [-.2, .3, .1], [.2, .1, -.1]])
    probe = jnp.array([.3, .4, .5])
    cfg = EigenSolverConfig(method='dense', nroots=2, atol=1e-12)
    policy = _policy('subspace' if cluster else 'eigenpairs', method='gmres')
    point = jnp.array([0., 0.]) if cluster else jnp.array([.17, -.11])
    direction = jnp.array([.3, -.2])

    def outputs(x, budget):
        result = solve_hermitian(base+x[0]*b+x[1]*c+.2*x[0]*x[1]*b,
            config=cfg, response=policy, probes=probe if cluster else None,
            max_memory=budget)
        if cluster:
            return jnp.concatenate((result.projection, result.eigenvalue_sum[None]))
        return jnp.concatenate((result.values, (probe@result.vectors)**2))

    def evaluate(budget):
        f = lambda x: outputs(x, budget)
        values, tangent = jax.jvp(f, (point,), (direction,))
        weights = jnp.linspace(.2, .7, values.size)
        scalar = lambda x: weights@f(x)
        _, pullback = jax.vjp(f, point)
        gradient = pullback(weights)[0]
        hvp = jax.jvp(jax.grad(scalar), (point,), (direction,))[1]
        return values, tangent, gradient, hvp

    actual = jax.jit(lambda: evaluate(8.))()
    expected = jax.jit(lambda: evaluate(None))()
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, atol=3e-10, rtol=3e-9)
    weights = jnp.linspace(.2, .7, actual[0].size)
    gradient = jax.jit(jax.grad(lambda x: weights@outputs(x, None)))
    step = 2e-4
    finite = (gradient(point+step*direction)-gradient(point-step*direction))/(2*step)
    np.testing.assert_allclose(actual[-1], finite, atol=3e-8, rtol=3e-6)
    if cluster:
        invalid = solve_hermitian(base, config=EigenSolverConfig(method='dense', nroots=1),
            response=_policy(), max_memory=8.)
        assert not bool(jnp.all(invalid.response_valid))
        assert jnp.isnan(jax.grad(lambda t: solve_hermitian(base+t*jnp.eye(3),
            config=EigenSolverConfig(method='dense', nroots=1), response=_policy(),
            max_memory=8.).values[0])(0.))
