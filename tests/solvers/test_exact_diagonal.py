"""Exact diagonal structure is opt-in; implicit AD owns all responses."""
from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import LinearOperator, LinearSolverConfig, solve_linear
from gradscf.solvers.operators import as_operator


def _operator(diagonal, *, exact=True):
    return LinearOperator((diagonal.size,)*2, diagonal.dtype, lambda x: diagonal*x,
                          diagonal=diagonal, exact_diagonal=exact)


def test_exact_structure_bypasses_dense_and_gmres_without_implicit_detection(monkeypatch):
    implicit = import_module('gradscf.solvers.linear.implicit')
    d = jnp.linspace(.7, 2.3, 9)
    rhs = jnp.arange(27., dtype=d.dtype).reshape(9, 3)/11
    cfg = LinearSolverConfig(method='direct', max_dense=1, rtol=1e-12, atol=1e-14)

    def forbidden(*args, **kwargs):
        raise AssertionError('Exact structure reached a dense or Krylov numerical solver')

    monkeypatch.setattr(implicit, 'direct_solve', forbidden)
    monkeypatch.setattr(implicit, 'gmres_solve', forbidden)
    op = _operator(d)
    assert op.T.exact_diagonal and op.T.T.exact_diagonal
    actual = solve_linear(op.T, rhs, config=cfg)
    assert actual.converged
    np.testing.assert_allclose(actual.solution, rhs/d[:, None], atol=1e-13)
    cfg = LinearSolverConfig(rtol=1e-12, restart=1, maxiter=1)
    assert solve_linear(op, rhs[:, 0], config=cfg).converged
    ordinary = LinearOperator(op.shape, op.dtype, op.matvec, diagonal=d)
    assert not ordinary.exact_diagonal and not ordinary.T.exact_diagonal
    assert not as_operator(jnp.diag(d)).exact_diagonal
    with pytest.raises(AssertionError, match='Krylov'):
        solve_linear(ordinary, rhs[:, 0], config=cfg)


@pytest.mark.parametrize('block', [False, True], ids=['vector', 'multi-rhs'])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_joint_diagonal_rhs_jvp_vjp_hvp_and_dtype(block, dtype):
    d = jnp.asarray([-2., -.3, .4, 3.], dtype)
    b = jnp.asarray(np.random.default_rng(721).normal(size=(4, 3) if block else (4,)), dtype)
    point = (d, b)
    direction = (.03*jnp.ones_like(d), .02*jnp.ones_like(b))
    cfg = LinearSolverConfig(rtol=1e-6 if dtype == jnp.float32 else 1e-12, atol=0.)
    probe = jnp.arange(b.size, dtype=dtype).reshape(b.shape)*.1+.2

    def evaluate(values, exact):
        diagonal, rhs = values
        return (solve_linear(_operator(diagonal), rhs, config=cfg).solution if exact
                else rhs/diagonal.reshape((-1,)+(1,)*(rhs.ndim-1)))

    def bundle(values, exact):
        function = lambda p: evaluate(p, exact)
        value, tangent = jax.jvp(function, (values,), (direction,))
        _, pullback = jax.vjp(function, values)
        gradient, = pullback(probe)
        loss = lambda p: jnp.sum((function(p)*probe)**2)
        hvp = jax.jvp(jax.grad(loss), (values,), (direction,))[1]
        return value, tangent, gradient, hvp

    actual, expected = jax.jit(lambda x: bundle(x, True))(point), bundle(point, False)
    tolerance = 3e-5 if dtype == jnp.float32 else 3e-12
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert got.dtype == dtype
        np.testing.assert_allclose(got, want, atol=tolerance, rtol=tolerance)
    promoted = solve_linear(_operator(d.astype(jnp.float64)), b.astype(jnp.float32), config=cfg)
    assert promoted.solution.dtype == jnp.float64 and promoted.converged


@pytest.mark.parametrize('block', [False, True])
def test_zero_rhs_retains_rhs_jvp_transpose_and_second_parameter_response(block):
    d = jnp.array([.7, 1.2, -2.])
    b = jnp.zeros((3, 2) if block else (3,))
    direction = jnp.arange(b.size, dtype=b.dtype).reshape(b.shape)*.1+.2
    dd = jnp.array([.1, -.07, .03])
    cfg = LinearSolverConfig(rtol=1e-12, atol=1e-14)
    function = lambda rhs: solve_linear(_operator(d), rhs, config=cfg).solution
    expected = direction/d.reshape((-1,)+(1,)*(b.ndim-1))
    np.testing.assert_allclose(jax.jvp(function, (b,), (direction,))[1], expected, atol=1e-13)
    np.testing.assert_allclose(jax.linear_transpose(function, b)(direction)[0], expected, atol=1e-13)

    def moving(t):
        return solve_linear(_operator(d+t*dd), (t+.5*t*t)*direction, config=cfg).solution

    second = jax.jit(jax.jacfwd(jax.jacfwd(moving)))(0.)
    denominator = d.reshape((-1,)+(1,)*(b.ndim-1))
    derivative = dd.reshape(denominator.shape)
    np.testing.assert_allclose(second, expected-2*direction*derivative/denominator**2, atol=1e-12)


@pytest.mark.parametrize('bad', [0., jnp.nan, jnp.inf])
@pytest.mark.parametrize('zero_rhs', [False, True])
def test_singular_nonfinite_factor_is_invalid_even_for_zero_rhs(bad, zero_rhs):
    d = jnp.array([1., bad, 2.])
    b = jnp.zeros(3) if zero_rhs else jnp.ones(3)
    function = lambda rhs: solve_linear(_operator(d), rhs).solution
    out = solve_linear(_operator(d), b)
    assert not out.converged and np.isnan(out.solution).all()
    assert np.isnan(jax.jvp(function, (b,), (jnp.ones_like(b),))[1]).all()
    assert np.isnan(jax.linear_transpose(function, b)(jnp.ones_like(b))[0]).all()


def test_stale_diagonal_is_checked_again_for_tangent_and_transpose_rhs():
    a, stale = jnp.diag(jnp.array([2., 3.])), jnp.array([2., 4.])
    op = LinearOperator(a.shape, a.dtype, lambda v: a@v, diagonal=stale, exact_diagonal=True)
    cfg = LinearSolverConfig(rtol=1e-12, atol=1e-14)
    function = lambda rhs: solve_linear(op, rhs, config=cfg).solution
    assert solve_linear(op, jnp.array([1., 0.]), config=cfg).converged
    assert not solve_linear(op, jnp.array([0., 1.]), config=cfg).converged
    zero, probe = jnp.zeros(2), jnp.array([0., 1.])
    assert np.isnan(jax.jvp(function, (zero,), (probe,))[1]).all()
    assert np.isnan(jax.linear_transpose(function, zero)(probe)[0]).all()


def test_exact_structure_validation_and_cholesky_conflict():
    d = jnp.ones(2)
    with pytest.raises(ValueError, match='diagonal'):
        LinearOperator((2, 2), d.dtype, lambda x: x, exact_diagonal=True)
    with pytest.raises(ValueError, match='square|diagonal'):
        LinearOperator((2, 3), d.dtype, lambda x: x[:2], diagonal=d, exact_diagonal=True)
    with pytest.raises((TypeError, ValueError), match='bool|static'):
        LinearOperator((2, 2), d.dtype, lambda x: x, diagonal=d, exact_diagonal=jnp.asarray(True))
    with pytest.raises(ValueError, match='Cholesky|diagonal|factor'):
        solve_linear(_operator(d), d, config=LinearSolverConfig(method='direct'), cholesky=jnp.eye(2))
    with pytest.raises(NotImplementedError, match='real'):
        solve_linear(_operator(d.astype(jnp.complex128)), d)
