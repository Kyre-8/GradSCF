"""Checked shared linear solver with multiple right-hand sides."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("method", ["direct", "gmres"])
def test_block_solve_response_and_failure(method):
    from gradscf.solvers import solve_linear, LinearSolverConfig

    cfg = LinearSolverConfig(method=method, rtol=1e-12)
    rhs = jnp.arange(6, dtype=float).reshape(2, 3) * 0.1

    def value(x):
        a = jnp.array([[2.0 + x, 0.3], [0.1, 1.4]])
        return jnp.sum(solve_linear(a, rhs, config=cfg).solution ** 2)

    out = solve_linear(jnp.array([[2.0, 0.3], [0.1, 1.4]]), rhs, config=cfg)
    assert out.converged
    np.testing.assert_allclose(
        out.solution, np.linalg.solve([[2.0, 0.3], [0.1, 1.4]], rhs), atol=1e-13
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(value))(0.0), (value(1e-5) - value(-1e-5)) / 2e-5, atol=1e-10
    )
    matrix = jnp.array([[2.0, 0.3], [0.1, 1.4]])
    action = lambda b: solve_linear(matrix, b, config=cfg).solution
    transposed = jax.linear_transpose(action, rhs)(jnp.ones_like(rhs))[0]
    np.testing.assert_allclose(transposed, np.linalg.solve(matrix.T, np.ones_like(rhs)),
                               atol=1e-12)
    bad = solve_linear(jnp.zeros((2, 2)), rhs, config=cfg)
    assert not bad.converged and np.isnan(bad.solution).all()


def test_cached_cholesky_response_and_transpose_failure():
    from gradscf.solvers import solve_linear, LinearSolverConfig

    cfg = LinearSolverConfig(method="direct", rtol=1e-12, atol=1e-14)
    rhs = jnp.array([[.2, .7], [.5, -.3]])

    def value(t, cached):
        matrix = jnp.array([[2.+t, .3], [.3, 1.4-.2*t]])
        factor = jnp.linalg.cholesky(matrix) if cached else None
        return solve_linear(matrix, rhs*(1+.1*t), config=cfg, cholesky=factor).solution

    np.testing.assert_allclose(jax.jit(lambda t: value(t, True))(0.), value(0., False), atol=1e-13)
    for transform in (lambda f: jax.jvp(f, (0.,), (1.,))[1],
                      lambda f: jax.grad(lambda t: jnp.sum(f(t)**2))(0.)):
        np.testing.assert_allclose(transform(lambda t: value(t, True)),
                                   transform(lambda t: value(t, False)), atol=1e-12)

    # A stale factor solves the first RHS exactly but fails on its adjoint.
    matrix = jnp.diag(jnp.array([2., 3.]))
    stale_factor = jnp.diag(jnp.sqrt(jnp.array([2., 4.])))
    action = lambda b: solve_linear(matrix, b, config=cfg, cholesky=stale_factor)
    assert action(jnp.array([1., 0.])).converged
    failed = action(jnp.array([0., 1.]))
    assert not failed.converged and np.isnan(failed.solution).all()
    transpose = jax.linear_transpose(lambda b: action(b).solution, jnp.zeros(2))
    assert np.isnan(transpose(jnp.array([0., 1.]))[0]).all()
