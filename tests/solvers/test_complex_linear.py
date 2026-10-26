"""Complex systems reuse checked real solves, including their AD rules."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import LinearOperator, LinearSolverConfig


@pytest.mark.parametrize("method", ["direct", "gmres"])
def test_complex_vector_and_block_solve_and_response(method):
    from gradscf.solvers import solve_complex

    a = jnp.array([[1.2 + 0.3j, 0.2 - 0.1j], [-0.1 + 0.4j, 1.7 + 0.2j]])
    b = jnp.array([[0.4 + 0.3j, -0.2j], [0.1 - 0.2j, 0.6 + 0.1j]])
    cfg = LinearSolverConfig(method=method, rtol=1e-11, restart=4)
    for rhs in (b[:, 0], b):
        out = jax.jit(lambda x: solve_complex(x, rhs, config=cfg))(a)
        assert bool(out.converged)
        np.testing.assert_allclose(out.solution, np.linalg.solve(a, rhs), atol=2e-11)
    loss = lambda t: jnp.sum(
        abs(solve_complex(a + t * jnp.eye(2), b, config=cfg).solution) ** 2
    )
    expected = lambda t: jnp.sum(abs(jnp.linalg.solve(a + t * jnp.eye(2), b)) ** 2)
    np.testing.assert_allclose(
        jax.jit(jax.grad(loss))(0.1), jax.grad(expected)(0.1), atol=2e-10
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(jax.grad(loss)))(0.1),
        jax.grad(jax.grad(expected))(0.1),
        atol=2e-9,
    )
    op = LinearOperator(a.shape, a.dtype, lambda x: a @ x)
    np.testing.assert_allclose(
        solve_complex(op, b, config=cfg).solution, jnp.linalg.solve(a, b), atol=2e-11
    )


def test_complex_solve_preserves_failures_and_dense_bound():
    from gradscf.solvers import solve_complex

    bad = solve_complex(
        jnp.zeros((2, 2), complex),
        jnp.ones(2),
        config=LinearSolverConfig(method="direct"),
    )
    assert not bool(bad.converged) and np.isnan(bad.solution).all()
    with pytest.raises(ValueError, match="max_dense"):
        solve_complex(
            jnp.eye(3, dtype=complex),
            jnp.ones(3),
            config=LinearSolverConfig(method="direct", max_dense=5),
        )
