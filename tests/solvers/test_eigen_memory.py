"""A requested array budget includes Hermitian solver state before allocation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import EigenSolverConfig, LinearOperator, solve_hermitian
from gradscf.solvers.eigen import primal


def _operator(n=32, calls=None):
    diagonal = jnp.arange(n, dtype=jnp.float64) + 1.
    coupling = jnp.full((n,), .02, dtype=diagonal.dtype)

    def apply(v):
        if calls is not None:
            calls.append(1 if v.ndim == 1 else v.shape[1])
        return diagonal.reshape((-1,) + (1,) * (v.ndim - 1)) * v + coupling.reshape(
            (-1,) + (1,) * (v.ndim - 1)) * jnp.sum(coupling.reshape(
                (-1,) + (1,) * (v.ndim - 1)) * v, axis=0)

    return LinearOperator((n, n), diagonal.dtype, apply,
                          diagonal=diagonal + coupling**2, matmat=apply)


@pytest.mark.parametrize("method", ["davidson", "dense"])
def test_impossible_budget_rejects_before_guesses_identity_or_action(method, monkeypatch):
    op = _operator(32)

    def forbidden(*args, **kwargs):
        pytest.fail("solver allocated a guess/identity before checking its budget")

    monkeypatch.setattr(jax.random, "normal", forbidden)
    monkeypatch.setattr(jnp, "eye", forbidden)
    with pytest.raises(ValueError, match="memory|budget|minimum"):
        solve_hermitian(op, config=EigenSolverConfig(method=method), max_memory=.001)


def test_budget_clips_optional_subspace_preserving_boundary_root(monkeypatch):
    op = _operator()
    configurations = []
    davidson = primal._davidson_lowest_symmetric

    def record(*args, **kwargs):
        configurations.append(kwargs)
        return davidson(*args, **kwargs)

    monkeypatch.setattr(primal, "_davidson_lowest_symmetric", record)
    cfg = EigenSolverConfig(nroots=2, max_subspace=24, maxiter=120, atol=1e-10)
    high = solve_hermitian(op, config=cfg, max_memory=1.)
    low = solve_hermitian(op, config=cfg, max_memory=.045)
    assert configurations[0]["max_subspace"] == 24
    assert 5 <= configurations[1]["max_subspace"] < 24
    assert configurations[1]["nroots"] == 3
    assert bool(jnp.all(low.converged))
    np.testing.assert_allclose(low.values, high.values, atol=2e-10, rtol=2e-10)
    assert cfg.max_subspace == 24


def test_memory_check_preserves_all_explicit_initial_columns(monkeypatch):
    op = _operator(32)
    initial = jnp.arange(32 * 12, dtype=jnp.float64).reshape(32, 12)

    def forbidden(*args, **kwargs):
        pytest.fail("initial vectors were truncated or solved before the budget error")

    monkeypatch.setattr(primal, "_davidson_lowest_symmetric", forbidden)
    with pytest.raises(ValueError, match="memory|budget|initial"):
        solve_hermitian(op, config=EigenSolverConfig(max_subspace=24),
                        initial_vectors=initial, max_memory=.015)


def test_forward_only_budget_does_not_reserve_a_hypothetical_response():
    op = _operator(8)
    cfg = EigenSolverConfig(method="dense")
    answer = solve_hermitian(op, config=cfg, max_memory=.012)
    assert bool(jnp.all(answer.converged))
    np.testing.assert_allclose(answer.values, jnp.linalg.eigvalsh(
        op.apply(jnp.eye(8)))[:1], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_column_batches_preserve_tail_dtype_and_nested_ad(dtype):
    from gradscf.solvers.memory import apply_columns

    op = _operator(5)
    vectors = jnp.arange(35, dtype=dtype).reshape(5, 7) / 13.
    direction = jnp.sin(vectors) * .07
    expected = op.apply(vectors)
    actual = apply_columns(op, vectors, 3)
    assert actual.dtype == expected.dtype
    tolerance = 2e-6 if dtype == jnp.float32 else 2e-12
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    functions = [lambda x: jnp.sum(jnp.sin(apply_columns(op, x, 3))),
                 lambda x: jnp.sum(jnp.sin(op.apply(x)))]
    output = [(jax.value_and_grad(f)(vectors),
               jax.jvp(jax.grad(f), (vectors,), (direction,))[1]) for f in functions]
    for low, high in zip(jax.tree.leaves(output[0]), jax.tree.leaves(output[1])):
        np.testing.assert_allclose(low, high, atol=tolerance, rtol=tolerance)


def test_budgeted_bad_initial_shape_has_a_clear_error():
    with pytest.raises(ValueError, match="initial_vectors.*shape"):
        solve_hermitian(_operator(8), initial_vectors=jnp.ones(8), max_memory=1.)


@pytest.mark.parametrize("reserved", [-1, -.5, True, 1.5])
def test_explicit_solver_reservation_must_be_a_nonnegative_integer(reserved):
    with pytest.raises(ValueError, match="reserved_bytes"):
        solve_hermitian(_operator(8), max_memory=1., reserved_bytes=reserved)


def test_trial_capacity_retains_existing_mandatory_root_minimum(monkeypatch):
    op = _operator(12)
    observed = []
    davidson = primal._davidson_lowest_symmetric

    def record(*args, **kwargs):
        observed.append(kwargs)
        return davidson(*args, **kwargs)

    monkeypatch.setattr(primal, "_davidson_lowest_symmetric", record)
    result = solve_hermitian(op, max_memory=1., config=EigenSolverConfig(
        nroots=2, max_subspace=10, max_trial_vectors=1, maxiter=150, atol=1e-10))
    assert observed[0]["max_trial_vectors"] == 3
    assert observed[0]["nroots"] == 3
    assert bool(jnp.all(result.converged))


@pytest.mark.parametrize("representation", ["list", "tuple", "numpy", "jax"])
def test_dense_input_resident_bytes_include_the_normalized_matrix(representation, monkeypatch):
    from gradscf.solvers.eigen import api

    matrix = np.diag(np.arange(1., 9.))
    inputs = {"list": matrix.tolist(),
              "tuple": tuple(tuple(row) for row in matrix.tolist()),
              "numpy": matrix, "jax": jnp.asarray(matrix)}

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        assert kwargs["resident_bytes"] >= matrix.nbytes + matrix.shape[0]*matrix.dtype.itemsize
        raise Captured

    monkeypatch.setattr(api, "plan_hermitian", capture)
    with pytest.raises(Captured):
        solve_hermitian(inputs[representation], max_memory=1.)
