"""Known primal memory and bounded two-sided non-Hermitian actions."""
from dataclasses import replace
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import (LinearOperator, LinearSolverConfig,
                            NonHermitianSolverConfig, solve_nonhermitian)
from gradscf.solvers import memory


def _plan(*args, **kwargs):
    function = getattr(memory, 'plan_nonhermitian', None)
    assert callable(function), 'Missing shared non-Hermitian primal memory planner'
    return function(*args, **kwargs)


def _solve(*args, **kwargs):
    assert 'max_memory' in inspect.signature(solve_nonhermitian).parameters, (
        'Missing non-Hermitian solver memory-budget entry')
    return solve_nonhermitian(*args, **kwargs)


def _operator(n=32, calls=None):
    d = jnp.linspace(.4, 4., n)
    a = jnp.diag(d)+jnp.diag(jnp.full(n-1, .015), 1)
    a += jnp.diag(jnp.full(n-1, .004), -1)

    def apply(side, matrix, value):
        if calls is not None:
            calls[side].append(1 if value.ndim == 1 else value.shape[1])
        return matrix @ value

    right = lambda v: apply('right', a, v)
    left = lambda v: apply('left', a.T, v)

    class TwoSided(LinearOperator):
        @property
        def T(self):
            return LinearOperator(self.shape[::-1], self.dtype, left,
                                  diagonal=self.diagonal, matmat=left, transpose_matvec=right)

    return TwoSided(a.shape, a.dtype, right, diagonal=jnp.diag(a), matmat=right,
                    transpose_matvec=left), a


@pytest.mark.parametrize('method', ['dense', 'davidson'])
def test_impossible_budget_rejects_before_guesses_identity_or_action(method, monkeypatch):
    op, _ = _operator()

    def forbidden(*args, **kwargs):
        pytest.fail('Non-Hermitian solver allocated work before checking its budget')

    monkeypatch.setattr(jax.random, 'normal', forbidden)
    monkeypatch.setattr(jax.nn, 'one_hot', forbidden)
    monkeypatch.setattr(jnp, 'eye', forbidden)
    with pytest.raises(ValueError, match='memory|budget|minimum'):
        _solve(op, config=NonHermitianSolverConfig(method=method), max_memory=.0001)


def test_planner_clips_optional_krylov_preserving_root_guard_restart_and_user_config():
    n, dtype = 64, jnp.float64
    cfg = NonHermitianSolverConfig(method='davidson', nroots=2, guard_roots=1,
                                    max_space=40, maxiter=173)
    minimum = 4*(cfg.nroots+cfg.guard_roots)+2
    _, minimum_bytes, _ = _plan(n, dtype, replace(cfg, max_space=minimum), 2.)
    # Retain the same original MB budget and apply the shared 20% reserve once.
    resident, action_min, per_column = 1234, 512, 256
    required = minimum_bytes+action_min+(2*n*8+per_column)*cfg.nroots
    limit = (resident+required+64)*1.25/1e6
    planned, reserved, columns = _plan(n, dtype, cfg, limit, resident_bytes=resident,
        operator_min_bytes=action_min, operator_bytes_per_column=per_column)
    assert minimum <= planned.max_space < cfg.max_space
    assert cfg.max_space == 40
    for name in ('nroots', 'guard_roots', 'maxiter', 'atol', 'seed'):
        assert getattr(planned, name) == getattr(cfg, name)
    assert reserved+action_min+(2*n*8+per_column)*columns <= int(limit*1e6)*.8-resident+1
    with pytest.raises(ValueError, match='minimum|budget|memory'):
        _plan(n, dtype, cfg, .00001, resident_bytes=resident)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_dense_plan_charges_complex_eigenspaces_and_real_cluster_repair(dtype):
    n = 8
    cfg = NonHermitianSolverConfig(nroots=2)
    planned, reserved, columns = _plan(n, dtype, cfg, 1.)
    real = jnp.dtype(dtype).itemsize
    complex_ = jnp.dtype(jnp.result_type(dtype, jnp.complex64)).itemsize
    # Two complex eigenspaces and the real U/V/overlap of cluster SVD coexist.
    assert reserved >= 2*n*n*complex_+3*n*n*real
    assert planned.nroots == 2 and 2 <= columns <= n


def test_iterative_legal_minimum_and_explicit_guesses_are_not_silently_changed():
    cfg = NonHermitianSolverConfig(method='davidson', nroots=2, max_space=4)
    with pytest.raises(ValueError, match='max_space'):
        _plan(40, jnp.float64, cfg, 2.)
    cfg = replace(cfg, max_space=20)
    answer, _, _ = _plan(40, jnp.float64, cfg, 2., initial_columns=18)
    assert answer.max_space >= 18
    with pytest.raises(ValueError, match='initial|space'):
        _plan(40, jnp.float64, cfg, 2., initial_columns=21)


@pytest.mark.parametrize('method', ['dense', 'davidson'])
def test_bounded_right_left_action_batches_match_unbudgeted_pairs(method):
    n = 12 if method == 'dense' else 36
    calls = {'right': [], 'left': []}
    op, a = _operator(n, calls)
    cfg = NonHermitianSolverConfig(method=method, max_space=18, maxiter=180,
                                    block_size=n, atol=1e-10)
    required_space = min(n, 4*(cfg.nroots+cfg.guard_roots)+2)
    _, reservation, _ = _plan(n, a.dtype, replace(cfg, max_space=required_space), 2.)
    resident = a.nbytes+op.diagonal.nbytes
    limit = (resident+reservation+2*n*a.dtype.itemsize*cfg.nroots+64)*1.25/1e6
    planned, _, batch = _plan(n, a.dtype, cfg, limit, resident_bytes=resident)
    actual = _solve(op, config=cfg, max_memory=limit, reserved_bytes=a.nbytes)
    assert np.all(actual.converged & actual.response_valid)
    assert calls['right'] and calls['left']
    assert max(calls['right']) <= batch and max(calls['left']) <= batch
    assert actual.subspace_dimension <= (n if method == 'dense' else planned.max_space)
    expected = solve_nonhermitian(a, config=cfg)
    np.testing.assert_allclose(actual.values, expected.values, atol=3e-10, rtol=3e-10)
    np.testing.assert_allclose(actual.left_vectors.T@actual.right_vectors, jnp.eye(1), atol=2e-10)


@pytest.mark.parametrize('method', ['dense', 'davidson'])
def test_budgeted_values_and_both_vector_jvp_vjp_keep_response_restart(method, monkeypatch):
    from gradscf.solvers.eigen import nonhermitian
    a = jnp.array([[.5, .12, -.03], [.01, 1.3, .05], [.02, -.01, 2.1]])
    b = jnp.array([[.03, .01, -.02], [.04, -.03, .01], [.02, .03, .01]])
    cfg = NonHermitianSolverConfig(method=method, nroots=2, atol=1e-11)
    response = LinearSolverConfig(restart=7, maxiter=40, rtol=1e-11)
    attach, seen = nonhermitian.attach_pair_response, []

    def record(*args, **kwargs):
        seen.append(kwargs['config'])
        return attach(*args, **kwargs)

    monkeypatch.setattr(nonhermitian, 'attach_pair_response', record)

    def value(x, budgeted):
        options = dict(config=cfg, vector_response=True, response_config=response)
        out = (_solve(a+x*b, max_memory=.5, **options) if budgeted else
               solve_nonhermitian(a+x*b, **options))
        weights = jnp.array([[1.], [2.], [3.]])
        return jnp.sum(out.values)+.07*jnp.sum(weights*out.right_vectors*out.left_vectors)

    actual, expected = lambda x: value(x, True), lambda x: value(x, False)
    for operation in (lambda f: jax.jvp(f, (0.,), (1.,)),
                      lambda f: jax.value_and_grad(f)(0.)):
        for a, e in zip(operation(actual), operation(expected)):
            np.testing.assert_allclose(a, e, atol=3e-10, rtol=3e-9)
    assert seen and all(x is response and x.restart == 7 and x.maxiter == 40 for x in seen)


@pytest.mark.parametrize('method', ['dense', 'davidson'])
def test_budget_preserves_real_cluster_dual_and_invalid_response(method):
    a = jnp.array([[.5, 0., .2], [0., .5, .1], [0., 0., 2.]])
    cfg = NonHermitianSolverConfig(method=method, nroots=2, max_space=12)
    actual = _solve(a, config=cfg, max_memory=.5)
    expected = solve_nonhermitian(a, config=cfg)
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual.left_vectors.T@actual.right_vectors, jnp.eye(2), atol=1e-11)
    assert np.all(actual.converged) and not np.any(actual.response_valid)
    derivative = jax.grad(lambda x: _solve(x*a, config=cfg, max_memory=.5).values.sum())(1.)
    assert not np.isfinite(derivative)
