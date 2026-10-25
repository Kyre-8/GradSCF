"""The existing isolated-pair response relays one original memory budget."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import (LinearOperator, LinearSolverConfig,
                            NonHermitianSolverConfig, solve_nonhermitian)


@pytest.mark.parametrize('preplanned', [False, True])
def test_pair_response_receives_original_budget_without_replanning(preplanned, monkeypatch):
    owner = importlib.import_module('gradscf.solvers.eigen.nonhermitian')
    a = jnp.array([[.5, .02, -.01], [0., 1.3, .03], [0., 0., 2.1]])
    calls, original = [], owner.plan_nonhermitian
    def planned(*args, **kwargs):
        assert not preplanned, 'EOM response relay replanned its primal'
        calls.append(kwargs)
        return original(*args, **kwargs)
    class Captured(Exception):
        pass
    def attach(*args, **kwargs):
        assert kwargs['max_memory'] == .5
        assert kwargs['reserved_bytes'] >= 123+a.nbytes
        assert kwargs['action_columns'] == 1 if preplanned else kwargs['action_columns'] >= 2
        raise Captured
    monkeypatch.setattr(owner, 'plan_nonhermitian', planned)
    monkeypatch.setattr(owner, 'attach_pair_response', attach)
    relay = dict(max_memory=.5, reserved_bytes=123+a.nbytes,
                 operator_min_bytes=128, operator_bytes_per_column=256,
                 action_columns=1)
    options = (dict(_batch_size=1, _response_budget=relay) if preplanned else
               dict(max_memory=.5, reserved_bytes=123))
    with pytest.raises(Captured):
        solve_nonhermitian(a, config=NonHermitianSolverConfig(nroots=2, block_size=1),
            vector_response=True, **options)
    assert len(calls) == (0 if preplanned else 1)


def test_pair_response_forwards_known_phase_arrays_to_root_and_checked_solve(monkeypatch):
    owner = importlib.import_module('gradscf.solvers.eigen._nonhermitian_response')
    a = jnp.diag(jnp.array([.5, 1.3, 2.1]))
    vectors = jnp.eye(3)[:, :2]
    cfg = LinearSolverConfig(restart=6, maxiter=41, rtol=1e-11)
    checked, roots = [], []
    def solve(action, rhs, **kwargs):
        assert kwargs['max_memory'] == .5 and kwargs['reserved_bytes'] > 713
        assert kwargs['operator_min_bytes'] >= fixed
        assert kwargs['config'].rtol == cfg.rtol and kwargs['config'].maxiter == 41
        assert kwargs['config'].restart == rhs.shape[0]
        checked.append((rhs.shape[0], kwargs['operator_min_bytes']))
        return rhs, (jnp.asarray(0.), jnp.asarray(True))
    def attach(residual, solution, **kwargs):
        assert kwargs['max_memory'] == .5 and kwargs['reserved_bytes'] > 713
        assert kwargs['operator_min_bytes'] >= fixed
        roots.append(solution.shape[0])
        _, action = jax.linearize(residual, solution)
        answer = kwargs['tangent_solve'](action, jnp.zeros_like(solution))
        assert answer.shape == solution.shape
        return solution
    monkeypatch.setattr(owner, 'checked_linear_solve', solve)
    monkeypatch.setattr(owner, 'attach_root', attach)
    op = LinearOperator(a.shape, a.dtype, lambda value: a@value)
    for fixed in (0, 1024):
        owner.attach_pair_response(op, jnp.array([.5, 1.3]), vectors, vectors,
            config=cfg, valid=jnp.asarray(True), max_memory=.5, reserved_bytes=713,
            operator_min_bytes=fixed, operator_bytes_per_column=128, action_columns=1)
    assert roots == [4, 3, 4, 3] and [row for row, _ in checked] == roots
    # An independent fixed workspace survives even when the larger declared
    # array sum dominates the per-column estimate.
    assert [checked[i+2][1]-checked[i][1] for i in (0, 1)] == [1024, 1024]


def test_response_minimum_rejects_before_attaching_a_root(monkeypatch):
    owner = importlib.import_module('gradscf.solvers.eigen._nonhermitian_response')
    a = jnp.diag(jnp.array([.5, 1.3, 2.1]))
    vectors = jnp.eye(3)[:, :2]
    def forbidden(*args, **kwargs):
        pytest.fail('Impossible pair-response budget reached custom_root')
    monkeypatch.setattr(owner, 'attach_root', forbidden)
    primal = solve_nonhermitian(a, config=NonHermitianSolverConfig(nroots=2),
                               max_memory=.01, reserved_bytes=37)
    assert jnp.all(primal.converged)
    with pytest.raises(ValueError, match='memory|budget|minimum'):
        owner.attach_pair_response(LinearOperator(a.shape, a.dtype, lambda v: a@v),
            jnp.array([.5, 1.3]), vectors, vectors, config=LinearSolverConfig(),
            valid=jnp.asarray(True), max_memory=.01, reserved_bytes=37,
            operator_min_bytes=10000, operator_bytes_per_column=0, action_columns=1)


@pytest.mark.parametrize('method', ['direct', 'gmres'])
def test_budgeted_pairs_preserve_joint_jvp_vjp_hvp_and_finite_difference(method):
    from test_nonhermitian_vectors import observable, solved
    weights = jnp.array([.2, -.3, .7, -.5])
    value = lambda budget: lambda t: weights@observable(t, method=method, max_memory=budget)
    def bundle(function):
        return jax.jit(lambda t: jax.jvp(jax.value_and_grad(function),
            (t,), (jnp.ones_like(t),)))(jnp.asarray(.1))
    actual, expected = bundle(value(4.)), bundle(value(None))
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, atol=3e-10, rtol=3e-9)
    gradient = jax.jit(jax.grad(value(4.)))
    for step in (3e-4, 1e-4):
        difference = (gradient(.1+step)-gradient(.1-step))/(2*step)
        np.testing.assert_allclose(actual[1][1], difference, atol=2e-8, rtol=2e-7)
    frozen_vectors = jax.grad(lambda t: weights[:2]@solved(
        t, method=method, max_memory=4.).values)(.1)
    assert abs(actual[0][1]-frozen_vectors) > 1e-5
