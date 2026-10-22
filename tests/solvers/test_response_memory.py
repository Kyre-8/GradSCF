"""One MB budget for numerical history and implicit linear-response arrays."""
from dataclasses import asdict, replace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import LinearOperator, LinearSolverConfig, solve_linear
from gradscf.solvers.nonlinear.iterate import NonlinearConfig, solve_nonlinear


def _planner(name):
    function = getattr(importlib.import_module("gradscf.solvers.memory"), name, None)
    assert callable(function), f"Missing shared {name} working-array plan"
    return function


def test_linear_budget_changes_only_optional_restart_and_preserves_original_limit():
    plan = _planner("plan_linear")
    cfg = LinearSolverConfig(restart=40, maxiter=73, rtol=2e-11, atol=3e-13)
    n, resident = 1000, 12345
    _, minimum = plan(n, jnp.float64, replace(cfg, restart=1), 64.)
    _, original = plan(n, jnp.float64, cfg, 64.)
    budget = (resident+minimum+(original-minimum)//3)/.8/1e6
    planned, storage = plan(n, jnp.float64, cfg, budget, resident_bytes=resident)
    assert 1 <= planned.restart < cfg.restart
    before, after = asdict(cfg), asdict(planned)
    before.pop("restart"); after.pop("restart")
    assert before == after
    assert resident+storage <= int(budget*1e6)*.8
    with pytest.raises(ValueError, match="memory|budget|minimum"):
        plan(n, jnp.float64, cfg, (resident+minimum-1000)/.8/1e6,
             resident_bytes=resident)


def test_nonlinear_budget_bounds_ring_without_changing_physical_tolerances():
    plan = _planner("plan_nonlinear")
    cfg = NonlinearConfig(diis_space=8, maxiter=91, damping=.12)
    n, resident = 1300, 4096
    _, minimum = plan(n, jnp.float64, replace(cfg, diis_space=2), 64.)
    _, original = plan(n, jnp.float64, cfg, 64.)
    budget = (resident+minimum+(original-minimum)//3)/.8/1e6
    planned, storage = plan(n, jnp.float64, cfg, budget, resident_bytes=resident)
    assert 2 <= planned.diis_space < cfg.diis_space
    assert replace(planned, diis_space=cfg.diis_space) == cfg
    assert storage+resident <= int(budget*1e6)*.8
    with pytest.raises(ValueError, match="memory|budget|minimum"):
        plan(n, jnp.float64, cfg, (resident+minimum-1000)/.8/1e6,
             resident_bytes=resident)


def test_linear_minimum_fails_before_operator_or_numerical_solver(monkeypatch):
    implicit = importlib.import_module("gradscf.solvers.linear.implicit")

    def forbidden(*args, **kwargs):
        pytest.fail("Linear action/solver ran before minimum-memory rejection")

    monkeypatch.setattr(implicit, "gmres_solve", forbidden)
    op = LinearOperator((20, 20), jnp.float64, forbidden)
    with pytest.raises(ValueError, match="memory|budget|resident|minimum"):
        solve_linear(op, jnp.ones(20), max_memory=.00001)


def test_nonlinear_minimum_fails_before_first_residual():
    def forbidden(*args, **kwargs):
        pytest.fail("Nonlinear residual ran before history-budget rejection")

    with pytest.raises(ValueError, match="memory|budget|resident|minimum"):
        solve_nonlinear(forbidden, jnp.ones(20), update=forbidden,
                        observable=forbidden, max_memory=.00001)


def test_exact_diagonal_budget_does_not_charge_dense_or_krylov_storage():
    n = 128
    d, rhs = jnp.linspace(.7, 2., n), jnp.ones(n)
    cfg = LinearSolverConfig(method="direct", max_dense=1, rtol=1e-12)
    op = LinearOperator((n, n), d.dtype, lambda x: d*x,
                         diagonal=d, exact_diagonal=True)
    out = solve_linear(op, rhs, config=cfg, max_memory=.1)
    assert out.converged
    np.testing.assert_allclose(out.solution, 1/d, atol=2e-13)


@pytest.mark.parametrize("method", ["direct", "gmres"])
def test_budgeted_linear_jvp_vjp_hvp_matches_original(method):
    cfg = LinearSolverConfig(method=method, rtol=1e-12, atol=1e-14)

    def value(t, budget):
        a = jnp.array([[2.+t, .2], [.07, 1.3-.1*t]])
        b = jnp.array([[.3, .7], [.5, -.2]])*(1+.2*t)
        options = {} if budget is None else dict(max_memory=budget, reserved_bytes=1024)
        out = solve_linear(a, b, config=cfg, **options)
        return jnp.sum(out.solution**2)

    def bundle(t, budget):
        function = lambda x: value(x, budget)
        v, g = jax.value_and_grad(function)(t)
        h = jax.jvp(jax.grad(function), (t,), (.2,))[1]
        return v, g, h

    actual = jax.jit(lambda t: bundle(t, 1.))(0.)
    expected = jax.jit(lambda t: bundle(t, None))(0.)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def test_budgeted_nonlinear_root_keeps_implicit_second_response():
    cfg = NonlinearConfig(maxiter=100, residual_tol=1e-11, energy_tol=1e-12)

    def value(t, budget):
        p = jnp.array([2.+t, 3.-.2*t])
        residual = lambda x: x*x-p
        options = {} if budget is None else dict(max_memory=budget, reserved_bytes=1024)
        result = solve_nonlinear(residual, jnp.ones(2), update=lambda x, r: x-.2*r,
            observable=lambda x: jnp.sum(x*x), config=cfg,
            linear_config=LinearSolverConfig(method="direct", rtol=1e-12), **options)
        return jnp.sum(result.solution**3)

    def bundle(t, budget):
        function = lambda x: value(x, budget)
        v, g = jax.value_and_grad(function)(t)
        return v, g, jax.jvp(jax.grad(function), (t,), (.2,))[1]

    actual = jax.jit(lambda t: bundle(t, 1.))(0.)
    expected = jax.jit(lambda t: bundle(t, None))(0.)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-10)


def test_response_reservation_validation():
    for invalid in (-1, True, 2.5):
        with pytest.raises(ValueError, match="reserved_bytes"):
            solve_linear(jnp.eye(2), jnp.ones(2), max_memory=1., reserved_bytes=invalid)
