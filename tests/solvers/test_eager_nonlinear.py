"""Shared eager fixed-point iteration: true residuals and bounded DIIS."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers.nonlinear import NonlinearConfig


def _iterate():
    return importlib.import_module('gradscf.solvers.nonlinear.eager').iterate_eager


def test_eager_diis_converges_stiff_linear_map_to_independent_root():
    iterate = _iterate()
    matrix = jnp.array([[-1.2, .25], [.1, .85]])
    constant = jnp.array([.3, -.2])
    physical = lambda value: matrix@value+constant
    expected = np.linalg.solve(np.eye(2)-np.asarray(matrix), np.asarray(constant))
    initial = jnp.zeros(2)
    accelerated = iterate(physical, initial, config=NonlinearConfig(
        maxiter=8, diis_space=4, diis_start=2, damping=.1,
        residual_tol=1e-10, energy_tol=1e-10))
    plain = iterate(physical, initial, config=NonlinearConfig(
        maxiter=8, diis_space=1, damping=.1, residual_tol=1e-10, energy_tol=1e-10))
    assert bool(accelerated.converged)
    assert int(accelerated.iterations) <= 8
    np.testing.assert_allclose(accelerated.solution, expected, atol=2e-9, rtol=0)
    actual = np.max(abs(np.asarray(physical(accelerated.solution)-accelerated.solution)))
    np.testing.assert_allclose(accelerated.residual_norm, actual, atol=2e-15, rtol=0)
    assert actual < 1e-10
    assert not bool(plain.converged)


def test_eager_small_mixed_step_cannot_certify_a_false_fixed_point():
    iterate = _iterate()
    physical = lambda value: value+1.
    result = iterate(physical, jnp.zeros(2), config=NonlinearConfig(
        maxiter=3, diis_space=1, damping=.999999,
        residual_tol=1e-6, energy_tol=1e-3))
    assert not bool(result.converged)
    np.testing.assert_allclose(result.residual_norm, 1., atol=2e-15, rtol=0)
    np.testing.assert_allclose(physical(result.solution)-result.solution, 1., atol=2e-15, rtol=0)


def test_eager_domain_guard_halves_plain_map_before_expensive_evaluation():
    iterate = _iterate()
    evaluated = []
    def physical(value):
        assert np.min(np.asarray(value)) > 0
        evaluated.append(np.asarray(value))
        return .2-3*value
    result = iterate(physical, jnp.array([.1]),
        config=NonlinearConfig(maxiter=8, diis_space=3, residual_tol=1e-11, energy_tol=1e-11),
        valid_state=lambda value: bool(jnp.all(value > 0)))
    assert bool(result.converged)
    np.testing.assert_allclose(result.solution, [.05], atol=2e-12, rtol=0)
    assert len(evaluated) >= 2


def test_eager_iteration_explicitly_rejects_traced_forward_iterations():
    iterate = _iterate()
    with pytest.raises(NotImplementedError, match='eager|AD|differentiat'):
        jax.jit(lambda value: iterate(lambda x: .5*x, value).solution)(jnp.ones(2))


@pytest.mark.parametrize("transform", [jax.jit, jax.grad], ids=["jit", "grad"])
def test_eager_rejects_transforms_even_when_initial_values_are_closed_constants(transform):
    iterate = _iterate()
    initial = jnp.ones(2)

    def forbidden(value):
        pytest.fail("Physical map ran inside an unsupported transformation")

    def function(parameter):
        return iterate(forbidden, initial).solution[0]+parameter

    with pytest.raises(NotImplementedError, match="eager|AD|differentiat"):
        transform(function)(1.)


def test_eager_guard_keeps_legacy_public_jax_core_support(monkeypatch):
    # Older supported JAX releases export this through public jax.core.
    monkeypatch.setattr(jax.core, "trace_state_clean", lambda: False, raising=False)

    def forbidden(value):
        pytest.fail("Legacy public trace guard was ignored")

    with pytest.raises(NotImplementedError, match="eager|AD|differentiat"):
        _iterate()(forbidden, jnp.ones(2))


def test_eager_additional_check_can_veto_but_cannot_replace_true_residual():
    iterate = _iterate()
    config = NonlinearConfig(maxiter=3, diis_space=2, residual_tol=1e-10, energy_tol=1e-10)
    vetoed = iterate(lambda value: jnp.zeros_like(value), jnp.ones(2), config=config,
                     check=lambda state, mapped, norm: False)
    assert not bool(vetoed.converged)
    assert float(vetoed.residual_norm) < config.residual_tol
    false_success = iterate(lambda value: value+1., jnp.zeros(2), config=config,
                            check=lambda state, mapped, norm: True)
    assert not bool(false_success.converged)
    np.testing.assert_allclose(false_success.residual_norm, 1., atol=2e-15, rtol=0)
