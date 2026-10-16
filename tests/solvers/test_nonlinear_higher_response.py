"""Numerical DIIS trajectories must stay outside nested implicit derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import LinearSolverConfig
from gradscf.solvers.nonlinear import attach_root, NonlinearConfig, solve_nonlinear


LINEAR = LinearSolverConfig(method="direct", max_dense=4, rtol=1e-11)


def _upstream(parameters):
    target = 1.+jnp.sum(parameters**2)
    seed = jnp.sqrt(jax.lax.stop_gradient(target))[None]
    return attach_root(lambda x: x*x-target, seed, config=LINEAR)[0]


def _result(parameters, *, valid=True):
    upstream = _upstream(parameters)
    target = 2.+upstream+.2*parameters[1]
    return solve_nonlinear(
        lambda x: x*x-target,
        jnp.array([.7+.05*upstream+.03*parameters[0]]),
        update=lambda x, residual: x-(1.+.02*parameters[0])*residual/(2*x),
        observable=lambda x: .5*jnp.sum(x*x)+.1*upstream,
        config=NonlinearConfig(maxiter=80, diis_space=3,
                               residual_tol=1e-11, energy_tol=1e-11),
        linear_config=LINEAR, valid_inputs=valid,
    )


def _value(parameters):
    return _result(parameters).solution[0]


def _exact(parameters):
    return jnp.sqrt(2.+jnp.sqrt(1.+jnp.sum(parameters**2))+.2*parameters[1])


@pytest.mark.parametrize("mode", ["reverse_second", "forward_reverse", "reverse_third"])
def test_nested_nonlinear_root_higher_response_matches_closed_form(mode):
    parameters = jnp.array([.4, -.1])
    result = jax.jit(_result)(parameters)
    assert bool(result.converged)
    assert int(result.iterations) > 0
    assert float(result.residual_norm) < 1e-11
    np.testing.assert_allclose(result.solution[0], _exact(parameters), atol=1e-11, rtol=0)
    np.testing.assert_allclose(jax.jit(jax.grad(_value))(parameters),
                               jax.grad(_exact)(parameters), atol=2e-10, rtol=0)
    if mode == "reverse_second":
        derivative = lambda f: jax.jacrev(jax.grad(f))
    elif mode == "forward_reverse":
        direction = jnp.array([.3, -.7])
        derivative = lambda f: lambda p: jax.jvp(jax.grad(f), (p,), (direction,))[1]
    else:
        derivative = lambda f: jax.jacrev(jax.jacrev(jax.grad(f)))
    np.testing.assert_allclose(jax.jit(derivative(_value))(parameters),
                               derivative(_exact)(parameters), atol=2e-9, rtol=0)


def test_nested_nonlinear_diagnostics_remain_stopped_and_invalid_response_rejected():
    parameters = jnp.array([.4, -.1])
    diagnostic = lambda p: _result(p).residual_norm+_result(p).energy_change
    np.testing.assert_array_equal(jax.jit(jax.grad(diagnostic))(parameters), np.zeros(2))
    failed = _result(parameters, valid=False)
    assert not bool(failed.converged)
    assert int(failed.iterations) == 0
    assert np.isfinite(np.asarray(failed.solution)).all()
    derivative = jax.jit(jax.grad(lambda p: _result(p, valid=False).solution[0]))(parameters)
    assert not np.isfinite(np.asarray(derivative)).all()
