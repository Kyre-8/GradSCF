"""Variational values retain higher response while omitting E_z dz at a root."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_stationary_value_retains_state_response_at_higher_order(order):
    from gradscf.solvers import LinearSolverConfig
    from gradscf.solvers.nonlinear import attach_root, stationary_value

    def energy(z, p):
        k, b, c = jnp.exp(p), 1+.3*p+p*p, jnp.sin(p)
        return .5*k*z[0]**2-b*z[0]+c

    def value(p):
        seed = jnp.asarray([(1+.3*p+p*p)/jnp.exp(p)])
        state = attach_root(lambda z: jax.grad(energy)(z, p), seed,
                            config=LinearSolverConfig(method="direct", rtol=1e-12))
        return stationary_value(energy, state, p)

    expected = lambda p: jnp.sin(p)-.5*(1+.3*p+p*p)**2/jnp.exp(p)
    actual = value
    for _ in range(order):
        actual, expected = jax.grad(actual), jax.grad(expected)
    np.testing.assert_allclose(jax.jit(actual)(.17), expected(.17), atol=3e-11, rtol=1e-10)


def test_first_energy_derivative_does_not_need_an_undefined_flat_state_response():
    from gradscf.solvers import LinearSolverConfig
    from gradscf.solvers.nonlinear import attach_root, stationary_value

    energy = lambda z, p: .5*z[0]**2-p*z[0]

    def state(p):
        return attach_root(lambda z: jax.grad(energy)(z, p), jnp.array([p, 0.]),
                            config=LinearSolverConfig(method="direct", rtol=1e-12))

    value = lambda p: stationary_value(energy, state(p), p)
    np.testing.assert_allclose(jax.jit(jax.value_and_grad(value))(.7), [-.245, -.7], atol=1e-12)
    np.testing.assert_allclose(jax.jit(lambda p: jax.jvp(value, (p,), (1.,))[1])(.7), -.7, atol=1e-12)
    # A singular direct state solve remains invalid; no density/gauge response
    # has been assigned by the variational-value rule.
    assert not np.isfinite(np.asarray(jax.jvp(state, (.7,), (1.,))[1])).all()


def test_nonconverged_stationary_value_rejects_first_response():
    from gradscf.solvers.nonlinear import stationary_value

    value = lambda p: stationary_value(lambda z, q: (z[0]-q)**2+q,
                                        jnp.array([0.]), p, converged=False)
    assert np.isfinite(float(value(.2)))
    assert not np.isfinite(float(jax.jit(jax.grad(value))(.2)))
