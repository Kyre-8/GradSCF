"""Checked tensor inverses retain primal values under nested batched AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import solve_tensor_sum


def _inputs(shift, field):
    a = jnp.array([[2., .1], [.1, 3.]]) + jnp.array([
        [field[0] + .1*shift, field[1]], [field[1], field[2]]])
    b = jnp.array([[.7, .05], [.05, 1.1]]) + .1*jnp.array([
        [field[1], field[2]], [field[2], field[0] + shift]])
    rhs = jnp.array([[1., .2], [.3, .7]]) + .01*jnp.sum(field**2) + .1*shift
    return a, b, rhs


def _dense_energy(shift, field):
    a, b, rhs = _inputs(shift, field)
    matrix = jnp.kron(a, jnp.eye(2)) + jnp.kron(jnp.eye(2), b)
    return jnp.vdot(rhs.reshape(-1), jnp.linalg.solve(matrix, rhs.reshape(-1)))


def _mixed_response(energy, shift, field):
    dipole = lambda s, f: -jax.grad(energy, argnums=1)(s, f)
    alpha = lambda s: jax.jacfwd(dipole, argnums=1)(s, field)
    return jax.jvp(alpha, (shift,), (jnp.ones_like(shift),))[1]


@pytest.mark.parametrize('diagnostic', ['converged', 'residual_norm'])
def test_checked_diagnostic_mask_retains_nested_batched_response(diagnostic):
    def energy(shift, field):
        a, b, rhs = _inputs(shift, field)
        result = solve_tensor_sum((a, b), rhs)
        accepted = (result.converged if diagnostic == 'converged'
                    else result.residual_norm < 1e-10)
        return jnp.sum(rhs*result.solution)*jnp.where(accepted, 1., jnp.nan)

    shift, field = jnp.asarray(.13), jnp.array([.02, -.01, .03])
    actual = jax.jit(lambda s, f: _mixed_response(energy, s, f))(shift, field)
    expected = jax.jit(lambda s, f: _mixed_response(_dense_energy, s, f))(shift, field)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)
    np.testing.assert_allclose(actual, actual.T, atol=1e-13, rtol=0)
    alpha = jax.jit(lambda s: -jax.hessian(energy, argnums=1)(s, field))
    for step in (3e-4, 1e-4):
        finite = (alpha(shift+step)-alpha(shift-step))/(2*step)
        np.testing.assert_allclose(actual, finite, atol=2e-8, rtol=2e-6)


def test_singular_tensor_sum_remains_invalid_in_nested_response():
    def energy(shift, field):
        rhs = jnp.ones((2, 2))*jnp.exp(shift+jnp.sum(field**2))
        result = solve_tensor_sum((jnp.eye(2), -jnp.eye(2)), rhs)
        return jnp.sum(result.solution)*jnp.where(result.converged, 1., jnp.nan)

    shift, field = jnp.asarray(.13), jnp.array([.02, -.01, .03])
    result = solve_tensor_sum((jnp.eye(2), -jnp.eye(2)), jnp.ones((2, 2)))
    assert not result.converged
    assert not np.isfinite(np.asarray(result.solution)).any()
    assert not np.isfinite(float(energy(shift, field)))
    gradient = jax.jit(jax.grad(energy, argnums=1))(shift, field)
    nested = jax.jit(lambda s, f: _mixed_response(energy, s, f))(shift, field)
    assert not np.isfinite(np.asarray(gradient)).any()
    assert not np.isfinite(np.asarray(nested)).any()
