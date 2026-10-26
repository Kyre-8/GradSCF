"""Mixed parameter/geometry AD and second-coordinate derivative composition.

H2/STO-3G, CPU float64; geometry variables are in bohr. Model parameters
change integral weights; geometry probes also exercise coordinate Hessians.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals


def _geometry_value(operator):
    top, params = integrals.prepare_basis("H 0 0 0; H .2 .1 .8", basis="sto-3g")
    plan = integrals.make_plan(top, backend="native")
    coords = jnp.concatenate([params.nuclear_coords, params.centers])

    def value(r):
        return plan.evaluate(operator, replace(params, nuclear_coords=r[:2], centers=r[2:]))

    return value, coords


@pytest.mark.parametrize("operator", ["overlap", "kinetic", "nuclear", "eri"])
def test_native_nonlinear_force_loss_parameter_gradient(operator):
    value, coords = _geometry_value(operator)
    theta = jnp.array([.31, -.23])

    def energy(t, r):
        x = value(r)
        return jnp.sum(jnp.sin(t[0]*x) + t[1]*jnp.square(x))

    force = jax.jit(lambda t, r: -jax.grad(energy, argnums=1)(t, r))
    target = jnp.linspace(-.04, .07, coords.size).reshape(coords.shape)
    loss = jax.jit(lambda t: jnp.sum(jnp.square(force(t, coords)-target)))
    actual = jax.jit(jax.grad(loss))(theta)
    h = 1e-5
    fd = jnp.stack([(loss(theta+h*d)-loss(theta-h*d))/(2*h) for d in jnp.eye(2)])
    assert float(jnp.linalg.norm(actual)) > 1e-6
    np.testing.assert_allclose(actual, fd, atol=2e-7, rtol=2e-7)
    # Batching the higher-order derivative exercises both native linear maps.
    batched = jax.jit(jax.vmap(jax.grad(loss)))(jnp.stack([theta, theta+.03]))
    np.testing.assert_allclose(batched[0], actual, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("operator", ["overlap", "kinetic", "nuclear", "eri"])
def test_native_derivative_maps_vector_ad_and_adjoint(operator):
    value, coords = _geometry_value(operator)
    primal, pullback = jax.vjp(value, coords)
    cot = jnp.linspace(-.3, .7, primal.size).reshape(primal.shape)
    dcot = jnp.cos(jnp.arange(primal.size)).reshape(primal.shape)
    direction = jnp.linspace(-.2, .4, coords.size).reshape(coords.shape)
    push = jax.jit(lambda d: jax.jvp(value, (coords,), (d,))[1])
    pull = jax.jit(lambda c: pullback(c)[0])

    _, actual = jax.jit(lambda c, dc: jax.jvp(pull, (c,), (dc,)))(cot, dcot)
    np.testing.assert_allclose(actual, pull(dcot), atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(jnp.vdot(pull(cot), direction),
                               jnp.vdot(cot, push(direction)), atol=2e-12, rtol=2e-12)
    transposed_pull = jax.jit(jax.grad(lambda c: jnp.vdot(pull(c), direction)))(cot)
    np.testing.assert_allclose(transposed_pull, push(direction), atol=2e-12, rtol=2e-12)
    _, pushed = jax.jit(lambda d: jax.jvp(push, (d,), (2*d,)))(direction)
    np.testing.assert_allclose(pushed, 2*push(direction), atol=2e-12, rtol=2e-12)
    stacked = jnp.stack([cot, dcot])
    batched = jax.jit(jax.vmap(pull))(stacked)
    np.testing.assert_allclose(batched, jnp.stack([pull(cot), pull(dcot)]), atol=2e-12)


@pytest.mark.parametrize("zero_direction", [False, True])
def test_native_second_geometry_derivative_including_zero_direction(zero_direction):
    value, coords = _geometry_value("overlap")
    force = jax.grad(lambda r: jnp.sum(value(r)))
    direction = jnp.zeros_like(coords) if zero_direction else jnp.arange(coords.size).reshape(coords.shape)*.1
    actual = jax.jvp(force, (coords,), (direction,))[1]
    hessian = jax.jit(jax.jacrev(force))(coords).reshape(coords.size,coords.size)
    np.testing.assert_allclose(actual.ravel(),hessian@direction.ravel(),atol=2e-12)
    if zero_direction:
        np.testing.assert_array_equal(actual,0.)
