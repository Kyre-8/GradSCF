"""DF contraction derivatives and auxiliary-block reverse memory boundaries."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.molecular.density_fitting import project_factors


def inputs():
    rng = np.random.default_rng(20261008)
    primitive, contracted, rank = 8, 3, 19
    packed = jnp.asarray(rng.normal(size=(rank, primitive * (primitive + 1) // 2)))
    transform = rng.normal(size=(primitive, contracted))
    transform[0] = 0.; transform[1, 2] = 0.
    i, j = np.tril_indices(primitive)
    symmetric = np.zeros((len(i), primitive, primitive))
    symmetric[np.arange(len(i)), i, j] = 1.
    symmetric[np.arange(len(i)), j, i] = 1.
    def dense_reference(factors, matrix):
        dense = jnp.einsum('aP,Ppq->apq', factors, jnp.asarray(symmetric))
        return jnp.einsum('pi,apq,qj->aij', matrix, dense, matrix,
                         precision=jax.lax.Precision.HIGHEST)
    return packed, jnp.asarray(transform), dense_reference, rng


@pytest.mark.parametrize('block_size', [1, 4, 64])
def test_projection_forward_and_vjp_match_dense_at_zero_coefficients(block_size):
    factors, transform, reference, rng = inputs()
    project = lambda f, t: project_factors(f, t, block_size=block_size)
    expected = reference(factors, transform)
    actual, backward = jax.vjp(project, factors, transform)
    _, dense_backward = jax.vjp(reference, factors, transform)
    cotangent = jnp.asarray(rng.normal(size=expected.shape))
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    for derivative, correct in zip(backward(cotangent), dense_backward(cotangent)):
        assert np.isfinite(derivative).all()
        np.testing.assert_allclose(derivative, correct, atol=3e-11, rtol=3e-12)


def test_projection_jvp_and_hvp_match_dense_reference():
    factors, transform, reference, rng = inputs()
    directions = tuple(jnp.asarray(rng.normal(size=x.shape)) for x in (factors, transform))
    project = lambda f, t: project_factors(f, t, block_size=4)
    actual_jvp = jax.jvp(project, (factors, transform), directions)[1]
    expected_jvp = jax.jvp(reference, (factors, transform), directions)[1]
    np.testing.assert_allclose(actual_jvp, expected_jvp, atol=3e-11, rtol=3e-12)
    def objective(function):
        return lambda f, t: jnp.sum(jnp.sin(function(f, t)) + .07 * function(f, t)**2)
    actual = jax.jvp(jax.grad(objective(project), argnums=(0, 1)),
                     (factors, transform), directions)[1]
    expected = jax.jvp(jax.grad(objective(reference), argnums=(0, 1)),
                       (factors, transform), directions)[1]
    for derivative, correct in zip(actual, expected):
        assert np.isfinite(derivative).all()
        np.testing.assert_allclose(derivative, correct, atol=3e-9, rtol=2e-11)


def test_projection_reverse_retains_at_most_one_unpacked_auxiliary_block():
    # Abstract tracing allocates no factors. The old reverse saved every dense
    # block as a [number_of_blocks, block_size, primitive, primitive] residual.
    rank, primitive, contracted, block_size = 257, 64, 4, 16
    factors = jax.ShapeDtypeStruct((rank, primitive * (primitive + 1) // 2), jnp.float64)
    transform = jax.ShapeDtypeStruct((primitive, contracted), jnp.float64)
    objective = lambda t, f: jnp.sum(project_factors(f, t, block_size=block_size)**2)
    graph = jax.make_jaxpr(jax.grad(objective, argnums=0))(transform, factors).jaxpr
    saved_bytes = [math.prod(out.aval.shape) * out.aval.dtype.itemsize
        for equation in graph.eqns if equation.primitive.name == 'scan'
        for out in equation.outvars if getattr(out.aval, 'shape', ())]
    bound = 2 * block_size * primitive**2 * 8 + 4 * rank * contracted**2 * 8
    assert max(saved_bytes, default=0) <= bound, (
        'Reverse projection stores dense intermediates across all auxiliary blocks; '
        f'largest scan residual is {max(saved_bytes)} bytes, allowed {bound}.')
