"""Stable special functions used by the unchanged upstream screened kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft.xc._compat import scaled_erfcx, scaled_exp1

special = pytest.importorskip("scipy.special")


def test_scaled_exp1_batched_small_and_large_arguments_and_jvp():
    x = np.logspace(-30, 3, 40)
    actual, derivative = jax.jit(jax.vmap(jax.value_and_grad(scaled_exp1)))(jnp.asarray(x))
    expected = np.empty_like(x)
    modest = x <= 50
    expected[modest] = np.exp(x[modest])*special.exp1(x[modest])
    # scipy's integral representation is stable under Gauss-Laguerre for x>=50.
    nodes, weights = np.polynomial.laguerre.laggauss(64)
    expected[~modest] = np.sum(weights[None, :]/(x[~modest, None]+nodes[None, :]), axis=1)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(derivative, expected-1/x, rtol=2e-10, atol=2e-15)
    second = jax.jit(jax.vmap(jax.grad(jax.grad(scaled_exp1))))(jnp.asarray(x))
    assert np.all(np.isfinite(second))


def test_erfcx_avoids_inactive_wpbe_exp868_overflow():
    x = np.asarray([0., .1, 1., 10., 20., 29.46, 100., 1e4])
    actual, derivative = jax.jit(jax.vmap(jax.value_and_grad(scaled_erfcx)))(jnp.asarray(x))
    expected = special.erfcx(x)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(derivative, 2*x*expected-2/np.sqrt(np.pi), rtol=2e-7, atol=2e-15)
    second = jax.jit(jax.vmap(jax.grad(jax.grad(scaled_erfcx))))(jnp.asarray(x))
    assert np.all(np.isfinite(second))


def test_large_scaled_exp1_derivative_does_not_cancel_to_zero():
    x = np.asarray([1e6, 1e12, 1e20])
    derivative = jax.jit(jax.vmap(jax.grad(scaled_exp1)))(jnp.asarray(x))
    nodes, weights = np.polynomial.laguerre.laggauss(64)
    expected = -np.sum(weights[None, :]/(x[:, None]+nodes[None, :])**2, axis=1)
    np.testing.assert_allclose(derivative, expected, rtol=2e-13, atol=0.)
