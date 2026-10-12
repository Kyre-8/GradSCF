"""Bounded native VJP threads preserve CPU float64 derivatives."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals


def _gradient(cart=False, operator="overlap", coeff_only=False):
    atom = "H 0 0 0; H .2 .1 .8"
    top, p = integrals.prepare_basis(atom, {"H": [
        [0, [1.1, .8, .15], [.3, -.2, .9]],
        [1, [.9, .6], [.25, .2]], [2, [.7, .7], [.2, -.1]],
    ]}, cart=cart)
    if operator == "df":
        at, ap = integrals.prepare_basis(atom, "sto-3g", cart=cart)
        ri = integrals.make_auxiliary_plan(top, at)
        metric = ri.metric_factor(p, ap)
        value = lambda x: ri.factors(x, ap, metric_factor=metric)
    else:
        plan = integrals.make_plan(top)
        value = lambda x: plan.evaluate(operator, x)
    cot = jnp.asarray(np.random.default_rng(48).normal(size=value(p).shape))
    def loss(a, c):
        return jnp.sum(cot * value(replace(p, exponents=a, coefficients=c)))
    return jax.jit(jax.grad(loss, argnums=1 if coeff_only else (0, 1))), (p.exponents, p.coefficients)


@pytest.mark.parametrize("setting", ["0", "-1", "1x"])
def test_invalid_native_thread_limit_is_rejected(monkeypatch, setting):
    grad, args = _gradient()
    monkeypatch.setenv("GRADSCF_NATIVE_THREADS", setting)
    with pytest.raises(Exception, match="GRADSCF_NATIVE_THREADS"):
        jax.block_until_ready(grad(*args))


@pytest.mark.parametrize("cart", [False, True])
@pytest.mark.parametrize("operator", ["overlap", "kinetic", "nuclear", "dipole", "df"])
def test_parallel_joint_vjp_matches_serial(monkeypatch, cart, operator):
    grad, args = _gradient(cart, operator)
    monkeypatch.setenv("GRADSCF_NATIVE_THREADS", "1")
    serial = jax.block_until_ready(grad(*args))
    for threads in (2, 4, 32):
        monkeypatch.setenv("GRADSCF_NATIVE_THREADS", str(threads))
        actual = jax.block_until_ready(grad(*args))
        for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(serial)):
            np.testing.assert_allclose(x, y, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("cart", [False, True])
@pytest.mark.parametrize("operator", ["overlap", "df"])
def test_parallel_coefficient_hessian_matches_serial(monkeypatch, cart, operator):
    grad, (a, c) = _gradient(cart, operator, coeff_only=True)
    direction = jax.tree.map(lambda x: .07*jnp.cos(x), c)
    hvp = jax.jit(lambda c, d: jax.jvp(lambda x: grad(a, x), (c,), (d,))[1])
    monkeypatch.setenv("GRADSCF_NATIVE_THREADS", "1")
    serial = jax.block_until_ready(hvp(c, direction))
    for threads in (2, 4):
        monkeypatch.setenv("GRADSCF_NATIVE_THREADS", str(threads))
        actual = jax.block_until_ready(hvp(c, direction))
        for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(serial)):
            np.testing.assert_allclose(x, y, atol=2e-11, rtol=2e-11)
