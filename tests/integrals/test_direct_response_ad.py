"""Zero materialized basis tangents are valid; actual basis AD is rejected."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals


@pytest.fixture(scope="module")
def system():
    top, parameters = integrals.prepare_basis("H 0 0 0; H 0 0 1.4", "6-31g", cart=False)
    plan = integrals.make_plan(top)
    eri = plan.evaluate("eri", parameters)
    density = jnp.asarray(np.random.default_rng(907).normal(size=(2, top.nao, top.nao)))
    return plan, parameters, eri, density


def test_materialized_zero_basis_tangent_with_jit_vmap_and_density_hvp(system):
    plan, parameters, eri, density = system
    zero = jax.tree.map(jnp.zeros_like, parameters)
    direction = jnp.ones_like(density)

    def apply(d):
        return jax.jvp(lambda p, d: plan.get_jk(p, d),
                       (parameters, d), (zero, direction))[1]

    expected = (jnp.einsum("pqrs,brs->bpq", eri, direction),
                jnp.einsum("prqs,brs->bpq", eri, direction))
    for actual, reference in zip(jax.jit(apply)(density), expected):
        np.testing.assert_allclose(actual, reference, atol=2e-11)
    batched = jax.jit(jax.vmap(apply))(jnp.stack((density, 2 * density)))
    for actual, reference in zip(batched, expected):
        np.testing.assert_allclose(actual, jnp.stack((reference, reference)), atol=2e-11)

    def loss(t, d):
        value, tangent = jax.jvp(lambda p, x: plan.get_jk(p, x),
                                (parameters, d), (zero, direction))
        return sum(jnp.sum((v + t * dv)**2) for v, dv in zip(value, tangent))

    def oracle(t, d):
        x = d + t * direction
        return (jnp.sum(jnp.einsum("pqrs,brs->bpq", eri, x)**2)
                + jnp.sum(jnp.einsum("prqs,brs->bpq", eri, x)**2))

    hvp = lambda f: jax.jvp(jax.grad(lambda d: f(.1, d)), (density,), (direction,))[1]
    np.testing.assert_allclose(jax.jit(hvp, static_argnums=0)(loss), hvp(oracle), atol=2e-10)


def test_jit_nonzero_basis_tangent_is_explicitly_rejected(system):
    plan, parameters, _, density = system
    direction = jax.tree.map(jnp.ones_like, parameters)
    operation = jax.jit(lambda p, dp: jax.jvp(
        lambda p: plan.get_jk(p, density), (p,), (dp,))[1])
    with pytest.raises((NotImplementedError, RuntimeError), match="density AD|basis/geometry"):
        jax.block_until_ready(operation(parameters, direction))


def test_active_basis_vjp_is_explicitly_rejected(system):
    plan, parameters, _, density = system
    loss = lambda p: sum(jnp.sum(x*x) for x in plan.get_jk(p, density))
    with pytest.raises((NotImplementedError, RuntimeError), match="basis/geometry|density AD"):
        jax.block_until_ready(jax.jit(jax.grad(loss))(parameters))


def test_second_order_basis_variation_is_not_hidden_by_zero_first_tangent(system):
    plan, parameters, _, density = system
    direction = jax.tree.map(jnp.ones_like, parameters)

    def value(t):
        changed = jax.tree.map(lambda p, v: p + t*t*v, parameters, direction)
        return sum(jnp.sum(x*x) for x in plan.get_jk(changed, density))

    with pytest.raises((NotImplementedError, RuntimeError), match="density AD|basis/geometry"):
        jax.block_until_ready(jax.jit(jax.jacfwd(jax.jacfwd(value)))(0.))
