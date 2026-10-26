"""Public native coefficient AD at fixed exponents/geometry, CPU float64.

All references use one-electron matrices or 2c/3c RI factors. No test here
constructs a dense four-center ERI. Coordinates are Bohr, exponents Bohr^-2,
and integral values are atomic units.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.basis.contraction import contraction_matrix, primitive_basis
from gradscf.integrals.molecular.density_fitting import project_factors, unpack_factors


def _basis(cart):
    # General contractions include isolated zeros and negative raw values.
    basis = {
        "He": [
            [0, [1.7, 0.0, 0.7], [0.55, -0.8, 0.3], [0.18, 0.4, -0.2]],
            [1, [0.9, 0.9, -0.2], [0.3, 0.0, 0.6]],
            [2, [0.75, 0.4, 0.0], [0.24, -0.7, 0.8]],
        ],
        "H": [[0, [1.2, -0.3], [0.28, 0.8]]],
    }
    return integrals.prepare_basis(
        "He 0 0 0; H .4 -.3 1.4", basis, unit="Bohr", spin=1, cart=cart
    )


def _direction(coefficients, seed=382):
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.normal(size=c.shape)) for c in coefficients)


def _shift(coefficients, direction, scale):
    return jax.tree_util.tree_map(lambda c, d: c + scale * d, coefficients, direction)


def _fd4(value, coefficients, direction, step=1e-4):
    plus = value(_shift(coefficients, direction, step))
    minus = value(_shift(coefficients, direction, -step))
    plus2 = value(_shift(coefficients, direction, 2 * step))
    minus2 = value(_shift(coefficients, direction, -2 * step))
    return jax.tree_util.tree_map(
        lambda a, b, c, d: (8 * (a - b) - c + d) / (12 * step),
        plus, minus, plus2, minus2,
    )


def _assert_tree_allclose(actual, expected, **kwargs):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(a, b, **kwargs)


def _primitive_reference(top, parameters, operator):
    primitive_top, primitive_parameters = primitive_basis(top, parameters)
    primitive = integrals.make_plan(primitive_top, backend="native").evaluate(
        operator, primitive_parameters
    )

    def value(coefficients):
        transform = contraction_matrix(top, replace(parameters, coefficients=coefficients))
        return jnp.einsum("pi,...pq,qj->...ij", transform, primitive, transform)

    return value


@pytest.mark.parametrize("cart", [True, False])
@pytest.mark.parametrize("operator", ["overlap", "kinetic", "nuclear", "dipole"])
def test_native_raw_coefficient_jvp_vjp_fd_and_adjoint(operator, cart):
    top, p = _basis(cart)
    plan = integrals.make_plan(top, backend="native")
    value = lambda c: plan.evaluate(operator, replace(p, coefficients=c))
    c, direction = p.coefficients, _direction(p.coefficients)
    primal, tangent = jax.jit(lambda c, d: jax.jvp(value, (c,), (d,)))(c, direction)
    np.testing.assert_allclose(tangent, _fd4(value, c, direction), atol=2e-8, rtol=2e-7)
    rng = np.random.default_rng(739)
    cotangent = jnp.asarray(rng.normal(size=primal.shape))  # nonsymmetric
    gradient = jax.jit(jax.grad(lambda c: jnp.sum(cotangent * value(c))))(c)
    lhs = sum(jnp.vdot(g, d) for g, d in zip(gradient, direction))
    np.testing.assert_allclose(lhs, jnp.vdot(cotangent, tangent), atol=2e-10, rtol=2e-11)


@pytest.mark.parametrize("cart", [True, False])
@pytest.mark.parametrize("operator", ["overlap", "kinetic", "nuclear", "dipole"])
def test_native_coefficient_gradient_hvp_matches_primitive_projection(operator, cart):
    top, p = _basis(cart)
    plan = integrals.make_plan(top, backend="native")
    value = lambda c: plan.evaluate(operator, replace(p, coefficients=c))
    reference = _primitive_reference(top, p, operator)
    c, direction = p.coefficients, _direction(p.coefficients, seed=617)
    actual = value(c)
    np.testing.assert_allclose(actual, reference(c), atol=2e-11, rtol=2e-11)
    rng = np.random.default_rng(561)
    cotangent = jnp.asarray(rng.normal(size=actual.shape))
    loss = lambda c: jnp.sum(cotangent * value(c))
    reference_loss = lambda c: jnp.sum(cotangent * reference(c))
    gradient = jax.jit(jax.grad(loss))
    _assert_tree_allclose(gradient(c), jax.grad(reference_loss)(c), atol=2e-9, rtol=2e-9)
    hvp = jax.jit(lambda c, d: jax.jvp(jax.grad(loss), (c,), (d,))[1])(c, direction)
    expected = jax.jvp(jax.grad(reference_loss), (c,), (direction,))[1]
    _assert_tree_allclose(hvp, expected, atol=3e-8, rtol=3e-8)
    _assert_tree_allclose(hvp, _fd4(gradient, c, direction), atol=3e-7, rtol=3e-7)


@pytest.mark.parametrize("cart", [True, False])
def test_native_coefficient_jacfwd_jacrev_and_vmap(cart):
    top, p = _basis(cart)
    plan = integrals.make_plan(top, backend="native")
    value = lambda c: plan.evaluate("kinetic", replace(p, coefficients=(c, *p.coefficients[1:])))
    c = p.coefficients[0]
    forward = jax.jit(jax.jacfwd(value))(c)
    reverse = jax.jit(jax.jacrev(value))(c)
    np.testing.assert_allclose(forward, reverse, atol=2e-11, rtol=2e-11)
    batch = jnp.stack((c, c + 0.02 * _direction((c,))[0]))
    loss = lambda c: jnp.sum(value(c) ** 2)
    actual = jax.jit(jax.vmap(jax.grad(loss)))(batch)
    expected = jnp.stack(tuple(jax.grad(loss)(coefficients) for coefficients in batch))
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-11)
    reference = _primitive_reference(top, p, "kinetic")
    reference_loss = lambda c: jnp.sum(reference((c, *p.coefficients[1:])) ** 2)
    forward_hessian = jax.jit(jax.jacfwd(jax.grad(loss)))(c)
    reverse_hessian = jax.jit(jax.jacrev(jax.grad(loss)))(c)
    expected_hessian = jax.hessian(reference_loss)(c)
    np.testing.assert_allclose(forward_hessian, expected_hessian, atol=3e-8, rtol=3e-8)
    np.testing.assert_allclose(reverse_hessian, expected_hessian, atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("operator", ["overlap", "kinetic", "nuclear", "dipole"])
def test_native_coefficient_normalization_scale_gauge(operator):
    top, p = _basis(cart=False)
    plan = integrals.make_plan(top, backend="native")
    value = lambda c: plan.evaluate(operator, replace(p, coefficients=c))
    c = p.coefficients
    # Every contraction column has an independent positive scale gauge.
    direction = tuple(raw * jnp.arange(1, raw.shape[1] + 1)[None, :] for raw in c)
    primal, tangent = jax.jvp(value, (c,), (direction,))
    np.testing.assert_allclose(tangent, 0.0, atol=2e-11, rtol=0.0)
    cotangent = jnp.arange(primal.size, dtype=jnp.float64).reshape(primal.shape) / primal.size
    gradient = jax.grad(lambda c: jnp.sum(value(c) * cotangent))(c)
    for raw, g in zip(c, gradient):
        np.testing.assert_allclose(jnp.sum(raw * g, axis=0), 0.0, atol=2e-11, rtol=0.0)


def test_native_entire_zero_contraction_column_is_invalid():
    top, p = _basis(cart=False)
    plan = integrals.make_plan(top, backend="native")
    zero_column = p.coefficients[0].at[:, 0].set(0.0)
    parameters = replace(p, coefficients=(zero_column, *p.coefficients[1:]))
    # Normalization of an entirely zero contraction is undefined: an explicit
    # failure or nonfinite result must reach the caller, never a finite basis.
    try:
        value = plan.evaluate("overlap", parameters)
    except (ValueError, RuntimeError):
        return
    assert not bool(jnp.all(jnp.isfinite(value)))


@pytest.mark.parametrize("operator", ["overlap", "kinetic", "nuclear", "dipole"])
def test_native_exponent_second_derivatives_raise_including_normalization(operator):
    top, p = _basis(cart=False)
    plan = integrals.make_plan(top, backend="native")

    def loss(alpha):
        parameters = replace(p, exponents=(alpha, *p.exponents[1:]))
        return jnp.sum(plan.evaluate(operator, parameters))

    with pytest.raises(NotImplementedError, match="(?i)(exponent|basis)"):
        jax.hessian(loss)(p.exponents[0])


def _auxiliary(cart):
    return integrals.prepare_basis(
        "He 0 0 0; H .4 -.3 1.4",
        {"He": [[0, [0.8, 1.0]], [1, [0.6, 1.0]], [2, [0.4, 1.0]]],
         "H": [[0, [0.55, 1.0]]]},
        unit="Bohr", spin=1, cart=cart,
    )


@pytest.mark.parametrize("cart", [True, False])
def test_native_df_coefficient_ad_matches_primitive_projection(cart):
    top, p = _basis(cart)
    auxiliary_top, ap = _auxiliary(cart)
    plan = integrals.make_auxiliary_plan(top, auxiliary_top)
    # Both metric factorizations and the primitive factors are built outside AD.
    metric = plan.metric_factor(p, ap)
    primitive_top, primitive_parameters = primitive_basis(top, p)
    primitive_plan = integrals.make_auxiliary_plan(primitive_top, auxiliary_top)
    primitive_metric = primitive_plan.metric_factor(primitive_parameters, ap)
    primitive = primitive_plan.factors(primitive_parameters, ap, metric_factor=primitive_metric)

    def value(c):
        packed = plan.factors(replace(p, coefficients=c), ap, metric_factor=metric)
        return unpack_factors(packed, top.nao)

    def reference(c):
        transform = contraction_matrix(top, replace(p, coefficients=c))
        return project_factors(primitive, transform, block_size=3)

    c, direction = p.coefficients, _direction(p.coefficients, seed=492)
    actual = value(c)
    np.testing.assert_allclose(actual, reference(c), atol=5e-11, rtol=3e-11)
    _, tangent = jax.jit(lambda c, d: jax.jvp(value, (c,), (d,)))(c, direction)
    np.testing.assert_allclose(tangent, _fd4(value, c, direction), atol=3e-8, rtol=3e-7)
    rng = np.random.default_rng(413)
    cotangent = jnp.asarray(rng.normal(size=actual.shape))  # nonsymmetric AO pair
    loss = lambda c: jnp.sum(cotangent * value(c))
    reference_loss = lambda c: jnp.sum(cotangent * reference(c))
    gradient = jax.jit(jax.grad(loss))
    _assert_tree_allclose(gradient(c), jax.grad(reference_loss)(c), atol=3e-9, rtol=3e-9)
    lhs = sum(jnp.vdot(g, d) for g, d in zip(gradient(c), direction))
    np.testing.assert_allclose(lhs, jnp.vdot(cotangent, tangent), atol=2e-10, rtol=2e-11)
    hvp = jax.jit(lambda c, d: jax.jvp(jax.grad(loss), (c,), (d,))[1])(c, direction)
    expected = jax.jvp(jax.grad(reference_loss), (c,), (direction,))[1]
    _assert_tree_allclose(hvp, expected, atol=3e-8, rtol=3e-8)
    _assert_tree_allclose(hvp, _fd4(gradient, c, direction), atol=3e-7, rtol=3e-7)
    reverse_hvp = jax.jit(jax.grad(lambda c: jax.jvp(loss, (c,), (direction,))[1]))(c)
    _assert_tree_allclose(reverse_hvp, expected, atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("target,field", [
    ("orbital", "centers"), ("orbital", "nuclear_coords"),
    ("auxiliary", "coefficients"), ("auxiliary", "exponents"),
    ("auxiliary", "centers"), ("auxiliary", "nuclear_coords"),
])
def test_native_df_fixed_metric_unsupported_derivatives_raise(target, field):
    top, p = _basis(cart=False)
    auxiliary_top, ap = _auxiliary(cart=False)
    plan = integrals.make_auxiliary_plan(top, auxiliary_top)
    metric = plan.metric_factor(p, ap)
    parameters = p if target == "orbital" else ap
    values = getattr(parameters, field)
    initial = values[0] if isinstance(values, tuple) else values

    def loss(value):
        changed = (value, *values[1:]) if isinstance(values, tuple) else value
        varied = replace(parameters, **{field: changed})
        orbital, auxiliary = (varied, ap) if target == "orbital" else (p, varied)
        return jnp.sum(plan.factors(orbital, auxiliary, metric_factor=metric))

    with pytest.raises(NotImplementedError, match="(?i)(exponent|auxiliary|geometry|coordinate|basis)"):
        jax.grad(loss)(initial)
