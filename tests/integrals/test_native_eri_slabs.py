"""Dynamic padded native ERI slabs and coordinate response.

CPU float64; coordinates and finite-difference steps are in Bohr. Small
static shell slices are independent oracles, never part of the dynamic path.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.backends.native.autodiff import geometry


ATOM = "H 0 0 0; H .2 -.1 .8"
D_BASIS = {"H": [[0, [1.1, 1.]], [2, [.7, 1.]]]}
GENERAL = (1, 3, 2, 4, 1, 2, 3, 4)
DISJOINT = (0, 1, 1, 2, 2, 3, 3, 4)
TAIL = (3, 4, 0, 1, 2, 4, 1, 3)
WINDOWS = (GENERAL, DISJOINT, TAIL)


def _axis_sizes(topology, bounds):
    counts = [((l + 1) * (l + 2) // 2 if topology.cart else 2 * l + 1) * nc
              for l, nc in zip(topology.angular_momenta, topology.contraction_counts)]
    return tuple(sum(counts[start:stop])
                 for start, stop in zip(bounds[::2], bounds[1::2]))


def _valid_mask(topology, bounds, shape):
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(slice(0, size) for size in _axis_sizes(topology, bounds))] = True
    return mask


def _fd(function, x, direction, step):
    return (function(x + step * direction) - function(x - step * direction)) / (2 * step)


def _problem(cart=True, omega=0., basis="3-21g"):
    top, params = integrals.prepare_basis(ATOM, basis=basis, cart=cart)
    plan = integrals.make_plan(top, backend="native")
    nnuc = len(top.nuclear_charges)
    centers = params.centers + jnp.arange(params.centers.size).reshape(-1, 3) * .003
    coords = jnp.concatenate((params.nuclear_coords, centers))
    fixed = replace(params, nuclear_coords=jnp.zeros_like(params.nuclear_coords),
                    centers=jnp.zeros_like(params.centers))
    atm, bas, env = plan.pack(fixed)
    env = env.at[8].set(omega)
    sizes = tuple(_axis_sizes(top, bounds) for bounds in WINDOWS)
    shape = tuple(max(axis) + pad for axis, pad in zip(zip(*sizes), (1, 2, 1, 2)))

    def value(r, bounds, padded_shape=shape):
        return geometry.evaluate_eri_slab(atm, bas, env, r, bounds, padded_shape, cart=cart)

    def reference(r, bounds):
        parameters = replace(params, nuclear_coords=r[:nnuc], centers=r[nnuc:])
        block = plan.evaluate("eri", parameters, shell_slice=bounds, omega=omega)
        padding = tuple((0, padded - actual) for padded, actual in zip(shape, block.shape))
        return jnp.pad(block, padding)

    return top, coords, shape, value, reference, (atm, bas, env)


@pytest.mark.parametrize("cart,basis,omega", [
    (True, "3-21g", 0.), (True, D_BASIS, 0.), (False, D_BASIS, .2),
], ids=["s", "cart-d", "spherical-d-long-range"])
def test_native_dynamic_eri_slab_values_and_zero_padding(cart, basis, omega):
    assert callable(getattr(geometry, "evaluate_eri_slab", None)), (
        "Native dynamic ERI slabs require geometry.evaluate_eri_slab")
    top, coords, shape, value, reference, _ = _problem(cart, omega, basis)
    compiled = jax.jit(value)
    for bounds in WINDOWS:
        actual = compiled(coords, jnp.asarray(bounds, dtype=jnp.int32))
        expected = reference(coords, bounds)
        assert actual.shape == shape
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)
        np.testing.assert_array_equal(np.asarray(actual)[~_valid_mask(top, bounds, shape)], 0.)


def test_native_dynamic_eri_slab_raw_ffi_matches_geometry_entry():
    from gradscf.integrals.backends.native import ffi

    _, coords, shape, value, _, (atm, bas, env) = _problem(False, .2, D_BASIS)
    offsets = atm[:, 1, None] + np.arange(3)[None, :]
    packed = env.at[offsets].set(coords, unique_indices=True)
    bounds = jnp.asarray(TAIL, dtype=jnp.int32)
    raw = jax.jit(lambda e, b: ffi.evaluate_eri_slab(atm, bas, e, b, shape, cart=False))
    np.testing.assert_allclose(raw(packed, bounds), value(coords, bounds),
                               atol=3e-12, rtol=3e-11)


def test_native_dynamic_eri_slab_jit_reuses_trace_for_changing_windows():
    _, coords, _, value, reference, _ = _problem(basis=D_BASIS)
    traces = []

    def evaluate(r, bounds):
        traces.append(None)
        return value(r, bounds)

    compiled = jax.jit(evaluate)
    for bounds in (*WINDOWS, GENERAL):
        actual = compiled(coords, jnp.asarray(bounds, dtype=jnp.int32))
        np.testing.assert_allclose(actual, reference(coords, bounds), atol=3e-12, rtol=3e-11)
    assert len(traces) == 1, "Window values must be runtime operands, not JIT specialization keys."


@pytest.mark.parametrize("cart,basis,omega,bounds", [
    (True, "3-21g", 0., TAIL),
    (True, D_BASIS, 0., DISJOINT),
    (False, D_BASIS, .2, TAIL),
], ids=["s-tail", "cart-d-disjoint", "spherical-d-long-range-tail"])
def test_native_dynamic_eri_slab_coordinate_jvp_vjp(cart, basis, omega, bounds):
    top, coords, shape, value, reference, _ = _problem(cart, omega, basis)
    dynamic_bounds = jnp.asarray(bounds, dtype=jnp.int32)
    rng = np.random.default_rng(20261022)
    direction = jnp.asarray(rng.normal(scale=.15, size=coords.shape))
    primal, tangent = jax.jit(lambda r, d, b: jax.jvp(
        lambda x: value(x, b), (r,), (d,)))(coords, direction, dynamic_bounds)
    oracle = lambda r: reference(r, bounds)
    expected_tangent = jax.jit(lambda r, d: jax.jvp(oracle, (r,), (d,))[1])(
        coords, direction)
    np.testing.assert_allclose(tangent, expected_tangent, atol=2e-11, rtol=2e-10)
    mask = _valid_mask(top, bounds, shape)
    np.testing.assert_array_equal(np.asarray(tangent)[~mask], 0.)
    cotangent = jnp.asarray(rng.normal(size=primal.shape))  # Intentionally nonsymmetric.
    gradient = jax.jit(lambda r, b, c: jax.grad(
        lambda x: jnp.vdot(value(x, b), c))(r))(coords, dynamic_bounds, cotangent)
    loss = lambda r: jnp.vdot(value(r, dynamic_bounds), cotangent)
    expected_loss = lambda r: jnp.vdot(oracle(r), cotangent)
    np.testing.assert_allclose(gradient, jax.jit(jax.grad(expected_loss))(coords),
                               atol=2e-11, rtol=2e-10)
    directional = jnp.vdot(gradient, direction)
    np.testing.assert_allclose(directional, jnp.vdot(tangent, cotangent),
                               atol=2e-11, rtol=2e-10)
    for step in (3e-4, 1e-4):
        np.testing.assert_allclose(tangent, _fd(lambda r: value(r, dynamic_bounds),
                                               coords, direction, step),
                                   atol=2e-8, rtol=3e-6)
        np.testing.assert_allclose(directional, _fd(loss, coords, direction, step),
                                   atol=2e-8, rtol=3e-6)
    np.testing.assert_allclose(gradient[:len(top.nuclear_charges)], 0., atol=1e-13)


@pytest.mark.parametrize("cart", [True, False])
def test_native_dynamic_eri_slab_ignores_padded_cotangents(cart):
    top, coords, shape, value, _, _ = _problem(cart=cart, basis=D_BASIS)
    bounds = jnp.asarray(DISJOINT, dtype=jnp.int32)
    mask = _valid_mask(top, DISJOINT, shape)
    rng = np.random.default_rng(420)
    cotangent = jnp.asarray(rng.normal(size=shape))
    padding_only = jnp.where(mask, 0., cotangent)
    active_only = jnp.where(mask, cotangent, 0.)

    def pullback(r, b, c):
        return jax.vjp(lambda x: value(x, b), r)[1](c)[0]

    compiled = jax.jit(pullback)
    np.testing.assert_allclose(compiled(coords, bounds, cotangent),
                               compiled(coords, bounds, active_only), atol=1e-13, rtol=1e-13)
    np.testing.assert_array_equal(compiled(coords, bounds, padding_only), 0.)


@pytest.mark.parametrize("cart,basis,omega,bounds", [
    (True, "3-21g", 0., GENERAL), (False, D_BASIS, .2, DISJOINT),
], ids=["s", "spherical-d-long-range"])
def test_native_dynamic_eri_slab_nonlinear_coordinate_hvp(cart, basis, omega, bounds):
    _, coords, _, value, reference, _ = _problem(cart, omega, basis)
    dynamic_bounds = jnp.asarray(bounds, dtype=jnp.int32)

    def loss(r, b):
        block = value(r, b)
        return jnp.sum(jnp.sin(.3 * block) + .1 * block**2)

    def expected_loss(r):
        block = reference(r, bounds)
        return jnp.sum(jnp.sin(.3 * block) + .1 * block**2)

    gradient = jax.jit(jax.grad(loss, argnums=0))
    expected_gradient = jax.jit(jax.grad(expected_loss))
    direction = jnp.asarray(np.random.default_rng(421).normal(scale=.2, size=coords.shape))
    actual = jax.jit(lambda r, d, b: jax.jvp(
        lambda x: gradient(x, b), (r,), (d,))[1])(coords, direction, dynamic_bounds)
    expected = jax.jit(lambda r, d: jax.jvp(expected_gradient, (r,), (d,))[1])(
        coords, direction)
    np.testing.assert_allclose(gradient(coords, dynamic_bounds), expected_gradient(coords),
                               atol=2e-11, rtol=2e-10)
    np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-9)
    for step in (3e-4, 1e-4):
        np.testing.assert_allclose(actual, _fd(lambda r: gradient(r, dynamic_bounds),
                                              coords, direction, step),
                                   atol=5e-8, rtol=4e-6)


def _equations(value):
    """Inspect FFI outputs inside nested JIT, loop and derivative jaxprs."""
    value = getattr(value, "jaxpr", value)
    if hasattr(value, "eqns"):
        for equation in value.eqns:
            yield equation
            yield from _equations(equation.params)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _equations(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _equations(child)


def _equation_shapes(equations):
    return tuple(tuple(shape) for equation in equations for variable in equation.outvars
                 if (shape := getattr(getattr(variable, "aval", None), "shape", None)) is not None)


def test_native_dynamic_eri_slab_fori_loop_has_compact_graph_and_matches_slices():
    top, coords, shape, value, reference, _ = _problem()
    windows = jnp.asarray(WINDOWS, dtype=jnp.int32)

    def summed(r, table, count):
        def body(i, result):
            return result + value(r, table[i % table.shape[0]])
        return jax.lax.fori_loop(0, count, body, jnp.zeros(shape, dtype=jnp.float64))

    actual = jax.jit(lambda r, b: summed(r, b, 3))(coords, windows)
    expected = sum((reference(coords, b) for b in WINDOWS), jnp.zeros(shape))
    np.testing.assert_allclose(actual, expected, atol=4e-12, rtol=4e-11)

    def loss(r, table):
        return jnp.sum(jnp.sin(.2 * summed(r, table, 3)))

    def expected_loss(r):
        result = sum((reference(r, b) for b in WINDOWS), jnp.zeros(shape))
        return jnp.sum(jnp.sin(.2 * result))

    gradient = jax.jit(jax.grad(loss, argnums=0))
    expected_gradient = jax.jit(jax.grad(expected_loss))
    np.testing.assert_allclose(gradient(coords, windows), expected_gradient(coords),
                               atol=3e-11, rtol=3e-10)
    direction = jnp.asarray(np.random.default_rng(422).normal(scale=.1, size=coords.shape))
    actual_hvp = jax.jit(lambda r, d, b: jax.jvp(
        lambda x: gradient(x, b), (r,), (d,))[1])(coords, direction, windows)
    expected_hvp = jax.jit(lambda r, d: jax.jvp(expected_gradient, (r,), (d,))[1])(
        coords, direction)
    np.testing.assert_allclose(actual_hvp, expected_hvp, atol=3e-10, rtol=3e-9)

    short = tuple(_equations(jax.make_jaxpr(jax.jit(
        lambda r, b: summed(r, b, 3)))(coords, windows)))
    long = tuple(_equations(jax.make_jaxpr(jax.jit(
        lambda r, b: summed(r, b, 300)))(coords, windows)))
    assert any(e.primitive.name in ("scan", "while") for e in short), short
    assert len(long) <= len(short) + 2, "Increasing iterations must not unroll slab evaluations."
    for equations in (short, long):
        shapes = _equation_shapes(equations)
        assert shapes, "The structural check must inspect nested equation outputs."
        assert (top.nao,) * 4 not in shapes, shapes


def test_native_dynamic_eri_slab_response_jaxprs_have_no_dense_eri():
    top, coords, _, value, _, _ = _problem()
    bounds = jnp.asarray(TAIL, dtype=jnp.int32)
    loss = lambda r, b: jnp.sum(jnp.sin(.3 * value(r, b)))
    gradient = jax.grad(loss, argnums=0)
    direction = jnp.ones_like(coords) * .1
    functions = (value, gradient,
                 lambda r, b: jax.jvp(lambda x: gradient(x, b), (r,), (direction,))[1])
    for function in functions:
        equations = tuple(_equations(jax.make_jaxpr(jax.jit(function))(coords, bounds)))
        shapes = _equation_shapes(equations)
        assert shapes, "The structural check must inspect nested equation outputs."
        assert (top.nao,) * 4 not in shapes, shapes


@pytest.mark.parametrize("shape", [
    (1, 1, 1), (1, 1, 1, 1, 1), (0, 1, 1, 1), (-1, 1, 1, 1),
    (1., 1, 1, 1), (True, 1, 1, 1),
])
def test_native_dynamic_eri_slab_rejects_invalid_static_shape(shape):
    _, coords, _, value, _, _ = _problem()
    bounds = jnp.asarray((0, 1) * 4, dtype=jnp.int32)
    with pytest.raises((TypeError, ValueError), match="(?i)(shape|static|positive)"):
        jax.eval_shape(lambda r: value(r, bounds, shape), coords)


def test_native_dynamic_eri_slab_shape_must_remain_static_under_jit():
    _, coords, shape, value, _, _ = _problem()
    bounds = jnp.asarray((0, 1) * 4, dtype=jnp.int32)
    with pytest.raises((TypeError, ValueError), match="(?i)(shape|static|concrete)"):
        jax.jit(lambda s: value(coords, bounds, s))(jnp.asarray(shape, dtype=jnp.int32))


@pytest.mark.parametrize("bounds", [
    np.zeros((4, 2), dtype=np.int32), np.zeros(7, dtype=np.int32),
    np.asarray(GENERAL, dtype=np.float64), np.asarray(GENERAL, dtype=np.int64),
    np.asarray(GENERAL, dtype=bool),
])
def test_native_dynamic_eri_slab_requires_int32_vector_bounds(bounds):
    _, coords, _, value, _, _ = _problem()
    with pytest.raises((TypeError, ValueError), match="(?i)(bound|int32|shape)"):
        jax.eval_shape(lambda b: value(coords, b), jnp.asarray(bounds))


@pytest.mark.parametrize("cart", [True, False])
def test_native_dynamic_eri_slab_minimum_buffer_depends_on_selected_shells(cart):
    _, coords, _, value, reference, _ = _problem(cart=cart, basis=D_BASIS)
    bounds = (0, 1) * 4  # A one-AO s shell fits despite unselected d shells.
    actual = jax.jit(lambda r, b: value(r, b, (1, 1, 1, 1)))(
        coords, jnp.asarray(bounds, dtype=jnp.int32))
    assert actual.shape == (1, 1, 1, 1)
    np.testing.assert_allclose(actual[0, 0, 0, 0], reference(coords, bounds)[0, 0, 0, 0],
                               atol=3e-12, rtol=3e-11)


def test_native_dynamic_eri_slab_rejects_buffer_smaller_than_selected_window():
    _, coords, _, value, _, _ = _problem(basis=D_BASIS)
    compiled = jax.jit(lambda r, b: value(r, b, (1, 1, 1, 1)))
    with pytest.raises((ValueError, RuntimeError), match="(?i)(shape|slab|padd|window|fit|dimension)"):
        compiled(coords, jnp.asarray((1, 2) * 4, dtype=jnp.int32)).block_until_ready()


def test_native_dynamic_eri_slab_basis_environment_derivative_is_explicitly_unsupported():
    _, coords, shape, _, _, (atm, bas, env) = _problem()
    bounds = jnp.asarray(GENERAL, dtype=jnp.int32)
    loss = lambda e: geometry.evaluate_eri_slab(atm, bas, e, coords, bounds, shape).sum()
    with pytest.raises(NotImplementedError, match="(?i)(basis|exponent|coefficient)"):
        jax.grad(loss)(env)
