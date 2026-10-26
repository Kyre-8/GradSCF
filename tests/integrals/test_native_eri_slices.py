"""Bounded native ERI shell slabs and their coordinate response.

CPU float64; coordinates and finite-difference steps are in Bohr. Dense ERIs
are separate tiny test oracles, never part of the slab calculation under test.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals


ATOM = "H 0 0 0; H .2 -.1 .8"
BASIS = {"H": [[0, [1.1, 1.]], [2, [.7, 1.]]]}
GENERAL = (1, 3, 2, 4, 1, 2, 3, 4)
DISJOINT = (0, 1, 1, 2, 2, 3, 3, 4)
DIAGONAL = (1, 2, 1, 2, 1, 2, 1, 2)


def _ao_slices(topology, shell_slice):
    counts = [((l + 1) * (l + 2) // 2 if topology.cart else 2 * l + 1) * nc
              for l, nc in zip(topology.angular_momenta, topology.contraction_counts)]
    offsets = np.cumsum([0, *counts])
    return tuple(slice(int(offsets[start]), int(offsets[stop]))
                 for start, stop in zip(shell_slice[::2], shell_slice[1::2]))


def _problem(cart=True, omega=0., shell_slice=GENERAL, basis=None):
    top, params = integrals.prepare_basis(ATOM, basis=BASIS if basis is None else basis,
                                          cart=cart)
    plan = integrals.make_plan(top, backend="native")
    nnuc = len(top.nuclear_charges)
    # The AO centers float independently of both nuclei and other shells.
    centers = params.centers + jnp.arange(params.centers.size).reshape(-1, 3) * .003
    coords = jnp.concatenate((params.nuclear_coords, centers))

    def parameters(r):
        return replace(params, nuclear_coords=r[:nnuc], centers=r[nnuc:])

    def value(r):
        return plan.evaluate("eri", parameters(r), shell_slice=shell_slice, omega=omega)

    def full(r):
        return plan.evaluate("eri", parameters(r), omega=omega)

    return top, params, plan, coords, parameters, value, full


def _fd(function, x, direction, step):
    return (function(x + step * direction) - function(x - step * direction)) / (2 * step)


def _assert_tree_close(actual, expected, *, atol, rtol):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for observed, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(observed, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("cart", [True, False])
@pytest.mark.parametrize("omega", [0., .2])
def test_native_shell_slice_values_match_full_eri(cart, omega):
    top, _, plan, coords, parameters, value, full = _problem(cart, omega)
    # Keep this first: before the feature exists, RED is the missing API.
    actual = value(coords)
    dense = full(coords)
    expected = dense[_ao_slices(top, GENERAL)]
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)
    np.testing.assert_allclose(jax.jit(value)(coords), expected, atol=3e-12, rtol=3e-11)
    for shell_slice in (DISJOINT, DIAGONAL):
        evaluate = lambda r: plan.evaluate("eri", parameters(r), shell_slice=shell_slice,
                                           omega=omega)
        expected = dense[_ao_slices(top, shell_slice)]
        np.testing.assert_allclose(evaluate(coords), expected, atol=3e-12, rtol=3e-11)
        np.testing.assert_allclose(jax.jit(evaluate)(coords), expected,
                                   atol=3e-12, rtol=3e-11)
    np.testing.assert_array_equal(
        plan.evaluate("eri", parameters(coords), shell_slice=None, omega=omega), dense)


@pytest.mark.parametrize("cart,omega,shell_slice", [
    (True, 0., GENERAL), (False, .2, GENERAL),
    (True, 0., DISJOINT), (False, 0., DIAGONAL),
], ids=["cart-general", "spherical-long-range", "disjoint", "diagonal"])
def test_native_shell_slice_geometry_jvp_vjp(cart, omega, shell_slice):
    top, _, _, coords, _, value, full = _problem(cart, omega, shell_slice)
    reference = lambda r: full(r)[_ao_slices(top, shell_slice)]
    rng = np.random.default_rng(20261021)
    direction = jnp.asarray(rng.normal(scale=.15, size=coords.shape))
    primal, tangent = jax.jit(lambda r, d: jax.jvp(value, (r,), (d,)))(coords, direction)
    expected_tangent = jax.jit(lambda r, d: jax.jvp(reference, (r,), (d,))[1])(
        coords, direction)
    np.testing.assert_allclose(tangent, expected_tangent, atol=2e-11, rtol=2e-10)
    cotangent = jnp.asarray(rng.normal(size=primal.shape))  # Intentionally nonsymmetric.
    loss = lambda r: jnp.vdot(value(r), cotangent)
    expected_loss = lambda r: jnp.vdot(reference(r), cotangent)
    gradient = jax.jit(jax.grad(loss))(coords)
    np.testing.assert_allclose(gradient, jax.jit(jax.grad(expected_loss))(coords),
                               atol=2e-11, rtol=2e-10)
    directional = jnp.vdot(gradient, direction)
    np.testing.assert_allclose(directional, jnp.vdot(tangent, cotangent),
                               atol=2e-11, rtol=2e-10)
    for step in (3e-4, 1e-4):
        np.testing.assert_allclose(tangent, _fd(value, coords, direction, step),
                                   atol=2e-8, rtol=3e-6)
        np.testing.assert_allclose(directional, _fd(loss, coords, direction, step),
                                   atol=2e-8, rtol=3e-6)
    np.testing.assert_allclose(gradient[:2], 0., atol=1e-13)


def test_native_shell_slice_nonlinear_coordinate_hvp():
    top, _, _, coords, _, value, full = _problem(basis="3-21g")
    reference = lambda r: full(r)[_ao_slices(top, GENERAL)]

    def loss(evaluate, r):
        block = evaluate(r)
        return jnp.sum(jnp.sin(.3 * block) + .1 * block**2)

    gradient = jax.jit(jax.grad(lambda r: loss(value, r)))
    expected_gradient = jax.jit(jax.grad(lambda r: loss(reference, r)))
    direction = jnp.asarray(np.random.default_rng(91).normal(scale=.2, size=coords.shape))
    actual = jax.jit(lambda r, d: jax.jvp(gradient, (r,), (d,))[1])(coords, direction)
    expected = jax.jit(lambda r, d: jax.jvp(expected_gradient, (r,), (d,))[1])(
        coords, direction)
    np.testing.assert_allclose(gradient(coords), expected_gradient(coords),
                               atol=2e-11, rtol=2e-10)
    np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-9)
    for step in (3e-4, 1e-4):
        np.testing.assert_allclose(actual, _fd(gradient, coords, direction, step),
                                   atol=5e-8, rtol=4e-6)


def test_native_shell_slice_mo_coefficients_and_geometry_hvp():
    top, _, _, coords, _, value, full = _problem(cart=False, omega=.2, basis="3-21g")
    ao_slices = _ao_slices(top, GENERAL)
    sizes = tuple(axis.stop - axis.start for axis in ao_slices)
    rng = np.random.default_rng(92)
    coefficients = tuple(jnp.asarray(rng.normal(scale=.4, size=(n, m)))
                         for n, m in zip(sizes, (2, 1, 2, 1)))
    point = (coords, *coefficients)
    direction = tuple(jnp.asarray(rng.normal(scale=.1, size=x.shape)) for x in point)

    def loss(evaluate, args):
        block = jnp.einsum("pqrs,pi,qj,rk,sl->ijkl", evaluate(args[0]), *args[1:],
                           precision="highest", optimize="optimal")
        return jnp.sum(jnp.sin(.2 * block) + .05 * block**2)

    reference = lambda r: full(r)[ao_slices]
    actual_loss = lambda p: loss(value, p)
    actual_gradient = jax.jit(jax.grad(actual_loss))
    expected_gradient = jax.jit(jax.grad(lambda p: loss(reference, p)))
    actual_grad = actual_gradient(point)
    _assert_tree_close(actual_grad, expected_gradient(point), atol=2e-11, rtol=2e-10)
    actual = jax.jit(lambda p, d: jax.jvp(actual_gradient, (p,), (d,))[1])(point, direction)
    expected = jax.jit(lambda p, d: jax.jvp(expected_gradient, (p,), (d,))[1])(
        point, direction)
    _assert_tree_close(actual, expected, atol=3e-10, rtol=3e-9)
    directional = sum(jnp.vdot(g, d) for g, d in zip(actual_grad, direction))
    for step in (3e-4, 1e-4):
        plus = jax.tree.map(lambda p, d: p + step * d, point, direction)
        minus = jax.tree.map(lambda p, d: p - step * d, point, direction)
        np.testing.assert_allclose(directional, (actual_loss(plus) - actual_loss(minus)) / (2 * step),
                                   atol=2e-8, rtol=3e-6)
        difference = jax.tree.map(lambda a, b: (a - b) / (2 * step),
                                  actual_gradient(plus), actual_gradient(minus))
        _assert_tree_close(actual, difference, atol=5e-8, rtol=4e-6)


def _equation_shapes(value):
    """Visit nested call, scan and checkpoint jaxprs, including FFI outputs."""
    value = getattr(value, "jaxpr", value)
    if hasattr(value, "eqns"):
        for equation in value.eqns:
            for variable in equation.outvars:
                shape = getattr(getattr(variable, "aval", None), "shape", None)
                if shape is not None:
                    yield tuple(shape)
            yield from _equation_shapes(equation.params)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _equation_shapes(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _equation_shapes(child)


def test_native_shell_slice_response_jaxprs_have_no_dense_eri():
    top, _, _, coords, _, value, _ = _problem(basis="3-21g")
    loss = lambda r: jnp.sum(jnp.sin(.3 * value(r)))
    gradient = jax.grad(loss)
    direction = jnp.ones_like(coords) * .1
    functions = (value, gradient, lambda r: jax.jvp(gradient, (r,), (direction,))[1])
    for function in functions:
        shapes = tuple(_equation_shapes(jax.make_jaxpr(jax.jit(function))(coords)))
        assert shapes, "The recursive structural check must inspect equation outputs."
        assert (top.nao,) * 4 not in shapes, shapes


def test_native_shell_slice_rejects_invalid_ranges():
    _, params, plan, _, _, _, _ = _problem()
    invalid = (
        GENERAL[:-1],
        (-1, 1, 0, 1, 0, 1, 0, 1),
        (0, 5, 0, 1, 0, 1, 0, 1),
        (1, 1, 0, 1, 0, 1, 0, 1),
        (2, 1, 0, 1, 0, 1, 0, 1),
    )
    for shell_slice in invalid:
        with pytest.raises(ValueError, match="(?i)shell.?slice"):
            plan.evaluate("eri", params, shell_slice=shell_slice)


def test_native_shell_slice_rejects_noninteger_and_boolean_endpoints():
    _, params, plan, _, _, _, _ = _problem()
    for start in (.0, True, np.bool_(False)):
        shell_slice = (start, 1, 0, 1, 0, 1, 0, 1)
        with pytest.raises((TypeError, ValueError), match="(?i)shell.?slice"):
            plan.evaluate("eri", params, shell_slice=shell_slice)


def test_native_shell_slice_rejects_dynamic_jit_endpoints():
    _, params, plan, _, _, _, _ = _problem()
    evaluate = jax.jit(lambda s: plan.evaluate("eri", params, shell_slice=s))
    with pytest.raises((TypeError, ValueError, NotImplementedError), match="(?i)static"):
        evaluate(jnp.asarray(GENERAL))


@pytest.mark.parametrize("cart,nctr", [(True, 64), (False, 2)],
                         ids=["cart-shell-workspace", "spherical-cartesian-cache"])
def test_native_shell_slice_rejects_static_workspace_overflow(cart, nctr):
    from gradscf.integrals.backends.native.ffi import _eri_shell_slice

    # Metadata only: do not build a primitive, output buffer or basis ENV.
    basis = np.zeros((2, 8), dtype=np.int32)
    basis[:, 2:4] = 1
    basis[1, 1], basis[1, 3] = 12, nctr
    count = (91 if cart else 25) * nctr
    assert count**2 < np.iinfo(np.int32).max
    if not cart:
        # Spherical output fits, but libcint still contracts in Cartesian space.
        assert count**4 < np.iinfo(np.int32).max < 3 * (91 * nctr)**4
    with pytest.raises(ValueError, match="(?i)(shell|cache).*int32"):
        _eri_shell_slice(basis, cart, (1, 2) * 4)


def test_native_shell_slice_is_only_an_eri_parameter():
    _, params, plan, _, _, _, _ = _problem()
    for operator in ("overlap", "kinetic", "nuclear", "dipole", "ecp"):
        with pytest.raises(ValueError, match="(?i)(shell.?slice.*eri|eri.*shell.?slice)"):
            plan.evaluate(operator, params, shell_slice=GENERAL)


@pytest.mark.parametrize("aosym", ["s4", "s8"])
def test_native_shell_slice_rejects_packed_layouts(aosym):
    _, params, plan, _, _, _, _ = _problem()
    with pytest.raises(ValueError, match="(?i)(shell.?slice|s1)"):
        plan.evaluate("eri", params, shell_slice=GENERAL, aosym=aosym)


def test_shell_slice_reference_backend_is_explicitly_unsupported():
    top, params, _, _, _, _, _ = _problem()
    plan = integrals.make_plan(top, backend="jax_reference")
    with pytest.raises(NotImplementedError, match="(?i)(shell.?slice|native)"):
        plan.evaluate("eri", params, shell_slice=GENERAL)


def test_native_shell_slice_retains_static_nonnegative_omega_validation():
    _, params, plan, _, _, _, _ = _problem()
    for omega in (-.2, np.inf, np.nan):
        with pytest.raises(ValueError, match="(?i)omega.*(nonnegative|finite)"):
            plan.evaluate("eri", params, shell_slice=GENERAL, omega=omega)
    loss = lambda omega: plan.evaluate("eri", params, shell_slice=GENERAL, omega=omega).sum()
    for transform in (jax.grad, jax.jit):
        with pytest.raises(NotImplementedError, match="(?i)omega.*static"):
            transform(loss)(.2)


@pytest.mark.parametrize("zero_direction", [False, True])
def test_native_shell_slice_basis_and_third_geometry_ad_remain_unsupported(zero_direction):
    _, params, plan, coords, _, value, _ = _problem(basis="3-21g")
    for field in ("exponents", "coefficients"):
        values = getattr(params, field)
        loss = lambda x: plan.evaluate(
            "eri", replace(params, **{field: (x, *values[1:])}), shell_slice=GENERAL).sum()
        with pytest.raises(NotImplementedError, match="(?i)(exponent|coefficient|basis)"):
            jax.grad(loss)(values[0])
    direction = jnp.linspace(-.2, .3, coords.size).reshape(coords.shape)
    first = lambda r: jax.jvp(value, (r,), (direction,))[1]
    second = lambda r: jax.jvp(first, (r,), (direction,))[1]
    third_direction = jnp.zeros_like(coords) if zero_direction else direction
    with pytest.raises(NotImplementedError, match="(?i)third.*geometry"):
        jax.jvp(second, (coords,), (third_direction,))
