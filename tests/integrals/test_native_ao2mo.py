"""Exact non-DF AO->MO blocks; tiny dense arrays are independent test oracles."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.ao2mo import native
from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis
from gradscf.ao2mo import transform_eri_block


def _problem(basis_spec='3-21g', cart=True):
    top, parameters = integrals.prepare_basis('H 0 0 0; H .2 -.1 .8', basis=basis_spec, cart=cart)
    plan = integrals.make_plan(top, backend='native')
    rng = np.random.default_rng(218)
    coefficients = tuple(jnp.asarray(rng.normal(scale=.3, size=(top.nao, n)))
                         for n in (2, 3, 1, 2))
    return NativeDirectBasis(plan, parameters), coefficients


def _dense(basis, coefficients):
    return jnp.einsum('pqrs,pP,qQ,rR,sS->PQRS',
                      basis.plan.evaluate('eri', basis.parameters), *coefficients,
                      precision='highest', optimize='optimal')


def _equations(value):
    value = getattr(value, 'jaxpr', value)
    if hasattr(value, 'eqns'):
        for equation in value.eqns:
            yield equation
            yield from _equations(equation.params)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _equations(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _equations(child)


@pytest.mark.parametrize('cart,basis_spec', [
    (True, '3-21g'), (True, {'H': [[0, [1.1, 1.]], [2, [.7, 1.]]]}),
    (False, {'H': [[0, [1.1, 1.]], [2, [.7, 1.]]]}),
], ids=['s', 'cartesian-d', 'spherical-d'])
def test_native_ao2mo_asymmetric_block_matches_dense(cart, basis_spec):
    basis, coefficients = _problem(basis_spec, cart)
    expected = _dense(basis, coefficients)
    actual = jax.jit(lambda b, cs: transform_eri_block(
        cs, direct_basis=b, max_memory=3.))(basis, coefficients)
    assert actual.shape == (2, 3, 1, 2)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)


def test_native_ao2mo_coordinate_coefficient_response():
    basis, coefficients = _problem()
    parameters = basis.parameters
    coords = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
    rng = np.random.default_rng(219)
    direction = (jnp.asarray(rng.normal(scale=.1, size=coords.shape)),
                 tuple(jnp.asarray(rng.normal(scale=.1, size=c.shape)) for c in coefficients))

    def loss(r, cs, direct):
        moving = NativeDirectBasis(basis.plan, replace(
            parameters, nuclear_coords=r[:2], centers=r[2:]))
        g = (transform_eri_block(cs, direct_basis=moving, max_memory=.12)
             if direct else _dense(moving, cs))
        return jnp.sum(jnp.sin(g) + .2*g*g)

    direct = lambda r, cs: loss(r, cs, True)
    dense = lambda r, cs: loss(r, cs, False)
    inputs = (coords, coefficients)
    for operation in (lambda f, x: jax.jvp(f, x, direction),
                      lambda f, x: jax.grad(f, argnums=(0, 1))(*x),
                      lambda f, x: jax.jvp(jax.grad(f, argnums=(0, 1)), x, direction)[1]):
        actual = jax.jit(lambda r, cs: operation(direct, (r, cs)))(*inputs)
        expected = operation(dense, inputs)
        for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_allclose(a, e, atol=3e-10, rtol=3e-9)
    gradient = jax.grad(direct, argnums=(0, 1))(*inputs)
    slope = sum(jnp.vdot(a, b) for a, b in zip(jax.tree.leaves(gradient), jax.tree.leaves(direction)))
    for step in (3e-4, 1e-4):
        plus = jax.tree.map(lambda a, d: a + step*d, inputs, direction)
        minus = jax.tree.map(lambda a, d: a - step*d, inputs, direction)
        finite_difference = (direct(*plus) - direct(*minus))/(2*step)
        np.testing.assert_allclose(slope, finite_difference, atol=3e-8, rtol=3e-7)


def test_native_ao2mo_no_full_ao_primal_or_cotangent():
    basis, coefficients = _problem()
    def loss(cs):
        return jnp.sum(transform_eri_block(cs, direct_basis=basis, max_memory=.12)**2)
    for graph in (jax.make_jaxpr(loss)(coefficients),
                  jax.make_jaxpr(jax.grad(loss))(coefficients)):
        for equation in _equations(graph):
            for variable in equation.outvars:
                assert getattr(getattr(variable, 'aval', None), 'shape', None) != (basis.nao,)*4


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64], ids=['float32-transform', 'float64-transform'])
def test_projected_native_ao2mo_keeps_transform_response(dtype):
    basis, coefficients = _problem()
    transform = (jnp.eye(basis.nao)[:, :3] + .03).astype(dtype)
    cs = tuple(c[:3] for c in coefficients)
    def value(t, direct):
        primitive = tuple(t @ c for c in cs)
        return jnp.sum((transform_eri_block(cs,
            direct_basis=ProjectedNativeDirectBasis(basis, t), max_memory=.12)
            if direct else _dense(basis, primitive))**2)
    actual = jax.value_and_grad(lambda t: value(t, True))(transform)
    expected = jax.value_and_grad(lambda t: value(t, False))(transform)
    for a, e in zip(actual, expected):
        tolerance = 3e-6 if dtype == jnp.float32 else 2e-10
        np.testing.assert_allclose(a, e, atol=tolerance/10, rtol=tolerance)


def test_native_ao2mo_budget_error_precedes_integral_call(monkeypatch):
    basis, coefficients = _problem()
    def forbidden(*args, **kwargs):
        pytest.fail('Insufficient budget reached an integral evaluation')
    monkeypatch.setattr(native, 'evaluate_eri_slab', forbidden)
    with pytest.raises(ValueError, match='(?i)(budget|memory|minimum)'):
        transform_eri_block(coefficients, direct_basis=basis, max_memory=.0001)


def test_projected_native_ao2mo_rejects_complex_transform():
    basis, coefficients = _problem()
    transform = jnp.eye(basis.nao, dtype=jnp.complex128) + .01j
    with pytest.raises(NotImplementedError, match='(?i)(real|complex)'):
        transform_eri_block(coefficients,
            direct_basis=ProjectedNativeDirectBasis(basis, transform), max_memory=.12)


def test_projected_native_ao2mo_checks_budget_before_projection():
    basis, coefficients = _problem()
    class UnallocatedProjection:
        shape = (basis.nao, basis.nao)
        dtype = jnp.dtype(jnp.float64)
        ndim = 2
        nbytes = basis.nao**2 * 8

        def __jax_array__(self):
            pytest.fail('Insufficient memory reached projection-array materialization')

    with pytest.raises(ValueError, match='(?i)(memory|budget|resident|minimum)'):
        transform_eri_block(coefficients, max_memory=.0001,
            direct_basis=ProjectedNativeDirectBasis(basis, UnallocatedProjection()))


def test_native_ao2mo_empty_space_does_not_evaluate_integrals(monkeypatch):
    basis, coefficients = _problem()
    from gradscf.ao2mo import native
    def forbidden(*args, **kwargs):
        pytest.fail('Empty MO block reached native ERI evaluation')
    monkeypatch.setattr(native, 'evaluate_eri_slab', forbidden)
    cs = (coefficients[0][:, :0], *coefficients[1:])
    actual = transform_eri_block(cs, direct_basis=basis, max_memory=.12)
    assert actual.shape == (0, 3, 1, 2) and actual.dtype == jnp.float64


def test_projected_native_ao2mo_budgets_matmul_input_promotion():
    basis, _ = _problem()
    c = jnp.zeros((100000, 1), dtype=jnp.float64)
    class UnallocatedProjection:
        shape = (basis.nao, 100000)
        dtype = jnp.dtype(jnp.float32)
        nbytes = basis.nao * 100000 * 4

        def __jax_array__(self):
            pytest.fail('Missing promotion storage allowed a projection allocation')

    # Original T and a shared c use 2.4 MB. T's additional float64 3.2 MB
    # does not fit the 4.8 MB usable budget, before even native scratch.
    with pytest.raises(ValueError, match='(?i)(memory|budget|resident|minimum)'):
        transform_eri_block((c,)*4, max_memory=6.,
            direct_basis=ProjectedNativeDirectBasis(basis, UnallocatedProjection()))
