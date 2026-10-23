"""Bounded direct J/K geometry response versus an independent small AO-ERI oracle."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.integrals.plan import IntegralPlan


@pytest.fixture(params=[True, False], ids=["cartesian", "spherical"])
def problem(request):
    topology, parameters = integrals.prepare_basis(
        "H 0 0 0; H .2 -.1 .8", "3-21g", cart=request.param)
    plan = integrals.make_plan(topology)
    nnuc = len(topology.nuclear_charges)
    coordinates = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
    rng = np.random.default_rng(932)
    density = jnp.asarray(rng.normal(size=(2, topology.nao, topology.nao)))
    direction = jnp.asarray(rng.normal(size=coordinates.shape)) * .17

    def at(coords):
        return replace(parameters, nuclear_coords=coords[:nnuc], centers=coords[nnuc:])

    def actual(coords, dm, *, with_j=True, with_k=True, cutoff=0.):
        return plan.get_jk(at(coords), dm, screening_threshold=cutoff,
                           workers=2, with_j=with_j, with_k=with_k)

    def oracle(coords, dm, *, with_j=True, with_k=True):
        eri = plan.evaluate("eri", at(coords))
        return (jnp.einsum("pqrs,brs->bpq", eri, dm) if with_j else jnp.zeros_like(dm),
                jnp.einsum("prqs,brs->bpq", eri, dm) if with_k else jnp.zeros_like(dm))

    return plan, parameters, coordinates, density, direction, at, actual, oracle


def _forbid_eri(monkeypatch):
    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == "eri":
            raise AssertionError("The tested direct geometry action must not build AO ERIs")
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, "evaluate", checked)


@pytest.mark.parametrize("with_j,with_k", [(True, True), (True, False), (False, True)])
def test_direct_geometry_jvp_vjp_without_ao_eri(problem, monkeypatch, with_j, with_k):
    _, _, coords, density, direction, _, actual, oracle = problem
    action = lambda r, d: actual(r, d, with_j=with_j, with_k=with_k)
    reference = lambda r, d: oracle(r, d, with_j=with_j, with_k=with_k)
    cot = (jnp.cos(density), jnp.sin(density))
    loss = lambda operation, r, d: sum(jnp.sum(x*c) for x, c in zip(operation(r, d), cot))
    expected_jvp = jax.jvp(lambda r: reference(r, density), (coords,), (direction,))[1]
    expected_vjp = jax.grad(lambda r, d: loss(reference, r, d), (0, 1))(coords, density)
    _forbid_eri(monkeypatch)
    observed_jvp = jax.jit(lambda r, u: jax.jvp(lambda q: action(q, density), (r,), (u,))[1])(
        coords, direction)
    observed_vjp = jax.jit(jax.grad(lambda r, d: loss(action, r, d), (0, 1)))(coords, density)
    for observed, expected in zip(jax.tree.leaves((observed_jvp, observed_vjp)),
                                  jax.tree.leaves((expected_jvp, expected_vjp))):
        np.testing.assert_allclose(observed, expected, atol=3e-10, rtol=2e-10)


def test_direct_mixed_density_geometry_hvp_and_finite_difference(problem, monkeypatch):
    _, _, coords, density, direction, _, actual, oracle = problem
    density_direction = .13 * jnp.sin(density)

    def loss(operation, r, d):
        coulomb, exchange = operation(r, d)
        return .7*jnp.sum(coulomb**2) + .3*jnp.sum(exchange**2)

    point, tangent = (coords, density), (direction, density_direction)
    expected_gradient = jax.grad(lambda r, d: loss(oracle, r, d), (0, 1))
    expected = jax.jvp(expected_gradient, point, tangent)[1]
    _forbid_eri(monkeypatch)
    gradient = jax.jit(jax.grad(lambda r, d: loss(actual, r, d), (0, 1)))
    observed = jax.jit(lambda p, v: jax.jvp(gradient, p, v)[1])(point, tangent)
    step = 1e-4
    plus = gradient(coords+step*direction, density+step*density_direction)
    minus = gradient(coords-step*direction, density-step*density_direction)
    finite_difference = jax.tree.map(lambda a, b: (a-b)/(2*step), plus, minus)
    for result, reference, fd in zip(observed, expected, finite_difference):
        np.testing.assert_allclose(result, reference, atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(result, fd, atol=2e-7, rtol=3e-7)


def test_direct_geometry_translation_sum_rule(problem):
    _, _, coords, density, _, _, actual, _ = problem
    tangent = jax.jit(lambda r: jax.jvp(lambda q: actual(q, density),
                                       (r,), (jnp.ones_like(r),))[1])(coords)
    for matrix in tangent:
        np.testing.assert_allclose(matrix, 0., atol=3e-11)


def test_direct_geometry_retains_requested_workers(problem, monkeypatch):
    from gradscf.integrals.backends.native import jk
    from gradscf.integrals.backends.native.autodiff import direct_jk as geometry
    plan, _, coords, density, _, at, _, _ = problem
    original = geometry.geometry_products
    observed = []

    def products(*args, **kwargs):
        observed.append(kwargs.get("workers"))
        return original(*args, **kwargs)

    jk._direct_geometry_value.cache_clear()
    monkeypatch.setattr(geometry, "geometry_products", products)

    def energy(r):
        coulomb, exchange = plan.get_jk(at(r), density, workers=3)
        return jnp.sum(coulomb**2)+jnp.sum(exchange**2)

    assert np.isfinite(jax.jit(jax.grad(energy))(coords)).all()
    assert observed and all(worker == 3 for worker in observed)


def test_traced_geometry_cache_factory_uses_direct_response(problem, monkeypatch):
    plan, _, coords, density, direction, at, _, oracle = problem
    expected = jax.jvp(lambda r: oracle(r, density), (coords,), (direction,))[1]
    _forbid_eri(monkeypatch)

    def operation(r):
        provider = NativeDirectBasis(plan, at(r))
        return provider.prepare_jk(cache_bytes=provider.eri_cache_nbytes)(density)

    observed = jax.jit(lambda r, u: jax.jvp(operation, (r,), (u,))[1])(coords, direction)
    for result, reference in zip(observed, expected):
        np.testing.assert_allclose(result, reference, atol=3e-10, rtol=2e-10)


def test_screened_geometry_direction_is_explicitly_rejected(problem):
    _, _, coords, density, direction, _, actual, _ = problem
    with pytest.raises((NotImplementedError, ValueError, RuntimeError), match="(?i)(cutoff|screen)"):
        jax.jvp(lambda r: actual(r, density, cutoff=1e-12), (coords,), (direction,))


def test_real_third_direct_geometry_response_remains_explicitly_unsupported(problem):
    _, _, coords, density, direction, _, actual, _ = problem
    first = lambda r: jax.jvp(lambda q: actual(q, density), (r,), (direction,))[1]
    second = lambda r: jax.jvp(first, (r,), (direction,))[1]
    with pytest.raises((NotImplementedError, RuntimeError), match="(?i)third"):
        jax.block_until_ready(jax.jit(lambda r: jax.jvp(second, (r,), (direction,))[1])(coords))


@pytest.mark.parametrize("field", ["exponents", "coefficients"])
def test_direct_basis_parameter_derivatives_remain_explicitly_unsupported(problem, field):
    plan, parameters, _, density, _, _, _, _ = problem
    values = getattr(parameters, field)
    with pytest.raises((NotImplementedError, ValueError, RuntimeError),
                       match="(?i)(basis|exponent|coefficient)"):
        jax.jvp(lambda value: plan.get_jk(replace(parameters, **{field: (value, *values[1:])}), density),
                (values[0],), (jnp.ones_like(values[0]),))


def test_projected_direct_geometry_and_transform_mixed_hvp(monkeypatch):
    from gradscf.integrals.backends.native.jk import ProjectedNativeDirectBasis
    from gradscf.integrals.basis.contraction import primitive_basis, contraction_matrix

    topology, parameters = integrals.prepare_basis("H 0 0 0; H .2 -.1 .8", "sto-3g")
    primitive_topology, primitive_parameters = primitive_basis(topology, parameters)
    plan = integrals.make_plan(primitive_topology)
    transform = contraction_matrix(topology, parameters)
    nnuc = len(primitive_topology.nuclear_charges)
    coordinates = jnp.concatenate((primitive_parameters.nuclear_coords, primitive_parameters.centers))
    density = jnp.array([[.8, -.2], [.3, .6]])

    def at(coords):
        return replace(primitive_parameters, nuclear_coords=coords[:nnuc], centers=coords[nnuc:])

    def loss(r, t, *, oracle):
        if oracle:
            eri = plan.evaluate("eri", at(r))
            dm = t @ density @ t.T
            coulomb = t.T @ jnp.einsum("pqrs,rs->pq", eri, dm) @ t
            exchange = t.T @ jnp.einsum("prqs,rs->pq", eri, dm) @ t
        else:
            provider = ProjectedNativeDirectBasis(NativeDirectBasis(plan, at(r)), t)
            coulomb, exchange = provider.get_jk(density)
        return .7*jnp.sum(coulomb**2) + .3*jnp.sum(exchange**2)

    direction = (jnp.asarray(np.random.default_rng(933).normal(size=coordinates.shape))*.07,
                 jnp.ones_like(transform)*.03)
    point = coordinates, transform
    reference_gradient = jax.grad(lambda r, t: loss(r, t, oracle=True), (0, 1))
    expected = jax.jvp(reference_gradient, point, direction)[1]
    _forbid_eri(monkeypatch)
    gradient = jax.jit(jax.grad(lambda r, t: loss(r, t, oracle=False), (0, 1)))
    observed = jax.jit(lambda p, u: jax.jvp(gradient, p, u)[1])(point, direction)
    step = 1e-4
    plus = gradient(*(value+step*delta for value, delta in zip(point, direction)))
    minus = gradient(*(value-step*delta for value, delta in zip(point, direction)))
    for result, reference, a, b in zip(observed, expected, plus, minus):
        np.testing.assert_allclose(result, reference, atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(result, (a-b)/(2*step), atol=2e-7, rtol=3e-7)
