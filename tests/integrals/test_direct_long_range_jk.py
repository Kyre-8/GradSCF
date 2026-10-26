"""Static-omega shell-direct J/K preserves density and coordinate response."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.backends.native.jk import (
    NativeDirectBasis, ProjectedNativeDirectBasis, _selected_jk,
)
from gradscf.integrals.plan import IntegralPlan


OMEGA = .3  # inverse Bohr; erf(omega*r12)/r12


@pytest.fixture(params=[True, False], ids=["cartesian", "spherical"])
def context(request):
    topology, parameters = integrals.prepare_basis(
        "H 0 0 0; H .2 -.1 .8", {"H": [[0, [1.1, 1.]], [2, [.7, 1.]]]},
        cart=request.param)
    plan = integrals.make_plan(topology)
    nnuc = len(topology.nuclear_charges)
    coordinates = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
    density = jnp.asarray(np.random.default_rng(961).normal(size=(2, topology.nao, topology.nao)))

    def at(coords):
        return replace(parameters, nuclear_coords=coords[:nnuc], centers=coords[nnuc:])

    def oracle(coords, dm, *, with_j=True, with_k=True):
        eri = plan.evaluate("eri", at(coords), omega=OMEGA)
        return (jnp.einsum("pqrs,brs->bpq", eri, dm) if with_j else jnp.zeros_like(dm),
                jnp.einsum("prqs,brs->bpq", eri, dm) if with_k else jnp.zeros_like(dm))

    return plan, parameters, coordinates, density, at, oracle


def forbid_eri(monkeypatch):
    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == "eri":
            raise AssertionError("Long-range direct J/K must not materialize AO ERIs or packed integrals")
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, "evaluate", checked)


@pytest.mark.parametrize("with_j,with_k", [(True, True), (True, False), (False, True)])
def test_long_range_direct_values_density_and_geometry_response(context, monkeypatch, with_j, with_k):
    plan, _, coords, density, at, oracle = context
    direction = jnp.asarray(np.random.default_rng(962).normal(size=coords.shape))*.1
    density_direction = .07*jnp.sin(density)
    reference = lambda r, d: oracle(r, d, with_j=with_j, with_k=with_k)

    def loss(function, r, d):
        j, k = function(r, d)
        return .7*jnp.sum(j*j) + .3*jnp.sum(k*k)

    expected_values = reference(coords, density)
    expected_jvp = jax.jvp(reference, (coords, density), (direction, density_direction))[1]
    reference_gradient = jax.grad(lambda r, d: loss(reference, r, d), (0, 1))
    expected_gradient = reference_gradient(coords, density)
    expected_hvp = jax.jvp(reference_gradient, (coords, density), (direction, density_direction))[1]
    forbid_eri(monkeypatch)
    action = lambda r, d: plan.get_jk(at(r), d, omega=OMEGA, workers=2,
                                      with_j=with_j, with_k=with_k)
    observed_values = jax.jit(action)(coords, density)
    observed_jvp = jax.jit(lambda r, d: jax.jvp(action, (r, d),
                                                (direction, density_direction))[1])(coords, density)
    gradient = jax.jit(jax.grad(lambda r, d: loss(action, r, d), (0, 1)))
    observed_gradient = gradient(coords, density)
    observed_hvp = jax.jit(lambda r, d: jax.jvp(gradient, (r, d),
                                                (direction, density_direction))[1])(coords, density)
    for observed, expected in zip(jax.tree.leaves((observed_values, observed_jvp, observed_gradient, observed_hvp)),
                                  jax.tree.leaves((expected_values, expected_jvp, expected_gradient, expected_hvp))):
        np.testing.assert_allclose(observed, expected, atol=3e-9, rtol=3e-9)
    step = 1e-4
    plus = gradient(coords+step*direction, density+step*density_direction)
    minus = gradient(coords-step*direction, density-step*density_direction)
    for observed, a, b in zip(observed_hvp, plus, minus):
        np.testing.assert_allclose(observed, (a-b)/(2*step), atol=2e-7, rtol=4e-7)


def test_long_range_provider_preparation_bypasses_coulomb_s8_cache(context, monkeypatch):
    plan, parameters, coords, density, _, oracle = context
    expected = oracle(coords, density)
    provider = NativeDirectBasis(plan, parameters)
    forbid_eri(monkeypatch)
    action = provider.prepare_jk(cache_bytes=provider.eri_cache_nbytes, omega=OMEGA)
    for observed, reference in zip(jax.jit(action)(density), expected):
        np.testing.assert_allclose(observed, reference, atol=3e-10, rtol=3e-10)
    for observed, reference in zip(_selected_jk(provider, density, omega=OMEGA), expected):
        np.testing.assert_allclose(observed, reference, atol=3e-10, rtol=3e-10)


def test_projected_provider_forwards_long_range_exchange(context):
    plan, parameters, _, _, _, _ = context
    transform = jnp.asarray(np.random.default_rng(963).normal(size=(plan.topology.nao, 3)))*.2
    density = jnp.asarray(np.random.default_rng(964).normal(size=(2, 3, 3)))
    primitive_density = jnp.einsum("pi,bij,qj->bpq", transform, density, transform)
    eri = plan.evaluate("eri", parameters, omega=OMEGA)
    primitive_exchange = jnp.einsum("prqs,brs->bpq", eri, primitive_density)
    expected = jnp.einsum("pi,bpq,qj->bij", transform, primitive_exchange, transform)
    provider = ProjectedNativeDirectBasis(NativeDirectBasis(plan, parameters), transform)
    j, k = _selected_jk(provider, density, omega=OMEGA, with_j=False)
    np.testing.assert_array_equal(j, 0.)
    np.testing.assert_allclose(k, expected, atol=3e-10, rtol=3e-10)


def test_legacy_provider_keeps_coulomb_signature_but_rejects_nonzero_omega():
    class Provider:
        def get_jk(self, density):
            return density, 2*density

    density = jnp.eye(2)
    j, k = _selected_jk(Provider(), density)
    np.testing.assert_array_equal(j, density)
    np.testing.assert_array_equal(k, 2*density)
    with pytest.raises(NotImplementedError, match="(?i)(omega|range|provider)"):
        _selected_jk(Provider(), density, omega=OMEGA)


def test_direct_zero_omega_is_the_ordinary_coulomb_operator(context):
    plan, parameters, _, density, _, _ = context
    expected = plan.get_jk(parameters, density)
    for observed, reference in zip(plan.get_jk(parameters, density, omega=0.), expected):
        np.testing.assert_array_equal(observed, reference)


def test_direct_omega_is_static_and_nondifferentiable(context):
    plan, parameters, _, density, _, _ = context
    function = lambda omega: sum(jnp.sum(value) for value in plan.get_jk(parameters, density, omega=omega))
    for transform in (jax.jit, jax.grad):
        with pytest.raises(NotImplementedError, match="(?i)omega.*static"):
            transform(function)(OMEGA)


@pytest.mark.parametrize("omega", [-.3, np.inf, np.nan, 1e200])
def test_invalid_direct_omega_is_rejected(omega):
    topology, parameters = integrals.prepare_basis("He 0 0 0", "sto-3g")
    with pytest.raises(ValueError, match="(?i)omega.*(finite|nonnegative)"):
        integrals.make_plan(topology).get_jk(parameters, jnp.eye(topology.nao), omega=omega)
