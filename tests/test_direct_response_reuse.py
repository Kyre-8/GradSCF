"""Budgeted exact integral reuse belongs to one response operator."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.tddft.response_options import plan_response_memory


@pytest.fixture(scope="module")
def system():
    topology, parameters = integrals.prepare_basis(
        "H 0 0 0; H 0 0 1.4", "6-31g", cart=False)
    basis = NativeDirectBasis(integrals.make_plan(topology), parameters)
    eri = basis.plan.evaluate("eri", parameters)
    density = jnp.asarray(np.random.default_rng(917).normal(size=(3, basis.nao, basis.nao)))
    return basis, eri, density


def test_cache_budget_reserves_native_worker_and_transpose_arrays(system):
    from gradscf.tddft.response_options import _native_direct_workspace_bytes

    basis, _, _ = system
    pairs = basis.nao * (basis.nao + 1) // 2
    cache = 8 * pairs * (pairs + 1) // 2
    assert getattr(basis, "eri_cache_nbytes", None) == cache
    plan = plan_response_memory(basis.nao, 3, 1, np.float64, 2., direct_basis=basis)
    assert plan.eri_cache_bytes == cache
    per_vector = 8 * (60 * basis.nao**2 + 4 * 3)
    native_fixed = 3*_native_direct_workspace_bytes(basis, 8, 0.)
    assert native_fixed + 2*cache + plan.batch_size * per_vector <= 800_000
    # Enough for the direct minimum, but not for the cache plus its workers.
    small = plan_response_memory(basis.nao, 3, 1, np.float64, .225, direct_basis=basis)
    assert small.eri_cache_bytes == 0
    disabled = plan_response_memory(basis.nao, 3, 1, np.float64, 2.,
                                    direct_basis=basis, integral_cache="none")
    assert disabled.eri_cache_bytes == 0


def test_cached_action_reuses_exact_s8_and_preserves_density_response(system, monkeypatch):
    basis, eri, density = system
    prepare = getattr(basis, "prepare_jk", None)
    assert callable(prepare), "The native provider must prepare one budgeted J/K action"
    calls = []
    original = type(basis.plan).evaluate

    def evaluate(plan, operator, parameters, **kwargs):
        calls.append((operator, kwargs.get("aosym")))
        return original(plan, operator, parameters, **kwargs)

    monkeypatch.setattr(type(basis.plan), "evaluate", evaluate)
    action = prepare(cache_bytes=basis.eri_cache_nbytes)
    assert calls == [("eri", "s8")]

    def oracle(d):
        return (jnp.einsum("pqrs,brs->bpq", eri, d),
                jnp.einsum("prqs,brs->bpq", eri, d))

    direction = jnp.flip(density, axis=-1) / 3
    for scale in (1., -.7):
        for actual, expected in zip(jax.jit(action)(scale * density), oracle(scale * density)):
            np.testing.assert_allclose(actual, expected, atol=3e-12)
    assert calls == [("eri", "s8")], "AX must not rebuild the integral cache"
    loss = lambda apply, d: sum(jnp.sum(x*x) for x in apply(d))
    for apply in (action, oracle):
        gradient = jax.jit(jax.grad(lambda d: loss(apply, d)))(density)
        hvp = jax.jit(lambda d: jax.jvp(jax.grad(lambda x: loss(apply, x)),
                                       (d,), (direction,))[1])(density)
        tangent = jax.jit(lambda d: jax.jvp(apply, (d,), (direction,))[1])(density)
        if apply is oracle:
            np.testing.assert_allclose(gradient, reference[0], atol=3e-11)
            np.testing.assert_allclose(hvp, reference[1], atol=3e-11)
            for actual, expected in zip(tangent, reference[2]):
                np.testing.assert_allclose(actual, expected, atol=3e-12)
        else:
            reference = gradient, hvp, tangent


def test_cache_below_admission_never_materializes_integrals(system, monkeypatch):
    basis, _, density = system
    prepare = getattr(basis, "prepare_jk", None)
    assert callable(prepare), "Missing budgeted J/K action"
    expected = basis.get_jk(density)

    def forbidden(*args, **kwargs):
        raise AssertionError("Insufficient cache budget must retain shell-direct J/K")

    monkeypatch.setattr(type(basis.plan), "evaluate", forbidden)
    action = prepare(cache_bytes=basis.eri_cache_nbytes - 1)
    for actual, reference in zip(jax.jit(action)(density), expected):
        np.testing.assert_allclose(actual, reference, atol=3e-12)


def test_cache_keeps_basis_dependence_and_strict_ad_boundary(system):
    basis, _, density = system
    assert callable(getattr(basis, "prepare_jk", None)), "Missing budgeted J/K action"

    def loss(parameters, d):
        provider = replace(basis, parameters=parameters)
        return sum(jnp.sum(x*x) for x in provider.prepare_jk(
            cache_bytes=basis.eri_cache_nbytes)(d))

    zero = jax.tree.map(jnp.zeros_like, basis.parameters)
    _, actual = jax.jit(lambda p, d: jax.jvp(loss, (p, d), (zero, jnp.ones_like(d))))(
        basis.parameters, density)
    _, expected = jax.jvp(lambda d: sum(jnp.sum(x*x) for x in basis.get_jk(d)),
                           (density,), (jnp.ones_like(density),))
    np.testing.assert_allclose(actual, expected, atol=3e-11)
    shifted = replace(basis.parameters, centers=basis.parameters.centers.at[-1, 2].add(.1))
    assert abs(float(loss(shifted, density) - loss(basis.parameters, density))) > 1e-5
    direction = replace(jax.tree.map(jnp.zeros_like, basis.parameters),
                        coefficients=jax.tree.map(jnp.ones_like, basis.parameters.coefficients))
    with pytest.raises((NotImplementedError, RuntimeError), match="(?i)(basis|exponent|coefficient)"):
        jax.block_until_ready(jax.jit(lambda p: jax.jvp(
            lambda p: loss(p, density), (p,), (direction,))[1])(basis.parameters))
    with pytest.raises((NotImplementedError, RuntimeError), match="(?i)(basis|exponent|coefficient)"):
        jax.block_until_ready(jax.jit(jax.grad(lambda coefficients: loss(
            replace(basis.parameters, coefficients=coefficients), density)))(basis.parameters.coefficients))


@pytest.mark.parametrize("full_td", [False, True])
def test_response_builder_reuses_cache_with_tail_batches(system, full_td):
    from gradscf.tddft._semilocal_response import SemilocalResponseFunctional
    from gradscf.tddft.response import build_restricted_tda_operator, build_restricted_tdhf_operator

    basis, _, _ = system
    molecule = SimpleNamespace(
        mo_coeff=jnp.eye(4), mo_occ=jnp.array([2., 2., 0., 0.]),
        mo_energy=jnp.array([-1.3, -.8, .4, .9]), nocc=2,
        ao=jnp.zeros((1, 4)), grid=SimpleNamespace(weights=jnp.ones(1)),
        rdm1=jnp.stack((jnp.diag(jnp.array([1., 1., 0., 0.])),) * 2),
        rep_tensor=None, eri_pair_matrix=None, df_factors=None, direct_basis=basis)
    builder = build_restricted_tdhf_operator if full_td else build_restricted_tda_operator
    actions = []
    for cache in ("auto", "none"):
        built = builder(molecule, SemilocalResponseFunctional("hf"),
                        response_kernel_options={"max_memory": 2., "integral_cache": cache})
        actions.append(built if full_td else built[0])
    assert actions[0].memory_plan.eri_cache_bytes == basis.eri_cache_nbytes
    assert actions[1].memory_plan.eri_cache_bytes == 0
    width = actions[0].memory_plan.batch_size
    vectors = jnp.asarray(np.random.default_rng(918).normal(size=(2*width+1, 8 if full_td else 4)))
    np.testing.assert_allclose(jax.jit(actions[0])(vectors), actions[1](vectors), atol=3e-12)
    gradients = [jax.jit(jax.grad(lambda x: jnp.sum(action(x)**2)))(vectors) for action in actions]
    np.testing.assert_allclose(*gradients, atol=3e-11)


def test_projected_and_cuda_plans_keep_direct_workspace(system, monkeypatch):
    from gradscf.integrals.backends.native.jk import ProjectedNativeDirectBasis

    basis, _, _ = system
    projected = ProjectedNativeDirectBasis(basis, jnp.eye(basis.nao))
    assert plan_response_memory(4, 3, 1, np.float64, .2,
                                direct_basis=projected).eri_cache_bytes == 0
    assert plan_response_memory(4, 3, 1, np.float64, .25,
                                direct_basis=basis,
                                device=SimpleNamespace(platform="gpu", memory_stats=lambda: {})).eri_cache_bytes == 0
