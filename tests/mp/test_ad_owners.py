"""MP2 response budgets retain the actual first/second saved-array owners."""
from dataclasses import replace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import mp
from gradscf._memory import tree_storage_bytes
from gradscf.mp.integrals import from_orbitals
from gradscf.mp.mp2 import evaluate_mp2
from gradscf.mp._memory import _channel_bytes, plan_mp2


def _problem(kind='blocks', dtype=jnp.float64, occupied=(3, 2), nmo=6):
    rng = np.random.default_rng(6742)
    factors = rng.normal(scale=.02, size=(5, nmo, nmo))
    factors = jnp.asarray(factors + factors.transpose(0, 2, 1), dtype)
    coefficients = (jnp.eye(nmo, dtype=dtype), jnp.eye(nmo, dtype=dtype))
    focks = tuple(jnp.diag(jnp.asarray([-1.] * no + [.6] * (nmo-no)))
                  for no in occupied)
    eri = jnp.einsum('Qpq,Qrs->pqrs', factors, factors, precision='highest')
    source = dict(eri=eri) if kind == 'blocks' else dict(df_factors=factors)
    data = from_orbitals(coefficients, focks, occupied, -3., **source)
    point = (data.focks, data.blocks if kind == 'blocks' else data.factors)
    tangent = jax.tree.map(lambda a: jnp.full_like(a, .01), point)
    return data, point, tangent


def _minimum_budget(observable, point, upper_bytes=2_000_000):
    low, high = 1, upper_bytes
    while high-low > 1:
        middle = (low+high)//2
        try:
            jax.eval_shape(lambda p: observable(p, middle/1e6), point)
        except ValueError as error:
            assert any(word in str(error) for word in ('memory', 'budget', 'resident'))
            low = middle
        else:
            high = middle
    return (high+128)/1e6


def test_plan_covers_actual_first_and_second_saved_owners():
    data, point, tangent = _problem()
    config = mp.MPConfig(orbital_basis='noncanonical', with_t2=False)

    for with_t2 in (False, True):
        options = replace(config, with_t2=with_t2)

        def energy(values):
            fs, blocks = values
            inputs = data._replace(focks=fs, blocks=blocks)
            return evaluate_mp2(inputs, options).total_energy

        value, pullback = jax.vjp(energy, point)
        saved = jax.tree.leaves(pullback)
        gradient = pullback(jnp.ones_like(value))[0]
        _, second = jax.vjp(jax.grad(energy), point)
        second_saved = jax.tree.leaves(second)
        hvp = jax.jvp(jax.grad(energy), (point,), (tangent,))[1]
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves((value, gradient, hvp)))
        charged = plan_mp2(data, options, 2., 0)
        assert charged >= tree_storage_bytes(data)+tree_storage_bytes(saved)
        assert charged >= tree_storage_bytes(data)+tree_storage_bytes(second_saved)


@pytest.mark.parametrize('with_t2', [False, True])
def test_forward_only_budget_cannot_hide_second_response_owners(with_t2):
    data, point, _ = _problem()
    config = mp.MPConfig(orbital_basis='noncanonical', with_t2=with_t2)
    virtual = tuple(e.size-no for e, no in zip(data.energies, data.nocc))
    output, work = _channel_bytes(data.nocc, virtual, config, jnp.dtype('float64'))
    # This is the original forward-only capacity: it contains neither the
    # simultaneous saved spin-channel tapes nor directional response outputs.
    maximum = (tree_storage_bytes(data)+output+work+1)/.8/1e6

    def energy(values):
        focks, blocks = values
        return evaluate_mp2(data._replace(focks=focks, blocks=blocks), config,
                            max_memory=maximum).total_energy

    with pytest.raises(ValueError, match='memory|budget|resident'):
        jax.vjp(jax.grad(energy), point)


def test_active_scalar_channel_reserves_the_shared_inverse_minimum(monkeypatch):
    module = importlib.import_module('gradscf.solvers.linear.tensor_sum')
    observed = []

    def record(maximum, **kwargs):
        observed.append(kwargs['resident_bytes'])
        return 1_000_000

    monkeypatch.setattr(module, 'workspace_bytes', record)
    factors = tuple(jnp.ones((1, 1))*(k+1) for k in range(4))
    result = module.solve_tensor_sum(factors, jnp.ones((1,)*4), max_memory=1.)
    assert result.converged and bool(jnp.all(jnp.isfinite(result.solution)))
    config = mp.MPConfig(orbital_basis='noncanonical', with_t2=False)
    _, working = _channel_bytes((1,), (1,), config, jnp.dtype('float64'))
    assert len(observed) == 1 and working >= observed[0]


@pytest.mark.parametrize('with_t2', [False, True])
def test_cached_budget_rejects_missing_owners_and_preserves_second_response(with_t2):
    data, point, direction = _problem()
    config = mp.MPConfig(orbital_basis='noncanonical', with_t2=with_t2)

    def observable(values, maximum):
        focks, blocks = values
        result = evaluate_mp2(data._replace(focks=focks, blocks=blocks), config,
                              max_memory=maximum)
        value = result.total_energy
        if with_t2:
            # The requested amplitudes actually participate in the observable.
            value += .01*sum(jnp.sum(t*jnp.arange(t.size).reshape(t.shape)/t.size)
                            for t in result.t2)
        return value

    # Find the smallest admitted array plan from public behavior. This does not
    # depend on private profile constants or the chosen internal plan type.
    maximum = _minimum_budget(observable, point)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        jax.eval_shape(lambda p: observable(p, maximum-.000130), point)
    limited = lambda p: observable(p, maximum)
    roomy = lambda p: observable(p, 2.)
    tight_graph, roomy_graph = jax.make_jaxpr(limited)(point), jax.make_jaxpr(roomy)(point)
    assert not any(eq.primitive.name.startswith('remat') for eq in tight_graph.eqns)
    assert not any(eq.primitive.name.startswith('remat') for eq in roomy_graph.eqns)
    bundle = lambda f: jax.jit(lambda p: jax.jvp(jax.value_and_grad(f),
                                               (p,), (direction,)))(point)
    for actual, expected in zip(jax.tree.leaves(bundle(limited)), jax.tree.leaves(bundle(roomy))):
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)
    _, tight_saved = jax.vjp(jax.grad(limited), point)
    _, roomy_saved = jax.vjp(jax.grad(roomy), point)
    assert tree_storage_bytes(jax.tree.leaves(tight_saved)) == tree_storage_bytes(jax.tree.leaves(roomy_saved))


@pytest.mark.parametrize('kind', ['blocks', 'factors'])
def test_mixed_integral_precision_keeps_complete_outputs_and_response(kind):
    data, point, direction = _problem(kind, jnp.float32)
    config = mp.MPConfig(orbital_basis='noncanonical', with_t2=True)

    def result(values, maximum):
        focks, two_body = values
        updates = dict(blocks=two_body) if kind == 'blocks' else dict(factors=two_body)
        return evaluate_mp2(data._replace(focks=focks, **updates), config, max_memory=maximum)

    def observable(values, maximum):
        state = result(values, maximum)
        return state.total_energy+.01*sum(jnp.sum(t) for t in state.t2)

    maximum = _minimum_budget(observable, point)
    # Full outputs/JVP and the scalar T2-dependent gradient/HVP are different
    # contracts; testing only total energy could discard required amplitudes.
    actual = jax.jit(lambda p: jax.jvp(lambda x: result(x, maximum), (p,), (direction,)))(point)
    expected = jax.jit(lambda p: jax.jvp(lambda x: result(x, None), (p,), (direction,)))(point)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert a.dtype == e.dtype
        if jnp.issubdtype(a.dtype, jnp.inexact):
            assert np.isfinite(a).all() and np.isfinite(e).all()
            np.testing.assert_allclose(a, e, atol=3e-6, rtol=3e-6)
        else:
            np.testing.assert_array_equal(a, e)
    bundle = lambda limit: jax.jit(lambda p: jax.jvp(jax.value_and_grad(
        lambda x: observable(x, limit)), (p,), (direction,)))(point)
    for a, e in zip(jax.tree.leaves(bundle(maximum)), jax.tree.leaves(bundle(None))):
        assert a.dtype == e.dtype and np.isfinite(a).all() and np.isfinite(e).all()
        np.testing.assert_allclose(a, e, atol=3e-6, rtol=3e-6)


def test_static_empty_spin_channels_do_not_enter_an_inverse(monkeypatch):
    module = importlib.import_module('gradscf.mp.mp2')
    data, point, _ = _problem(occupied=(1, 0))
    monkeypatch.setattr(module, 'solve_tensor_sum', lambda *a, **k: pytest.fail(
        'An empty/same-spin one-electron channel entered the inverse'))
    config = mp.MPConfig(orbital_basis='noncanonical', with_t2=True)
    result = evaluate_mp2(data, config, max_memory=1.)
    assert result.valid and result.correlation_energy == 0.
    assert result.min_abs_denominator == jnp.inf
    assert all(np.isfinite(t).all() and not np.any(t) for t in result.t2)
    derivative = jax.grad(lambda p: evaluate_mp2(data._replace(focks=p[0], blocks=p[1]),
                                                config, max_memory=1.).total_energy)(point)
    assert all(np.isfinite(x).all() and not np.any(x) for x in jax.tree.leaves(derivative))


def test_budgeted_channel_preserves_invalid_denominator_and_reference_guards():
    data, point, _ = _problem()
    config = mp.MPConfig(orbital_basis='noncanonical', with_t2=True)
    maximum = _minimum_budget(lambda p, limit: evaluate_mp2(
        data._replace(focks=p[0], blocks=p[1]), config, max_memory=limit).total_energy, point)
    invalid = (data._replace(focks=tuple(jnp.zeros_like(f) for f in data.focks)),
               data._replace(reference_error=jnp.asarray(1.)))
    for inputs in invalid:
        result = evaluate_mp2(inputs, config, max_memory=maximum)
        assert not result.valid and np.isnan(result.total_energy)
        assert all(np.isnan(t).all() for t in result.t2)


