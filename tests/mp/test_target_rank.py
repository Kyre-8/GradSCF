"""Exact output projection must precede high-rank tensor contractions."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import mp
from gradscf.mp._tensor import TensorAction, build_tensor_action, make_tensor_space
from gradscf.mp.integrals import _reference_data
from test_tensor_action import model


@pytest.fixture
def cold_series_kernel():
    from gradscf.mp.series import _run_series
    _run_series.clear_cache()
    yield _run_series
    _run_series.clear_cache()


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("target", [0, 1, 2])
def test_target_projection_matches_full_action_and_state_response(unrestricted, target):
    h, g, nocc = model(unrestricted)
    _, _, fs, _ = _reference_data(h, g, nocc)
    space = make_tensor_space(4, nocc, max_excitation=4)
    ranks = jnp.asarray(space.ranks)
    mask = ranks <= target
    rng = np.random.default_rng(2323)
    vector = jnp.asarray(rng.normal(size=space.size))
    direction = jnp.asarray(rng.normal(size=space.size))*.03

    def value(c, projected):
        action = build_tensor_action(g, fs, space)
        return (action.apply(c, max_output_rank=target) if projected
                else jnp.where(mask, action(c), 0.))

    for projected in (False, True):
        fn = lambda c: value(c, projected)
        scalar = lambda c: jnp.sum(fn(c)**2)
        bundle = lambda c: (fn(c), jax.jvp(fn, (c,), (direction,))[1],
                            jax.grad(scalar)(c),
                            jax.jvp(jax.grad(scalar), (c,), (direction,))[1])
        actual = jax.jit(bundle)(vector)
        if not projected:
            expected = actual
        else:
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-11)


def test_target_limit_skips_high_rank_contraction_before_allocation(monkeypatch):
    h, g, nocc = model(False)
    _, _, fs, _ = _reference_data(h, g, nocc)
    space = make_tensor_space(4, nocc, max_excitation=4)
    action = build_tensor_action(g, fs, space)
    original = jnp.tensordot
    visited = []

    def bounded(*args, **kwargs):
        value = original(*args, **kwargs)
        visited.append(value.ndim)
        assert value.ndim <= 4, "A discarded triples/quadruples target was contracted"
        return value

    monkeypatch.setattr(jnp, "tensordot", bounded)
    result = action.apply(jnp.ones(space.size), max_output_rank=2)
    assert visited
    np.testing.assert_array_equal(result[jnp.asarray(space.ranks) > 2], 0.)


@pytest.mark.parametrize("order,unrestricted,target", [(2, False, 0), (4, False, 2), (4, True, 2)])
def test_taylor_terminal_action_receives_exact_bra_rank(
        monkeypatch, order, unrestricted, target, cold_series_kernel):
    h, g, nocc = model(unrestricted)
    hs, _, fs, _ = _reference_data(h, g, nocc)
    canonical = tuple(a-(f-jnp.diag(jnp.diag(f))) for a, f in zip(hs, fs))
    h = canonical if unrestricted else canonical[0]
    original = TensorAction.apply
    requests = []

    def record(self, vector, **kwargs):
        requests.append(kwargs)
        return original(self, vector, **kwargs)

    monkeypatch.setattr(TensorAction, "apply", record)
    result = mp.run_mp(h, g, nocc=nocc,
        config=mp.MPConfig(order=order, algorithm="tensor", with_t2=False))
    assert bool(result.valid)
    k = order//2
    assert requests[k].get("max_output_rank") == target


@pytest.mark.parametrize("nmo,nocc,frozen,order,wave_order", [
    (6, 3, None, 6, None), (6, (3, 2), ([0], [5]), 5, None), (4, 2, None, 3, 2),
], ids=["sextuples", "open-shell-frozen", "extra-waves"])
def test_required_high_rank_waves_and_coefficients_are_unchanged(
        monkeypatch, nmo, nocc, frozen, order, wave_order, cold_series_kernel):
    rng = np.random.default_rng(2324)
    factors = rng.normal(size=(3, nmo, nmo))*.015
    factors += factors.transpose(0, 2, 1)
    g = jnp.asarray(np.einsum("Lpq,Lrs->pqrs", factors, factors))
    gs = (g, g*.97, g*1.03) if isinstance(nocc, tuple) else g
    h = jnp.diag(jnp.arange(nmo, dtype=float)*.6-2.)
    hs = (h, h+.02*jnp.eye(nmo)) if isinstance(nocc, tuple) else h
    hs, _, fs, _ = _reference_data(hs, gs, nocc)
    canonical = tuple(a-(f-jnp.diag(jnp.diag(f))) for a, f in zip(hs, fs))
    h = canonical if isinstance(nocc, tuple) else canonical[0]
    cfg = mp.MPConfig(order=order, algorithm="tensor", with_coefficients=True,
                      wavefunction_order=wave_order)
    def evaluate():
        # Trace the private kernel afresh for each physical/oracle action.
        cold_series_kernel.clear_cache()
        return jax.jit(lambda hh, gg: mp.run_mp(hh, gg, nocc=nocc,
            frozen=frozen, config=cfg))(h, gs)
    actual = evaluate()
    original = TensorAction.apply
    reference_calls = []

    def full_targets(self, vector, *, max_input_rank=None, max_output_rank=None):
        reference_calls.append(None)
        return original(self, vector, max_input_rank=max_input_rank)

    monkeypatch.setattr(TensorAction, "apply", full_targets)
    expected = evaluate()
    assert reference_calls, "The full-target oracle was hidden by a stale JIT"
    assert bool(actual.valid & expected.valid)
    np.testing.assert_allclose(actual.corrections, expected.corrections, atol=2e-12, rtol=2e-10)
    np.testing.assert_allclose(actual.wavefunction_coefficients,
                               expected.wavefunction_coefficients, atol=2e-12, rtol=2e-10)
