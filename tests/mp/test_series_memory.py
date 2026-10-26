"""One byte budget for Taylor state, requested output and existing solves."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import mp
from gradscf.mp import series
from _orbital_frame import orbital_frame, raw_oracle


@pytest.mark.parametrize('maximum', [0., -1., True, float('inf')])
def test_series_rejects_invalid_memory_budget(maximum):
    c, fs, energy, _, eri = orbital_frame(0.)
    h, g, enuc = raw_oracle(c, fs, energy, eri)
    with pytest.raises(ValueError, match='max_memory'):
        series.run_series(h, g, nocc=2, nuclear_repulsion=enuc,
            config=mp.MPConfig(order=4, algorithm='series', orbital_basis='noncanonical'),
            max_memory=maximum)


def test_series_minimum_budget_precedes_action_construction(monkeypatch):
    from gradscf.ci import hamiltonian
    c, fs, energy, _, eri = orbital_frame(0.)
    h, g, enuc = raw_oracle(c, fs, energy, eri)

    def forbidden(*args, **kwargs):
        raise AssertionError('An impossible state/source budget must fail before action construction')

    monkeypatch.setattr(hamiltonian, 'build_fluctuation', forbidden)
    with pytest.raises(ValueError, match='memory'):
        series.run_series(h, g, nocc=2, nuclear_repulsion=enuc,
            config=mp.MPConfig(order=4, algorithm='series', orbital_basis='noncanonical'),
            max_memory=1e-5)


def test_series_budget_precedes_dense_tensor_maps(monkeypatch):
    from gradscf.mp import _tensor
    c, fs, energy, _, eri = orbital_frame(0., canonical=True)
    h, g, enuc = raw_oracle(c, fs, energy, eri)
    original = _tensor.make_tensor_space
    materialize = getattr(_tensor, '_with_dense_maps', None)
    calls = []

    def observe(*args, **kwargs):
        calls.append(kwargs.get('packed_only', False))
        if not calls[-1]:
            pytest.fail('Dense maps were materialized before the memory preflight')
        return original(*args, **kwargs)

    monkeypatch.setattr(_tensor, 'make_tensor_space', observe)
    monkeypatch.setattr(_tensor, '_with_dense_maps', lambda *a: pytest.fail(
        'Dense maps were materialized before the memory preflight'), raising=False)
    config = mp.MPConfig(order=4, algorithm='tensor', with_coefficients=True)
    with pytest.raises(ValueError, match='memory'):
        series.run_series(h, g, nocc=2, nuclear_repulsion=enuc,
            config=config, max_memory=1e-5)
    assert calls == [True]
    calls.clear()
    actual = series.run_series(h, g, nocc=2, nuclear_repulsion=enuc,
        config=config, max_memory=.1)
    assert calls == [True] and bool(actual.valid)
    monkeypatch.setattr(_tensor, 'make_tensor_space', original)
    if materialize is not None:
        monkeypatch.setattr(_tensor, '_with_dense_maps', materialize)
    expected = series.run_series(h, g, nocc=2, nuclear_repulsion=enuc,
        config=config)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, atol=2e-12, rtol=2e-10)


def test_series_budget_uses_the_promoted_spin_history_dtype():
    from gradscf.mp._series_memory import plan_series
    from gradscf.mp.integrals import _reference_data
    ha = jnp.diag(jnp.array([-1., .5, .9], dtype=jnp.float32))
    hb = ha.astype(jnp.float64)
    aa = jnp.zeros((3,) * 4, dtype=jnp.float32)
    bb = aa.astype(jnp.float64)
    hs, gs, focks, energy = _reference_data((ha, hb), (aa, aa, bb), (1, 1))
    config = mp.MPConfig(order=4, algorithm='series', orbital_basis='noncanonical', with_t2=False)
    space = series.make_series_space(3, (1, 1), config=config)
    small = plan_series(hs, gs, focks, space, (1, 1), ((), ()), config, 1.)
    full = plan_series(hs, gs, focks, space, (1, 1), ((), ()),
        replace(config, with_t2=True, with_coefficients=True), 1.)
    assert focks[0].dtype == jnp.float32 and energy.dtype == jnp.float64
    # Three spin T2 blocks have four elements each; the three wave vectors
    # use the actual promoted reference-energy dtype, not alpha's dtype.
    output_bytes = (12 + 3 * space.size) * energy.dtype.itemsize
    assert full.resident_bytes >= small.resident_bytes + output_bytes


def test_requested_outputs_are_reserved_in_the_existing_linear_solve(monkeypatch):
    c, fs, energy, _, eri = orbital_frame(0.)
    h, g, enuc = raw_oracle(c, fs, energy, eri)
    original = series.solve_linear
    reservations = []

    def observe(*args, **kwargs):
        assert kwargs['max_memory'] == 1.
        reservations.append(kwargs['reserved_bytes'])
        return original(*args, **kwargs)

    monkeypatch.setattr(series, 'solve_linear', observe)
    config = mp.MPConfig(order=4, algorithm='series', orbital_basis='noncanonical', with_t2=False)
    records = []
    for requested in (False, True):
        series._run_series.clear_cache()
        reservations.clear()
        result = series.run_series(h, g, nocc=2, nuclear_repulsion=enuc,
            config=replace(config, with_t2=requested, with_coefficients=requested), max_memory=1.)
        assert bool(result.valid) and reservations
        records.append((result, max(reservations)))
    small, full = records[0][0], records[1][0]
    assert small.t2 is None and small.wavefunction_coefficients is None
    output_bytes = full.t2.nbytes + full.wavefunction_coefficients.nbytes
    assert records[1][1] >= records[0][1] + output_bytes
    np.testing.assert_allclose(full.corrections, small.corrections, atol=2e-12, rtol=2e-10)


def test_series_budget_choices_preserve_nested_response():
    config = mp.MPConfig(order=4, algorithm='series', orbital_basis='noncanonical', with_t2=False)

    def evaluate(maximum):
        def energy(parameter):
            c, fs, e0, _, eri = orbital_frame(parameter)
            h, g, enuc = raw_oracle(c, fs, e0, eri)
            return series.run_series(h, g, nocc=2, nuclear_repulsion=enuc,
                config=config, max_memory=maximum).total_energy
        return jax.jit(lambda x: jax.jvp(jax.value_and_grad(energy),
            (x,), (jnp.ones_like(x),)))(0.)

    for actual, expected in zip(jax.tree.leaves(evaluate(1.)),
                                jax.tree.leaves(evaluate(60000.))):
        np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-10)
