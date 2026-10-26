"""Standalone canonical MP3 shares selected CC preparation and its old action."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import mp
from gradscf.mp import mp3
from _orbital_frame import ao_source, orbital_frame, raw_oracle


def _entry():
    entry = getattr(mp3, 'run_from_orbitals', None)
    assert callable(entry), 'Missing selected standalone canonical MP3 entry'
    return entry


def _assert_tree(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert a.dtype == b.dtype
        np.testing.assert_allclose(a, b, atol=3e-11, rtol=3e-9)


def _problem(kind, frozen, with_t2):
    entry = _entry()
    config = mp.MPConfig(order=3, algorithm='direct', with_t2=with_t2)

    def result(parameter, selected):
        c, fs, energy, factors, eri = orbital_frame(parameter, canonical=True)
        if selected:
            return entry(c, fs, 2, energy, frozen=frozen, config=config,
                         max_memory=64., **ao_source(kind, factors, eri))
        h, g, enuc = raw_oracle(c, fs, energy, eri)
        return mp3.evaluate_mp3(h, g, nocc=2, frozen=frozen, nuclear_repulsion=enuc,
                                 config=config, max_memory=64.)

    def evaluate(selected):
        energy = lambda x: result(x, selected).total_energy
        return jax.jit(lambda x: jax.jvp(jax.value_and_grad(energy),
            (x,), (jnp.ones_like(x),)))(0.)

    return entry, config, result, evaluate


@pytest.mark.parametrize('kind,frozen,with_t2', [
    ('array', None, True), ('s4', 1, False), ('df', [0, 3], True),
])
def test_selected_mp3_keeps_raw_canonical_energy_amplitudes_and_joint_response(
        kind, frozen, with_t2, monkeypatch):
    entry, config, result, evaluate = _problem(kind, frozen, with_t2)
    expected, expected_response = result(0., False), evaluate(False)

    def forbidden(*args, **kwargs):
        pytest.fail('Selected MP3 re-entered a full-MO/raw-integral preparer')

    monkeypatch.setattr(mp3, 'prepare_integrals', forbidden)
    monkeypatch.setattr(mp3, 'prepare_mp_integrals', forbidden)
    actual, response = result(0., True), evaluate(True)
    assert bool(actual.valid) and abs(float(actual.e3)) > 1e-14
    assert (actual.t2 is not None) == with_t2
    _assert_tree(actual, expected)
    _assert_tree(response, expected_response)


def test_mp3_doubles_reservation_precedes_selected_transform(monkeypatch):
    from gradscf.cc import integrals
    entry = _entry()
    c, fs, energy, factors, _ = orbital_frame(0., canonical=True)
    original = integrals.transform_eri_blocks
    reservations = []

    def observe(*args, **kwargs):
        reservations.append(kwargs['reserved_bytes'])
        return original(*args, **kwargs)

    monkeypatch.setattr(integrals, 'transform_eri_blocks', observe)
    result = entry(c, fs, 2, energy, df_factors=factors, max_memory=64.,
                   config=mp.MPConfig(order=3, with_t2=True))
    assert bool(result.valid) and reservations
    assert min(reservations) >= 2 * result.t2.nbytes


def test_mp3_minimum_budget_rejects_before_selected_preparation(monkeypatch):
    from gradscf.cc import integrals
    entry = _entry()
    c, fs, energy, _, eri = orbital_frame(0., canonical=True)

    def forbidden(*args, **kwargs):
        pytest.fail('An impossible MP3 doubles/source budget reached preparation')

    monkeypatch.setattr(integrals, 'from_orbitals', forbidden)
    with pytest.raises(ValueError, match='memory'):
        entry(c, fs, 2, energy, eri=eri, max_memory=1e-5,
              config=mp.MPConfig(order=3))


def test_mp3_full_fock_guard_is_not_lost_when_freezing():
    entry = _entry()
    c, fs, energy, _, eri = orbital_frame(0., canonical=True)
    bad = (fs[0].at[0, 1].set(.02).at[1, 0].set(.02),)
    result = entry(c, bad, 2, energy, frozen=1, eri=eri,
                   config=mp.MPConfig(order=3))
    assert not bool(result.valid)
    assert np.isnan(result.correlation_energy)


def test_standalone_mp3_rejects_unrestricted_and_noncanonical_input():
    entry = _entry()
    c, fs, energy, _, eri = orbital_frame(0., canonical=True)
    config = mp.MPConfig(order=3)
    with pytest.raises(NotImplementedError, match='restricted'):
        entry((c, c), (fs[0], fs[0]), (2, 1), energy, eri=eri, config=config)
    with pytest.raises((ValueError, NotImplementedError), match='canonical'):
        entry(c, fs, 2, energy, eri=eri,
              config=replace(config, orbital_basis='noncanonical'))


def test_selected_mp3_matches_independent_taylor_third_order():
    from gradscf.mp.series import run_series
    c, fs, energy, factors, eri = orbital_frame(0., canonical=True)
    h, g, enuc = raw_oracle(c, fs, energy, eri)
    config = mp.MPConfig(order=3, with_t2=True)
    actual = _entry()(c, fs, 2, energy, df_factors=factors, config=config, max_memory=64.)
    expected = run_series(h, g, nocc=2, nuclear_repulsion=enuc,
        config=replace(config, algorithm='series'), max_memory=64.)
    assert actual.valid and expected.valid
    np.testing.assert_allclose(actual.corrections, expected.corrections, atol=3e-11, rtol=3e-9)
    np.testing.assert_allclose(actual.t2, expected.t2, atol=3e-11, rtol=3e-9)


def test_reference_precision_does_not_promote_mp3_interaction_or_amplitudes():
    c, fs, energy, _, eri = orbital_frame(0., canonical=True)
    c, fs, eri = c.astype(jnp.float32), tuple(f.astype(jnp.float32) for f in fs), eri.astype(jnp.float32)
    energy = jnp.asarray(energy, dtype=jnp.float64)
    h, g, nuclear = raw_oracle(c, fs, energy, eri)
    config = mp.MPConfig(order=3, with_t2=True)
    expected = mp3.evaluate_mp3(h, g, nocc=2, nuclear_repulsion=nuclear, config=config)
    actual = _entry()(c, fs, 2, energy, eri=eri, config=config, max_memory=64.)
    assert actual.valid and expected.valid
    assert actual.reference_energy.dtype == jnp.float64
    assert actual.t2.dtype == expected.t2.dtype == jnp.float32
    # The raw Fock/energy reconstruction itself has float32 input precision.
    # The existing all-float64 cases keep their original tighter thresholds.
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert a.dtype == b.dtype
        np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-5)


def test_mp3_promotion_forecast_rejects_before_actual_cast(monkeypatch):
    c, fs, energy, _, eri = orbital_frame(0., canonical=True)
    c, fs = c.astype(jnp.float32), tuple(f.astype(jnp.float32) for f in fs)
    original = jnp.asarray
    class CastTrap(jax.ShapeDtypeStruct):
        def astype(self, *args, **kwargs):
            pytest.fail('MP3 cast a full Fock before its memory preflight')
    fock = CastTrap(fs[0].shape, fs[0].dtype)
    def asarray(value, *args, **kwargs):
        return fock if value is fs[0] else original(value, *args, **kwargs)
    monkeypatch.setattr(mp3.jnp, 'asarray', asarray)
    with pytest.raises(ValueError, match='memory'):
        _entry()(c, fs, 2, energy, eri=eri, config=mp.MPConfig(order=3), max_memory=1e-5)
