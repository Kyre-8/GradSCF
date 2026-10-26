"""Selected AO inputs feed the existing noncanonical Taylor recurrence."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import mp
from gradscf.mp import series
from _orbital_frame import ao_source, orbital_frame, raw_oracle


def _entry():
    entry = getattr(series, 'run_from_orbitals', None)
    assert callable(entry), 'Missing selected noncanonical MP-series entry'
    return entry


def _assert_tree(actual, expected, atol=3e-10):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert a.dtype == b.dtype
        np.testing.assert_allclose(a, b, atol=atol, rtol=3e-9)


def _problem(unrestricted, algorithm, kind):
    entry = _entry()
    nocc = (2, 1) if unrestricted else 2
    config = mp.MPConfig(order=4, algorithm=algorithm, orbital_basis='noncanonical',
                          with_coefficients=True)

    def result(parameter, selected):
        c, fs, energy, factors, eri = orbital_frame(parameter, unrestricted=unrestricted)
        if selected:
            return entry(c, fs, nocc, energy, config=config,
                         max_memory=64., **ao_source(kind, factors, eri))
        h, g, enuc = raw_oracle(c, fs, energy, eri, unrestricted=unrestricted)
        return series.run_series(h, g, nocc=nocc, nuclear_repulsion=enuc,
                                 config=config, max_memory=64.)

    def evaluate(selected):
        def energy(x):
            primal = result(x, selected)
            return primal.total_energy, primal
        ((value, primal), gradient), ((direction, _), hvp) = jax.jit(
            lambda x: jax.jvp(jax.value_and_grad(energy, has_aux=True),
                (x,), (jnp.ones_like(x),)))(0.)
        return primal, (value, gradient, direction, hvp)

    return entry, nocc, config, result, evaluate


@pytest.mark.parametrize('unrestricted,algorithm,kind', [
    (False, 'series', 'df'), (True, 'tensor', 's4'),
], ids=['restricted-df', 'unrestricted-s4'])
def test_selected_series_joint_frame_fock_energy_source_response(
        unrestricted, algorithm, kind, monkeypatch):
    entry, nocc, config, result, evaluate = _problem(unrestricted, algorithm, kind)
    expected, expected_response = evaluate(False)

    def forbidden(*args, **kwargs):
        pytest.fail('Selected MP series rebuilt a complete MO reference')

    from gradscf import ao2mo
    monkeypatch.setattr(ao2mo, 'transform_integrals', forbidden)
    monkeypatch.setattr(ao2mo, 'transform_unrestricted_integrals', forbidden)
    monkeypatch.setattr(ao2mo, 'spin_orbital_integrals', forbidden)
    actual, actual_response = evaluate(True)
    assert bool(actual.valid)
    assert actual.t2 is not None and actual.wavefunction_coefficients is not None
    _assert_tree(actual, expected)
    _assert_tree(actual_response, expected_response)
    c, fs, e0, factors, eri = orbital_frame(0., unrestricted=unrestricted)
    compact = entry(c, fs, nocc, e0, config=replace(config, with_t2=False,
        with_coefficients=False), max_memory=64., **ao_source(kind, factors, eri))
    assert compact.t2 is None and compact.wavefunction_coefficients is None
    np.testing.assert_allclose(compact.corrections, actual.corrections, atol=2e-12, rtol=2e-10)


def test_selected_series_separate_frozen_spins_preserve_full_reference():
    entry = _entry()
    c, fs, energy, _, eri = orbital_frame(0., unrestricted=True)
    # Frozen boundaries are isolated here; the unfrozen response tests above
    # retain exact internal degeneracy and live off-diagonal Fock tangents.
    fs = tuple(f + jnp.diag(jnp.arange(4) * .11) for f in fs)
    h, g, enuc = raw_oracle(c, fs, energy, eri, unrestricted=True)
    frozen, config = ([0], [3]), mp.MPConfig(order=4, algorithm='series',
        orbital_basis='noncanonical', with_coefficients=True)
    expected = series.run_series(h, g, nocc=(2, 1), frozen=frozen,
        nuclear_repulsion=enuc, config=config, max_memory=64.)
    actual = entry(c, fs, (2, 1), energy, frozen=frozen, config=config,
                   eri=eri, max_memory=64.)
    assert bool(actual.valid)
    _assert_tree(actual, expected)


def test_selected_series_budget_precedes_preparation(monkeypatch):
    from gradscf.ci import integrals
    entry = _entry()
    c, fs, energy, _, eri = orbital_frame(0.)

    def forbidden(*args, **kwargs):
        pytest.fail('An impossible MP history/input budget reached integral preparation')

    monkeypatch.setattr(integrals, 'from_orbitals', forbidden)
    with pytest.raises(ValueError, match='memory'):
        entry(c, fs, 2, energy, eri=eri, max_memory=1e-5,
              config=mp.MPConfig(order=4, algorithm='series', orbital_basis='noncanonical'))


def test_selected_series_does_not_silently_change_canonical_raw_chart():
    entry = _entry()
    c, fs, energy, _, eri = orbital_frame(0., canonical=True)
    with pytest.raises((ValueError, NotImplementedError), match='noncanonical'):
        entry(c, fs, 2, energy, eri=eri, config=mp.MPConfig(order=4, algorithm='series'))
