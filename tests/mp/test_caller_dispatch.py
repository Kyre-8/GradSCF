"""Raw and eager MP entry points keep the selected chart and one budget."""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import mp
from gradscf._memory import tree_storage_bytes
from gradscf.mp import series
from gradscf.scf.reference import RestrictedReference
from _orbital_frame import orbital_frame, raw_oracle
from test_selected_series_geometry import source


@pytest.mark.parametrize('order,algorithm,target', [
    (2, 'direct', 'evaluate_mp2'), (3, 'direct', 'evaluate_mp3'), (4, 'series', 'run_series')])
def test_raw_dispatch_preserves_budget_arguments(order, algorithm, target, monkeypatch):
    core = importlib.import_module('gradscf.mp.mp2')
    module = (core if order == 2 else importlib.import_module(
        'gradscf.mp.mp3' if order == 3 else 'gradscf.mp.series'))
    class Reached(Exception):
        pass
    h, g = jnp.diag(jnp.array([-1., 1.])), jnp.zeros((2,)*4)
    def observe(*args, **kwargs):
        expected = 4096+(tree_storage_bytes(h, g) if order == 2 else 0)
        assert kwargs['max_memory'] == 2. and kwargs['reserved_bytes'] == expected
        raise Reached
    monkeypatch.setattr(module, target, observe)
    with pytest.raises(Reached):
        core.run_mp(h, g, nocc=1,
                    config=mp.MPConfig(order=order, algorithm=algorithm),
                    max_memory=2., reserved_bytes=4096)


def test_raw_series_exhausted_caller_reservation_precedes_actions():
    c, fs, e, _, eri = orbital_frame(0.)
    h, g, nuclear = raw_oracle(c, fs, e, eri)
    with pytest.raises(ValueError, match='memory|resident|budget'):
        series.run_series(h, g, nocc=2, nuclear_repulsion=nuclear,
            config=mp.MPConfig(order=4, algorithm='series', orbital_basis='noncanonical'),
            max_memory=1., reserved_bytes=2_000_000)


def test_reference_energy_shift_preserves_interaction_dtype_and_response():
    import jax
    c, fs, _, _, eri = orbital_frame(0.)
    c, fs, eri = c.astype(jnp.float32), tuple(f.astype(jnp.float32) for f in fs), eri.astype(jnp.float32)
    energy = jnp.asarray(-2.4, jnp.float64)
    config = mp.MPConfig(order=4, algorithm='series', orbital_basis='noncanonical',
                         with_coefficients=True)
    actual = series.run_from_orbitals(c, fs, 2, energy, eri=eri, config=config, max_memory=64.)
    h, g, nuclear = raw_oracle(c, fs, energy, eri)
    expected = series.run_series(h, g, nocc=2, nuclear_repulsion=nuclear, config=config)
    assert actual.valid and expected.valid
    assert actual.corrections.dtype == actual.wavefunction_coefficients.dtype == jnp.float32
    assert actual.total_energy.dtype == actual.reference_energy.dtype == jnp.float64
    np.testing.assert_allclose(actual.corrections, expected.corrections, atol=3e-10, rtol=3e-5)
    np.testing.assert_allclose(actual.wavefunction_coefficients,
                               expected.wavefunction_coefficients, atol=3e-8, rtol=3e-5)
    def shift(value):
        return series.run_from_orbitals(c, fs, 2, value, eri=eri, config=config).total_energy
    differentiated = jax.jit(jax.value_and_grad(shift))
    value, derivative = differentiated(energy)
    # Independent float32 contraction graphs can differ by their last bit.
    np.testing.assert_allclose(value, actual.total_energy, atol=3e-10, rtol=0.)
    shifted, _ = differentiated(energy+.25)
    np.testing.assert_allclose(shifted-value, .25, atol=0., rtol=0.)
    np.testing.assert_allclose(derivative, 1., atol=0., rtol=0.)


@pytest.mark.parametrize('source', ['h2', 'h3'], indirect=True)
def test_scf_dispatch_uses_selected_entries_and_keeps_inheritance(source, monkeypatch):
    api = importlib.import_module('gradscf.mp.api')
    from gradscf.ao2mo.scf import prepare_scf_source
    from gradscf.mp import mp3
    frame = prepare_scf_source(source, max_memory=16.)
    unrestricted = isinstance(frame.nocc, tuple)
    configs = [mp.MPConfig(order=4, algorithm='tensor' if unrestricted else 'series',
                           orbital_basis='noncanonical', with_t2=False)]
    if not unrestricted:
        configs.append(mp.MPConfig(order=3, algorithm='direct', with_t2=False))
    expected = [((series.run_from_orbitals if cfg.order == 4 else mp3.run_from_orbitals)(
        frame.coefficients, frame.focks, frame.nocc, frame.reference_energy,
        config=cfg, max_memory=16., **frame.ao_kwargs)) for cfg in configs]
    def forbidden(*a, **k):
        pytest.fail('Selected eager MP rebuilt a full MO reference')
    monkeypatch.setattr(api, 'reference_from_source', forbidden)
    monkeypatch.setattr(api, 'unrestricted_reference_from_source', forbidden)
    for cfg, reference in zip(configs, expected):
        model = mp.MP(source, order=cfg.order, algorithm=cfg.algorithm,
            orbital_basis=cfg.orbital_basis, with_t2=False, max_memory=16.).run()
        assert model.max_memory == 16. and model.converged
        np.testing.assert_allclose(model.e_tot, reference.total_energy, atol=3e-11, rtol=3e-10)
    assert mp.MP(source).max_memory == source.max_memory


@pytest.mark.parametrize('source', ['h2', 'h3'], indirect=True)
def test_legacy_reference_wrapper_preserves_caller_reservation(source, monkeypatch):
    module = importlib.import_module('gradscf.scf.reference')
    unrestricted = source.mo_coeff.ndim == 3
    class Reached(Exception):
        pass
    def observe(*a, **kwargs):
        assert kwargs['max_memory'] == 16. and kwargs['reserved_bytes'] == 4096
        raise Reached
    monkeypatch.setattr(module, 'transform_unrestricted_integrals' if unrestricted
                        else 'transform_integrals', observe)
    with pytest.raises(Reached):
        (module.unrestricted_reference_from_source if unrestricted else module.reference_from_source)(
            source, max_memory=16., reserved_bytes=4096)


def test_eager_tensor_metadata_preflights_then_restores_small_expand(monkeypatch):
    from gradscf.mp import _tensor
    c, fs, e, _, eri = orbital_frame(0., canonical=True)
    h, g, nuclear = raw_oracle(c, fs, e, eri)
    reference = RestrictedReference(h, g, 2, nuclear)
    original = _tensor._with_dense_maps
    monkeypatch.setattr(_tensor, '_with_dense_maps', lambda *a: pytest.fail(
        'Eager tensor maps preceded the memory preflight'))
    with pytest.raises(ValueError, match='memory|resident|budget'):
        mp.MP(reference, order=4, algorithm='tensor', max_memory=1e-5).run()
    monkeypatch.setattr(_tensor, '_with_dense_maps', original)
    model = mp.MP(reference, order=4, algorithm='tensor', max_memory=64.).run()
    vector = jnp.zeros(model.space.size)
    assert all(value is not None for value in model.space.expand(vector))
    assert model.space.amplitudes(vector).shape == model.t2.shape
