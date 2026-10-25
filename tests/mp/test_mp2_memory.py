"""MP2 spin channels and the tensor inverse share one original MB budget."""
import importlib
import inspect
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.mp import MPConfig
from gradscf.mp.integrals import prepare_integrals
from test_noncanonical import _model


def _module():
    module = importlib.import_module('gradscf.mp.mp2')
    assert 'max_memory' in inspect.signature(module.evaluate_mp2).parameters
    return module


@pytest.mark.parametrize('orbital_basis', ['canonical', 'noncanonical'])
def test_mp2_exhaustion_precedes_every_spin_channel(orbital_basis, monkeypatch):
    module = _module()
    h, g, no = _model(True)
    ints = prepare_integrals(h, g, nocc=no)
    forbidden = lambda *a, **k: pytest.fail('Exhausted MP2 budget reached an amplitude channel')
    monkeypatch.setattr(module, '_channel', forbidden)
    monkeypatch.setattr(module, '_noncanonical_channel', forbidden)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        module.evaluate_mp2(ints, MPConfig(orbital_basis=orbital_basis),
                             max_memory=.0001, reserved_bytes=4096)


def test_retained_t2_includes_all_three_spin_channels(monkeypatch):
    module = _module()
    memory = importlib.import_module('gradscf.mp._memory')
    h, g, no = _model(True)
    ints = prepare_integrals(h, g, nocc=no)
    calls = []

    def record(maximum, **kwargs):
        calls.append(kwargs['resident_bytes'])
        return 1_000_000

    monkeypatch.setattr(memory, 'workspace_bytes', record)
    config = MPConfig(with_t2=False)
    module.evaluate_mp2(ints, config, max_memory=2., reserved_bytes=4096)
    module.evaluate_mp2(ints, replace(config, with_t2=True), max_memory=2., reserved_bytes=4096)
    # Channel results remain live while validity-masked return arrays are made.
    expected = 2*sum(block.size for block in ints.blocks)*ints.focks[0].dtype.itemsize
    assert len(calls) == 2 and calls[1]-calls[0] == expected


@pytest.mark.parametrize('unrestricted', [False, True])
def test_budgeted_mp2_preserves_degenerate_joint_response(unrestricted, monkeypatch):
    module = _module()
    h, g, no = _model(unrestricted, degenerate=True)
    ns = no if unrestricted else (no,)
    directions = []
    for occupied in ns:
        direction = np.random.default_rng(6564+occupied).normal(scale=.02, size=(4, 4))
        direction = .5*(direction+direction.T)
        direction[:occupied, occupied:] = direction[occupied:, :occupied] = 0.
        directions.append(jnp.asarray(direction))
    directions = tuple(directions) if unrestricted else directions[0]
    config = MPConfig(orbital_basis='noncanonical', with_t2=True)
    original = module.solve_tensor_sum
    calls = []

    def checked(*args, **kwargs):
        if 'max_memory' in kwargs and kwargs['max_memory'] is not None:
            calls.append((kwargs['max_memory'], kwargs.get('reserved_bytes', 0)))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'solve_tensor_sum', checked)

    def energy(x, budgeted):
        hs = jax.tree.map(lambda a, d: jnp.asarray(a)+x*d, h, directions)
        ints = prepare_integrals(hs, jax.tree.map(lambda a: jnp.asarray(a)*(1+.01*x), g), nocc=no)
        options = dict(max_memory=2., reserved_bytes=4096) if budgeted else {}
        return module.evaluate_mp2(ints, config, **options).total_energy

    bundle = lambda budgeted: jax.jit(lambda x: jax.jvp(
        jax.value_and_grad(lambda y: energy(y, budgeted)), (x,), (1.,)))(0.)
    for actual, expected in zip(jax.tree.leaves(bundle(True)), jax.tree.leaves(bundle(False))):
        assert np.isfinite(np.asarray(actual)).all()
        assert np.isfinite(np.asarray(expected)).all()
        if jax.default_backend() == 'gpu':
            assert actual.devices() == {jax.devices('gpu')[0]}
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11, equal_nan=False)
    assert calls and all(maximum == 2. and resident >= 4096 for maximum, resident in calls)
