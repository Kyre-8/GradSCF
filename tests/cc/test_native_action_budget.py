"""Measured native worker policy and dense-first restricted preparation."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.ao2mo import VirtualERI
from gradscf.cc.integrals import from_orbitals
from test_native_virtual_action import _problem


@pytest.mark.parametrize('available,expected', [(1_000_000, 4), (100_000, 1)])
def test_native_action_charges_selected_workers_before_density(available, expected, monkeypatch):
    module = importlib.import_module('gradscf.ao2mo.actions')
    _, basis, c, _ = _problem()
    seen = []
    monkeypatch.setattr(module, 'workspace_bytes', lambda *args, **kwargs: available)
    monkeypatch.setattr(module, 'native_jk_workspace_bytes',
                        lambda basis, workers, cutoff: 20_000 if workers == 1 else 200_000)
    def record(self, density, **kwargs):
        seen.append(kwargs['workers'])
        return None, jnp.zeros_like(density)
    monkeypatch.setattr(type(basis), 'get_jk', record)
    jax.make_jaxpr(lambda rhs: VirtualERI(c, basis, max_memory=32.).contract(rhs))(
        jnp.ones((5, 3, 3)))
    assert seen and set(seen) == {expected}


def test_native_action_rejects_below_one_worker_minimum_before_density(monkeypatch):
    module = importlib.import_module('gradscf.ao2mo.actions')
    _, basis, c, _ = _problem()
    monkeypatch.setattr(module, 'workspace_bytes', lambda *args, **kwargs: 19_999)
    monkeypatch.setattr(module, 'native_jk_workspace_bytes',
                        lambda basis, workers, cutoff: 20_000*workers)
    monkeypatch.setattr(type(basis), 'get_jk',
                        lambda *args, **kwargs: pytest.fail('Insufficient action budget called J/K'))
    with pytest.raises(ValueError, match='memory|budget'):
        VirtualERI(c, basis, max_memory=32.).contract(jnp.ones((3, 3)))


@pytest.mark.parametrize('memory,compact', [(4., False), (.5, True)])
def test_restricted_auto_uses_dense_when_it_fits_and_exact_action_under_pressure(memory, compact):
    _, basis, _, _ = _problem()
    rng = np.random.default_rng(6560)
    # Algebraic rectangular frame stresses v^4 ownership; not a molecular
    # performance benchmark or a claim about a physical virtual dimension.
    c = jnp.asarray(rng.normal(scale=.08, size=(basis.nao, 16)))
    fock = jnp.diag(jnp.linspace(-1.1, 1.3, 16))
    dense = from_orbitals(c, fock, 1, jnp.asarray(-1.), direct_basis=basis, max_memory=4.)
    actual = from_orbitals(c, fock, 1, jnp.asarray(-1.), direct_basis=basis,
                           max_memory=memory, vvvv='auto')
    assert isinstance(actual.vvvv, VirtualERI) == compact
    for name in dense._fields:
        if name != 'vvvv':
            np.testing.assert_allclose(getattr(actual, name), getattr(dense, name), atol=2e-12)
    rhs = jnp.asarray(rng.normal(scale=.03, size=(2, 15, 15)))
    if compact:
        result = jax.jit(lambda x: actual.vvvv.contract(x))(rhs)
    else:
        result = jnp.einsum('acbd,...cd->...ab', actual.vvvv, rhs)
    np.testing.assert_allclose(result, jnp.einsum('acbd,...cd->...ab', dense.vvvv, rhs),
                               atol=2e-12, rtol=2e-11)


def test_restricted_ccs_auto_retains_no_unconsumed_four_virtual_tensor():
    _, basis, c, _ = _problem()
    ints = from_orbitals(c, jnp.diag(jnp.asarray([-1., .4, .8])), 1, jnp.asarray(-1.),
                         direct_basis=basis, max_memory=.5, model='ccs', vvvv='auto')
    assert ints.oooo is None and ints.vvvv is None
