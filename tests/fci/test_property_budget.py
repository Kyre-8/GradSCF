"""FCI property tiles share their source/state reservation."""
import importlib
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import fci
from gradscf._memory import tree_storage_bytes


@pytest.mark.parametrize('block', [False, True])
def test_property_tiles_charge_the_caller_before_enumerating_links(block, monkeypatch):
    module = importlib.import_module('gradscf.fci.rdm')
    space = fci.make_fci_space(2, (1, 1))

    def forbidden(*args, **kwargs):
        pytest.fail('An exhausted property budget reached link enumeration')

    monkeypatch.setattr(module, '_link_maps', forbidden)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        if block:
            module.trans_rdm2_block(jnp.ones(space.shape), jnp.ones(space.shape), space,
                ((0,),)*4, max_memory=.01, reserved_bytes=10000)
        else:
            module.trans_rdm12(jnp.ones(space.shape), jnp.ones(space.shape), space,
                               max_memory=.01, reserved_bytes=10000)


@pytest.mark.parametrize('name', ['make_rdm1', 'make_rdm12', 'make_rdm2_block'])
def test_bound_property_retains_integrals_reference_and_other_roots(name, monkeypatch):
    api = importlib.import_module('gradscf.fci.api')
    reference = fci.FCIReference(jnp.eye(2), jnp.zeros((2,)*4), (1, 1))
    method = fci.FCI(reference, max_memory=32.)
    vector = jnp.ones((2, 2))
    space = fci.make_fci_space(2, (1, 1))
    method._inputs = (jnp.ones((2, 2)), jnp.ones((2,)*4), 2, (1, 1), jnp.asarray(.4))
    method.result = SimpleNamespace(coefficients=vector, retained=jnp.ones(7))
    method._property_inputs = lambda *args: (vector, space)
    expected = tree_storage_bytes(reference, method._inputs, method.result, exclude=(vector,))
    captured = []

    def checked(*args, **kwargs):
        captured.append(kwargs['reserved_bytes'])
        return vector

    monkeypatch.setattr(api.rdm, name, checked)
    if name == 'make_rdm2_block':
        method.make_rdm2_block(((0,),)*4)
    else:
        getattr(method, name)()
    assert captured == [expected]


@pytest.mark.parametrize('array_kind', ['jax', 'numpy'])
def test_active_preparation_keeps_other_source_arrays_reserved(array_kind, monkeypatch):
    module = importlib.import_module('gradscf.fci.reference')
    original_budget = module.workspace_bytes
    cast = jnp.asarray if array_kind == 'jax' else np.asarray
    coefficients = cast(np.eye(2))
    inputs = SimpleNamespace(overlap=cast(np.eye(2)), hcore=cast(np.diag([-1., .5])),
        eri=cast(np.zeros((2,)*4)), nuclear_repulsion=cast(.7))
    source = SimpleNamespace(mo_coeff=coefficients, _scf_inputs=inputs,
        params={'held': cast(np.ones(100))}, max_memory=32.)
    # A backend may alias NumPy data; conservatively keep that caller storage
    # while the child owns its numerical JAX inputs. JAX leaves are aliases.
    expected = (100+4)*8 if array_kind == 'jax' else (100+4+4+4+16+1)*8
    captured = []

    def checked(*args, **kwargs):
        captured.append(kwargs.get('reserved_bytes', 0))
        return args[0], kwargs['eri'], kwargs['ecore']

    monkeypatch.setattr(module, '_active_from_orbitals', checked)
    module.active_reference(source, (), (0, 1))
    assert captured == [expected]
    residents = []

    def budget(*args, **kwargs):
        residents.append(kwargs['resident_bytes'])
        return 1_000_000

    monkeypatch.setattr(module, 'workspace_bytes', budget)
    monkeypatch.setattr(module, 'fold_core', lambda h,g,**kw: (h,g,kw['ecore']))
    explicit = fci.FCIReference(inputs.hcore, inputs.eri, (1, 1), inputs.nuclear_repulsion)
    module.active_reference(explicit, (), (0, 1))
    assert residents == [(4+16+1)*8+(0 if array_kind == 'jax' else (4+16)*8)]
    monkeypatch.setattr(module, 'workspace_bytes', original_budget)

    def forbidden():
        pytest.fail('A retained source exhausted the budget before legacy packed ERI construction')

    source._scf_inputs = SimpleNamespace(overlap=inputs.overlap, hcore=inputs.hcore,
        eri=None, nuclear_repulsion=inputs.nuclear_repulsion, response_eri_pair_matrix=forbidden)
    with pytest.raises(ValueError, match='memory|budget'):
        module.active_reference(source, (), (0, 1), max_memory=.001)
