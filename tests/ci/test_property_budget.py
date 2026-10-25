"""Property consumers retain their caller state under the shared byte budget."""
import importlib
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import ci
from gradscf._memory import tree_storage_bytes


def test_one_body_forwards_the_budget_to_primal_and_transpose(monkeypatch):
    module = importlib.import_module('gradscf.ci.direct')
    original = module._budgeted_columns
    calls = []

    def checked(*args, **kwargs):
        calls.append((args[5], args[6]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, '_budgeted_columns', checked)
    space = ci.make_ci_space(2, 1)
    h = jnp.asarray([[.3, .2], [-.1, .7]])
    operator = module.build_one_body(h, space, max_memory=32., reserved_bytes=128)
    vector = jnp.arange(space.size, dtype=h.dtype)+1
    operator.matvec(vector).block_until_ready()
    operator.T.matvec(vector).block_until_ready()
    assert len(calls) == 2
    assert all(maximum == 32. and reservation >= 128+h.nbytes for maximum, reservation in calls)


def test_observable_forwards_one_budget_to_hamiltonian_and_correction(monkeypatch):
    module = importlib.import_module('gradscf.ci.properties')
    calls = []
    for name in ('build_one_body', 'build_direct_hamiltonian'):
        original = getattr(module, name)

        def checked(*args, _name=name, _original=original, **kwargs):
            calls.append((_name, kwargs))
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, checked)
    space = ci.make_ci_space(2, 1)
    rng = np.random.default_rng(817)
    vector = jnp.asarray(rng.normal(size=space.size))
    h, g = jnp.asarray(rng.normal(size=(2, 2))), jnp.asarray(rng.normal(size=(2,)*4))
    value = module.expectation_value(vector, space, h, g, max_memory=32., reserved_bytes=256)
    assert np.isfinite(value)
    assert [name for name, _ in calls] == ['build_direct_hamiltonian', 'build_one_body']
    assert all(options['max_memory'] == 32. and options['reserved_bytes'] >= 256+vector.nbytes
               for _, options in calls)
    assert calls[-1][1]['reserved_bytes'] >= 256+vector.nbytes+h.nbytes+g.nbytes


def test_property_reservation_rejects_before_action_construction(monkeypatch):
    module = importlib.import_module('gradscf.ci.properties')
    space = ci.make_ci_space(2, 1)

    def forbidden(*args, **kwargs):
        pytest.fail('An exhausted property budget reached sigma construction')

    monkeypatch.setattr(module, 'build_one_body', forbidden)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        module.expectation_value(jnp.ones(space.size), space, jnp.eye(2),
                                 max_memory=.01, reserved_bytes=10000)
    with pytest.raises(ValueError, match='reserved_bytes'):
        ci.make_rdm1(jnp.ones(space.size), space, reserved_bytes=-1)


@pytest.mark.parametrize('name', ['make_rdm1', 'make_rdm2', 'make_rdm12'])
def test_facade_density_reserves_retained_source_and_other_roots(name, monkeypatch):
    module = importlib.import_module('gradscf.ci.api')
    source = SimpleNamespace(max_memory=32., held=jnp.ones(13))
    method = ci.CISD(source)
    vector = jnp.ones(4)
    method.reference = SimpleNamespace(fock=jnp.ones((2, 2)))
    method._integrals = SimpleNamespace(kernel=jnp.ones(9))
    method.result = SimpleNamespace(coefficients=vector, other=jnp.ones(5))
    method._density_vector = lambda root: vector
    method.space = ci.make_ci_space(2, 1)
    expected = tree_storage_bytes(source, method.reference, method._integrals,
                                  method.result, exclude=(vector,))
    calls = []

    def checked(c, space, **kwargs):
        calls.append(kwargs)
        return c

    monkeypatch.setattr(module, name, checked)
    getattr(method, name)()
    assert calls == [dict(max_memory=32., reserved_bytes=expected)]
