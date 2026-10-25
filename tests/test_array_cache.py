"""Byte limits concern static array payload, not Python keys or XLA caches."""
import gc
import importlib
import weakref

import numpy as np


def test_array_cache_keeps_lru_and_entry_limit_compatibility():
    cache = importlib.import_module('gradscf._cache').array_cache

    @cache(maxsize=2, max_bytes=24)
    def values(n):
        return np.zeros(n, np.uint8)

    first = values(8)
    assert values(8) is first
    victim = weakref.ref(values(12))
    assert values(8) is first
    values(10)
    gc.collect()
    assert victim() is None
    assert values.cache_bytes() == 18
    info = values.cache_info()
    assert tuple(info) == (2, 3, 2, 2)


def test_oversize_payload_is_not_retained_or_allowed_to_flush_small_entries():
    cache = importlib.import_module('gradscf._cache').array_cache

    @cache(maxsize=4, max_bytes=16)
    def values(n):
        return np.zeros(n, np.uint8)

    small = weakref.ref(values(8))
    large = weakref.ref(values(32))
    gc.collect()
    assert small() is not None and large() is None
    assert values.cache_bytes() == 8
    values.cache_clear()
    gc.collect()
    assert small() is None
    assert tuple(values.cache_info()) == (0, 0, 4, 0)
    assert values.cache_bytes() == 0


def test_factory_payload_has_an_explicit_owner_weight():
    cache = importlib.import_module('gradscf._cache').array_cache
    owners = []

    @cache(maxsize=4, max_bytes=16, payload_bytes=lambda _function, n: n)
    def factory(n):
        owned = np.zeros(n, np.uint8)
        owners.append(weakref.ref(owned))
        return lambda: owned

    first = weakref.ref(factory(12))
    factory(8)
    gc.collect()
    assert first() is None and owners[0]() is None
    assert owners[1]() is not None and factory.cache_bytes() == 8


def test_ci_fci_array_owners_and_cached_action_have_finite_payload_caps():
    direct = importlib.import_module('gradscf.ci.direct')
    hamiltonian = importlib.import_module('gradscf.ci.hamiltonian')
    strings = importlib.import_module('gradscf.fci.cistring')
    density = importlib.import_module('gradscf.ci._rdm')
    from gradscf import ci
    from gradscf._memory import tree_storage_bytes
    functions = (direct._choices, direct._topology, direct._annihilation_map,
                 direct._action, hamiltonian._connections, strings._link_maps,
                 density._density_connections)
    assert sum(f.cache_parameters()['max_bytes'] for f in functions) <= 288_000_000
    for function in functions:
        function.cache_clear()
    space = ci.make_ci_space(3, 1)
    topology = direct._topology(space)
    action = direct._action(space, 2, (), None, None)
    assert callable(action)
    assert direct._action.cache_bytes() >= tree_storage_bytes(topology) > 0
    for function in functions:
        assert function.cache_bytes() <= function.cache_parameters()['max_bytes']
