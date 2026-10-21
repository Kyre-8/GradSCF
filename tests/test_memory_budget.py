"""Memory plans use decimal MB and leave space for the runtime."""
from types import SimpleNamespace

import numpy as np
import pytest


def test_default_total_budget_is_sixty_decimal_gigabytes():
    from gradscf._memory import DEFAULT_MAX_MEMORY_MB
    assert DEFAULT_MAX_MEMORY_MB * 1_000_000 == 60_000_000_000


@pytest.mark.parametrize("value", [0., -1., float("inf"), float("nan"), True, "60"])
def test_invalid_memory_budget_fails_before_planning(value):
    from gradscf._memory import validate_memory_mb
    with pytest.raises(ValueError, match="max_memory"):
        validate_memory_mb(value)


def test_working_arrays_share_runtime_headroom_and_resident_charge(monkeypatch):
    from gradscf import _memory
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda device=None: None)
    assert _memory.workspace_bytes(60000., resident_bytes=2_000_000_000) == 46_000_000_000
    with pytest.raises(ValueError, match="resident"):
        _memory.workspace_bytes(1., resident_bytes=900_000)


def test_planning_uses_the_actual_device_without_initializing_other_devices():
    from gradscf._memory import effective_memory_mb
    device = SimpleNamespace(platform="gpu", memory_stats=lambda: {"bytes_limit": 24_000_000_000})
    assert effective_memory_mb(60000., device=device) == 24000.
    assert effective_memory_mb(8000., device=device) == 8000.


def test_device_without_memory_statistics_preserves_explicit_budget():
    from gradscf._memory import effective_memory_mb
    device = SimpleNamespace(platform="gpu", memory_stats=lambda: None)
    assert effective_memory_mb(4096., device=device) == 4096.


def test_active_device_allocations_reduce_available_capacity():
    from gradscf._memory import effective_memory_mb
    device = SimpleNamespace(platform="gpu", memory_stats=lambda: {
        "bytes_limit": 24_000_000_000, "bytes_in_use": 20_000_000_000})
    assert effective_memory_mb(60000., device=device) == 4000.


def test_resident_arrays_are_counted_once_without_copying():
    from gradscf._memory import array_storage_bytes
    array = np.empty((4, 7), dtype=np.float64)
    assert array_storage_bytes(array, None, array) == 224


def test_shape_only_abstract_arrays_are_charged_without_materialization():
    from gradscf._memory import array_storage_bytes
    abstract = SimpleNamespace(shape=(3, 5), dtype=np.dtype("float64"))
    assert array_storage_bytes(abstract, abstract) == 120
