"""Shared working-array budgets in decimal MB, without allocator changes."""
from math import isfinite, prod
from numbers import Real
import os
from pathlib import Path

import numpy as np


DEFAULT_MAX_MEMORY_MB = 60000.0


def validate_memory_mb(value, *, name="max_memory") -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite positive number of MB.")
    return float(value)


def resolve_memory_mb(max_memory, source=None) -> float:
    """Resolve an eager caller's override or inherited decimal-MB budget."""
    if max_memory is None:
        max_memory = getattr(source, "max_memory", DEFAULT_MAX_MEMORY_MB)
    return validate_memory_mb(max_memory)


def available_memory_bytes(device=None):
    """Return a conservative capacity estimate for the actual target device.

    GPU statistics describe the current allocator's capacity, not an enforced
    process limit. Querying only the supplied device avoids initializing other
    GPUs. Unsupported device statistics leave the caller's budget unchanged.
    """
    if device is not None and getattr(device, "platform", "cpu") != "cpu":
        try:
            statistics = device.memory_stats()
        except (AttributeError, RuntimeError, NotImplementedError):
            return None
        limit = (statistics or {}).get("bytes_limit")
        if limit is None:
            return None
        # Be conservative about other live JAX arrays. Known resident arrays
        # are charged separately; that overlap is safer than ignoring usage.
        return max(0, int(limit) - int((statistics or {}).get("bytes_in_use", 0)))

    available = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    except OSError:
        try:
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            pass
    # Linux cgroup v2 jobs may have a smaller limit than the host. Do not
    # change cgroups or global JAX allocation settings to implement a budget.
    try:
        relative = next(line.split("::", 1)[1] for line in
                        Path("/proc/self/cgroup").read_text().splitlines()
                        if line.startswith("0::"))
        group = Path("/sys/fs/cgroup") / relative.lstrip("/")
        limit = (group / "memory.max").read_text().strip()
        if limit != "max":
            remaining = max(0, int(limit) - int((group / "memory.current").read_text()))
            available = remaining if available is None else min(available, remaining)
    except (OSError, StopIteration, ValueError):
        pass
    return available


def effective_memory_mb(max_memory, device=None) -> float:
    """Cap a requested budget by detected capacity; never alter its owner."""
    requested = validate_memory_mb(max_memory)
    available = available_memory_bytes(device)
    if available is not None:
        if available <= 0:
            raise ValueError("No memory is available for the requested calculation.")
        requested = min(requested, available / 1_000_000)
    return requested


def workspace_bytes(max_memory, *, resident_bytes=0, device=None) -> int:
    """Reserve 20% for runtime/model temporaries, then charge resident arrays.

    These are array-planning estimates. Arbitrary external model internals,
    compilation and allocator residency still require process monitoring.
    """
    if resident_bytes < 0:
        raise ValueError("resident_bytes must be nonnegative.")
    total = int(effective_memory_mb(max_memory, device=device) * 1_000_000)
    remaining = total - total // 5 - int(resident_bytes)
    if remaining < 0:
        raise ValueError("Known resident arrays exceed the working-memory budget.")
    return remaining


def array_storage_bytes(*arrays) -> int:
    """Count distinct array objects by static shape, without device copies."""
    seen = set()
    total = 0
    for array in arrays:
        if array is not None and id(array) not in seen:
            seen.add(id(array))
            size = getattr(array, "nbytes", None)
            if size is None and hasattr(array, "shape") and hasattr(array, "dtype"):
                size = prod(array.shape) * np.dtype(array.dtype).itemsize
            total += int(size or 0)
    return total
