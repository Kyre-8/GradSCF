"""Shared working-array budgets in decimal MB, without allocator changes."""
from math import isfinite, prod
from numbers import Integral, Real
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import jax


DEFAULT_MAX_MEMORY_MB = 60000.0


def validate_reserved_bytes(value, *, name="reserved_bytes"):
    """Validate a static caller reservation before combining byte budgets."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


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


def array_device(*arrays):
    """Read an eager array device or the default context without initializing it.

    A concrete ``jax.default_device`` context is required to identify a GPU
    capacity while tracing. An outer ``jit(device=...)`` does not expose its
    target to Python planners; abstract arrays alone cannot establish it.
    Known eager inputs retain priority because committed arrays can execute on
    their own device even inside a different default-device context. Captured
    constants alone do not establish an outer JIT's execution target either.
    """
    for array in arrays:
        device = getattr(array, "device", None)
        device = device() if callable(device) else device
        if device is not None:
            return device
    return jax.config.jax_default_device


def available_memory_bytes(device=None):
    """Return a conservative capacity estimate for the actual target device.

    GPU statistics describe the current allocator's capacity, not an enforced
    process limit. Querying only the supplied device avoids initializing other
    GPUs. Unsupported device statistics leave the caller's budget unchanged.
    """
    device = array_device() if device is None else device
    device = device() if callable(device) else device
    if isinstance(device, str):
        if device != "cpu":
            raise ValueError("Memory planning requires a concrete accelerator device, not a platform label.")
        platform = "cpu"
    else:
        platform = "cpu" if device is None else getattr(device, "platform", None)
        if platform is None:
            raise TypeError("Memory planning requires a device with a known platform.")
    if platform != "cpu":
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
    available = available_memory_bytes(array_device() if device is None else device)
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


def tree_storage_bytes(*trees, exclude=()) -> int:
    """Count known array owners, including ordinary reference dataclasses.

    Traverse declared dataclass fields, mappings, tuples/lists and namespaces;
    do not inspect arbitrary objects, evaluate array values or transfer devices.
    Aliases and cycles are visited once. Exclusions apply recursively to arrays
    inside the supplied containers. Distinct views remain conservative charges,
    just as in ``array_storage_bytes``; this is not allocator introspection.
    """
    def arrays(roots):
        pending, seen = list(roots), set()
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                yield value
            elif is_dataclass(value) and not isinstance(value, type):
                pending.extend(getattr(value, field.name) for field in fields(value))
            elif isinstance(value, Mapping):
                pending.extend(value.values())
            elif isinstance(value, (tuple, list)):
                pending.extend(value)
            elif isinstance(value, SimpleNamespace):
                pending.extend(vars(value).values())

    excluded = {id(value) for value in arrays(exclude)}
    return sum(array_storage_bytes(value) for value in arrays(trees)
               if id(value) not in excluded)


def jaxpr_temporary_bytes(jaxpr) -> int:
    """Conservative sum of declared primitive output arrays, without execution.

    Nested call bodies are included. This intentionally assumes no allocator
    reuse or view aliasing; opaque library scratch and compiler/allocator state
    are outside this static array estimate. Inspecting a graph initializes no
    device and imports no solver or numerical backend.
    """
    if hasattr(jaxpr, 'jaxpr'):
        jaxpr = jaxpr.jaxpr
    total = 0
    for equation in jaxpr.eqns:
        for variable in equation.outvars:
            aval = getattr(variable, 'aval', None)
            if hasattr(aval, 'shape') and hasattr(aval, 'dtype'):
                total += prod(aval.shape)*np.dtype(aval.dtype).itemsize
        for value in equation.params.values():
            values = value if isinstance(value, (tuple, list)) else (value,)
            for nested in values:
                if hasattr(nested, 'eqns') or hasattr(nested, 'jaxpr'):
                    total += jaxpr_temporary_bytes(nested)
    return total
