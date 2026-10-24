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


def _requested_workspace_bytes(max_memory):
    """Requested usable bytes after the shared reserve, without device queries."""
    total = int(validate_memory_mb(max_memory)*1_000_000)
    return total-total//5


def workspace_bytes(max_memory, *, resident_bytes=0, device=None) -> int:
    """Reserve 20% for runtime/model temporaries, then charge resident arrays.

    These are array-planning estimates. Arbitrary external model internals,
    compilation and allocator residency still require process monitoring.
    """
    if resident_bytes < 0:
        raise ValueError("resident_bytes must be nonnegative.")
    remaining = _requested_workspace_bytes(effective_memory_mb(
        max_memory, device=device))-int(resident_bytes)
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


def _primitive_byte_charges(primitive_bytes):
    """Validate a caller's static opaque-workspace declarations once per graph."""
    if primitive_bytes is None:
        return {}
    if not isinstance(primitive_bytes, Mapping) or any(
            not isinstance(name, str) for name in primitive_bytes):
        raise TypeError("primitive_bytes must map primitive names to byte counts")
    return {name: validate_reserved_bytes(value, name="primitive_bytes")
            for name, value in primitive_bytes.items()}


def jaxpr_temporary_bytes(jaxpr, *, primitive_bytes=None) -> int:
    """Conservative sum of declared primitive output arrays, without execution.

    Nested call bodies are included. This intentionally assumes no allocator
    reuse or view aliasing. An optional name-to-byte mapping charges known
    opaque scratch at each matching primitive, including nested calls. Other
    library scratch and compiler/allocator state remain outside this estimate.
    Inspecting a graph initializes no device or numerical backend.
    """
    if hasattr(jaxpr, 'jaxpr'):
        jaxpr = jaxpr.jaxpr
    charges = _primitive_byte_charges(primitive_bytes)
    total = 0
    for equation in jaxpr.eqns:
        if charges:
            total += charges.get(equation.primitive.name, 0)
        for variable in equation.outvars:
            aval = getattr(variable, 'aval', None)
            if hasattr(aval, 'shape') and hasattr(aval, 'dtype'):
                total += prod(aval.shape)*np.dtype(aval.dtype).itemsize
        for value in equation.params.values():
            values = value if isinstance(value, (tuple, list)) else (value,)
            for nested in values:
                if hasattr(nested, 'eqns') or hasattr(nested, 'jaxpr'):
                    total += jaxpr_temporary_bytes(nested, primitive_bytes=charges)
    return total


def jaxpr_live_bytes(jaxpr, *, primitive_bytes=None) -> int:
    """Bound declared arrays in program order, releasing them after last use.

    The caller owns top-level inputs. Constants, literal operands, all primitive
    outputs and returned arrays are charged here; views are not assumed to alias.
    Nested bodies include their inputs, so loop carries cannot disappear from
    the estimate. Alternative bodies share scratch, while their constant stores
    remain conservatively resident. Full scan outputs follow the outer shapes.

    This describes Jaxpr array lifetimes, not an XLA scheduling/allocator or
    process-RSS guarantee. Opaque scratch requires the same optional static
    declarations as ``jaxpr_temporary_bytes``. No numerical operation is run.
    """
    from collections import Counter

    charges = _primitive_byte_charges(primitive_bytes)
    constants, peaks = {}, {}

    def bodies(value):
        if hasattr(value, 'eqns') or hasattr(value, 'jaxpr'):
            yield getattr(value, 'jaxpr', value)
        elif isinstance(value, Mapping):
            for child in value.values():
                yield from bodies(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                yield from bodies(child)

    def size(variable):
        aval = getattr(variable, 'aval', None)
        return (prod(aval.shape)*np.dtype(aval.dtype).itemsize
                if hasattr(aval, 'shape') and hasattr(aval, 'dtype') else 0)

    def constant_bytes(graph):
        key = id(graph)
        if key not in constants:
            constants[key] = sum(size(v) for v in graph.constvars) + sum(
                constant_bytes(child) for eq in graph.eqns for child in bodies(eq.params))
        return constants[key]

    def peak(graph, include_inputs):
        key = id(graph), include_inputs
        if key in peaks:
            return peaks[key]
        declared = {id(v) for v in (*graph.invars, *graph.constvars)}
        declared.update(id(v) for eq in graph.eqns for v in eq.outvars)
        returned_literals = {id(v): size(v) for v in graph.outvars if id(v) not in declared}
        live = {id(v): size(v) for v in graph.constvars}
        permanent = set(live)
        if include_inputs:
            live.update((id(v), size(v)) for v in graph.invars)
        held = sum(live.values()) + sum(returned_literals.values())
        held += constant_bytes(graph)-sum(size(v) for v in graph.constvars)
        maximum = held
        uses = Counter(id(v) for eq in graph.eqns for v in eq.invars)
        uses.update(id(v) for v in graph.outvars)
        for equation in graph.eqns:
            children = tuple(bodies(equation.params))
            nested = max((peak(child, True) for child in children), default=0)
            literals = {id(v): size(v) for v in equation.invars
                        if id(v) not in declared and id(v) not in returned_literals}
            opaque = charges.get(equation.primitive.name, 0) if charges else 0
            maximum = max(maximum, held + sum(size(v) for v in equation.outvars)
                          + sum(literals.values()) + nested + opaque)
            for variable in equation.outvars:
                live[id(variable)] = size(variable)
                held += size(variable)
            for variable in equation.invars:
                identifier = id(variable)
                uses[identifier] -= 1
                if uses[identifier] == 0 and identifier not in permanent:
                    held -= live.pop(identifier, 0)
            for variable in equation.outvars:
                if not uses[id(variable)]:
                    held -= live.pop(id(variable), 0)
        peaks[key] = maximum
        return maximum

    return peak(getattr(jaxpr, 'jaxpr', jaxpr), False)
