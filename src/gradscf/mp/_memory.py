"""Known MP2 spin-channel arrays and tensor-resolvent workspaces."""
from math import prod

import jax
import jax.numpy as jnp

from .._memory import tree_storage_bytes, validate_reserved_bytes, workspace_bytes
from ..ao2mo import frozen_indices, unrestricted_frozen_indices


def _channel_bytes(ns, virtual, config, dtype):
    pairs = ((0, 0),) if len(ns) == 1 else ((0, 0), (0, 1), (1, 1))
    shapes = tuple((ns[a], ns[b], virtual[a], virtual[b]) for a, b in pairs)
    output = sum(prod(shape) for shape in shapes)*dtype.itemsize if config.with_t2 else 0
    if config.orbital_basis == 'noncanonical':
        work = max((20*prod(shape)+4*sum(n*n for n in shape) for shape in shapes), default=0)
    else:
        work = max((16*prod(shape[1:]) if shape[0] else 0 for shape in shapes), default=0)
    return 2*output, work*dtype.itemsize


def preparation_bytes(nmo, nocc, frozen, config, dtype):
    """Reserve the same requested outputs/work before integral preparation."""
    ns = nocc if isinstance(nocc, tuple) else (nocc,)
    fr = (unrestricted_frozen_indices(nmo, nocc, frozen) if isinstance(nocc, tuple)
          else (frozen_indices(nmo, nocc, frozen),))
    occupied = tuple(no-sum(i < no for i in f) for no, f in zip(ns, fr))
    virtual = tuple(nmo-no-sum(i >= no for i in f) for no, f in zip(ns, fr))
    return sum(_channel_bytes(occupied, virtual, config, jnp.dtype(dtype)))


def plan_mp2(integrals, config, max_memory, reserved_bytes):
    """Keep all requested T2 outputs live while budgeting one physical channel.

    Canonical work scales with one occupied row; noncanonical work retains
    the existing complete tensor inverse. Compiler/allocator and arbitrary
    nested derivative state require separate measurement and reservations.
    """
    reserved = validate_reserved_bytes(reserved_bytes)
    if max_memory is None:
        return reserved
    ns = integrals.nocc
    virtual = tuple(e.size-no for e, no in zip(integrals.energies, ns))
    dtype = jnp.result_type(*jax.tree.leaves((integrals.energies, integrals.focks,
                                             integrals.blocks, integrals.factors)))
    # Keep the channel amplitudes while constructing their validity-masked
    # return arrays; releasing that Python list is not an AD lifetime promise.
    output, work = _channel_bytes(ns, virtual, config, dtype)
    resident = reserved+tree_storage_bytes(integrals)+output
    workspace_bytes(max_memory, resident_bytes=resident+work,
                    device=getattr(integrals.focks[0], 'device', None))
    return resident
