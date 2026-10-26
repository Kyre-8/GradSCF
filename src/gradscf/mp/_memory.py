"""Known MP2 spin-channel arrays and tensor-resolvent workspaces."""
from math import prod
from functools import lru_cache

import jax
import jax.numpy as jnp

from .._memory import (jaxpr_live_bytes, tree_storage_bytes,
                       validate_reserved_bytes, workspace_bytes)
from ..ao2mo import frozen_indices, unrestricted_frozen_indices
from ..solvers.linear.tensor_sum import _tensor_sum_workspace_bytes


def _channel_work(shape, dtype):
    if not all(shape):
        return 0
    return max((20*prod(shape)+4*sum(n*n for n in shape))*dtype.itemsize,
        _tensor_sum_workspace_bytes(tuple(jax.ShapeDtypeStruct((n, n), dtype) for n in shape),
                                    jax.ShapeDtypeStruct(shape, dtype)))


def _channel_bytes(ns, virtual, config, dtype):
    pairs = ((0, 0),) if len(ns) == 1 else ((0, 0), (0, 1), (1, 1))
    shapes = tuple((ns[a], ns[b], virtual[a], virtual[b]) for a, b in pairs)
    output = sum(prod(shape) for shape in shapes)*dtype.itemsize if config.with_t2 else 0
    if config.orbital_basis == 'noncanonical':
        work = max((_channel_work(shape, dtype) if (len(ns) == 1 or a != b
                    or (shape[0] >= 2 and shape[2] >= 2)) else 0
                    for (a, b), shape in zip(pairs, shapes)), default=0)
    else:
        work = max((16*prod(shape[1:]) if shape[0] else 0 for shape in shapes), default=0)*dtype.itemsize
    return 2*output, work


@lru_cache(maxsize=128)
def _response_profile(specifications, no, nj, same_spin, restricted, config):
    """Cache only integers derived from abstract first/second adjoint graphs."""
    from .mp2 import _noncanonical_channel

    left, right, block, factors = specifications
    shape = (no, nj, left.shape[0]-no, right.shape[0]-nj)
    dtype = jnp.result_type(*(x.dtype for x in jax.tree.leaves(specifications)))
    if not all(shape) or (same_spin and (no < 2 or shape[2] < 2)):
        return 0, 0, (2+(prod(shape) if config.with_t2 else 0))*dtype.itemsize

    def outputs(values):
        a, b, g, bs = values
        ss, os, _, t2, _ = _noncanonical_channel(a, b, no, nj, block=g, factors=bs,
            same_spin=same_spin, restricted=restricted, config=config)
        return ss, os, t2

    def adjoint(values, cotangent):
        _, backward = jax.vjp(outputs, values)
        return backward(cotangent)[0]

    first = jax.eval_shape(lambda x: jax.vjp(outputs, x), specifications)
    values = first[0]
    second = jax.eval_shape(lambda x, bar: jax.vjp(adjoint, x, bar), specifications, values)
    cache = max(tree_storage_bytes(jax.tree.leaves(first)),
                tree_storage_bytes(jax.tree.leaves(second)))
    working = max(jaxpr_live_bytes(jax.make_jaxpr(outputs)(specifications)),
        jaxpr_live_bytes(jax.make_jaxpr(adjoint)(specifications, values)),
        jaxpr_live_bytes(jax.make_jaxpr(lambda x, bar, dx, db:
            jax.jvp(adjoint, (x, bar), (dx, db)))(specifications, values, specifications, values)),
        _channel_work(shape, dtype))
    # The shared inverse charges the actual input/promotion dtypes, including
    # mixed Fock/integral precision. Same-spin numerator construction keeps g.
    rhs_dtype = block.dtype if block is not None else jnp.result_type(*factors)
    axes = tuple(jax.ShapeDtypeStruct((n, n), dt) for n, dt in zip(
        shape, (left.dtype, right.dtype, left.dtype, right.dtype)))
    inverse = _tensor_sum_workspace_bytes(axes, jax.ShapeDtypeStruct(shape, rhs_dtype))
    working = max(working, inverse+prod(shape)*jnp.dtype(rhs_dtype).itemsize*same_spin)
    return cache, working, tree_storage_bytes(values)


def preparation_bytes(nmo, nocc, frozen, config, dtype):
    """Reserve the same requested outputs/work before integral preparation."""
    ns = nocc if isinstance(nocc, tuple) else (nocc,)
    fr = (unrestricted_frozen_indices(nmo, nocc, frozen) if isinstance(nocc, tuple)
          else (frozen_indices(nmo, nocc, frozen),))
    occupied = tuple(no-sum(i < no for i in f) for no, f in zip(ns, fr))
    virtual = tuple(nmo-no-sum(i >= no for i in f) for no, f in zip(ns, fr))
    return sum(_channel_bytes(occupied, virtual, config, jnp.dtype(dtype)))


def plan_mp2(integrals, config, max_memory, reserved_bytes):
    """Budget required outputs and finite first/directional second response.

    Noncanonical channels retain their existing numerical trajectory and the
    simultaneous adjoint caches. A smaller budget fails before their solve;
    reduced tape size alone does not establish a smaller compiled workspace.
    No full Hessian or arbitrary outer AD graph is promised by this plan;
    compiler/library/allocator storage remains separate.
    """
    reserved = validate_reserved_bytes(reserved_bytes)
    ns = integrals.nocc
    pairs = ((0, 0),) if len(ns) == 1 else ((0, 0), (0, 1), (1, 1))
    if max_memory is None:
        return reserved
    virtual = tuple(e.size-no for e, no in zip(integrals.energies, ns))
    dtype = jnp.result_type(*jax.tree.leaves((integrals.energies, integrals.focks,
                                             integrals.blocks, integrals.factors)))
    # Keep the channel amplitudes while constructing their validity-masked
    # return arrays; releasing that Python list is not an AD lifetime promise.
    output, work = _channel_bytes(ns, virtual, config, dtype)
    resident = reserved+tree_storage_bytes(integrals)+output
    workspace_bytes(max_memory, resident_bytes=resident+work,
                    device=getattr(integrals.focks[0], 'device', None))
    if config.orbital_basis != 'noncanonical':
        return resident
    profiles = []
    for k, (a, b) in enumerate(pairs):
        arguments = (integrals.focks[a], integrals.focks[b],
            integrals.blocks[k] if integrals.blocks is not None else None,
            (integrals.factors[a], integrals.factors[b]) if integrals.factors is not None else None)
        specifications = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), arguments)
        profiles.append(_response_profile(specifications, ns[a], ns[b],
            len(ns) != 1 and a == b, len(ns) == 1, config))
    caches, working, values = tuple(zip(*profiles))
    physical = tuple(x for x in jax.tree.leaves(integrals)
                     if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.inexact))
    # Input direction, gradient and HVP outputs can coexist. Output cotangents
    # and their directions also remain live, notably for a requested T2.
    resident += 3*tree_storage_bytes(physical)+2*sum(values)
    available = workspace_bytes(max_memory, resident_bytes=resident,
                                device=getattr(integrals.focks[0], 'device', None))
    if available < sum(caches)+max(working):
        raise ValueError('Known noncanonical MP2 response arrays exceed the memory budget')
    return resident+sum(caches)
