"""Requested AO-to-MO blocks with bounded dense, packed and DF workspaces."""
from math import prod
from numbers import Integral

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from gradscf._memory import (DEFAULT_MAX_MEMORY_MB, array_storage_bytes,
                             validate_memory_mb, validate_reserved_bytes, workspace_bytes)
from gradscf.integrals.molecular.eri import _metadata_arrays


def transform_eri_block(
    coefficients, *, eri=None, eri_pair_matrix=None, df_factors=None, direct_basis=None,
    max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0,
):
    """Transform a requested (p q|r s) block, without constructing full MO ERIs.

    Four real coefficient matrices may select different orbital subspaces.
    Dense, s4 packed, DF and native-direct representations share this convention.
    Native direct blocks use bounded shell slabs with coordinate AD through
    order two; no full AO ERI tensor is required.

    ``max_memory`` is a static planning budget in decimal MB. Known resident
    inputs, output buffers and transformation scratch share the budget after the
    common 20% runtime reserve. DF accumulation reserves three output buffers;
    dense and s4 reserve two. DF scratch is tiled over the auxiliary axis; dense
    and s4 inputs remain small-system, resident-AO reference paths.
    Checkpointing recomputes DF intermediates for differentiation. This plans
    forward arrays, not total AD-process storage or the required input cotangent.
    ``reserved_bytes`` additionally charges caller-owned resident arrays.
    A traced input may not expose its target device. Without an explicitly
    selected default device, only the shared helper's host-capacity cap applies.
    """
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    if len(coefficients) != 4:
        raise ValueError("Supply four orbital coefficient matrices")
    c, d, e, f = map(jnp.asarray, coefficients)
    if any(x.ndim != 2 or x.shape[0] != c.shape[0] for x in (c, d, e, f)):
        raise ValueError("Coefficient matrices must share the AO dimension")
    if any(jnp.iscomplexobj(x) for x in (c, d, e, f)):
        raise NotImplementedError("MO block transformation requires real coefficients")
    if sum(value is not None for value in (eri, eri_pair_matrix, df_factors, direct_basis)) != 1:
        raise ValueError("Supply exactly one of eri, eri_pair_matrix, df_factors, direct_basis")
    if direct_basis is not None:
        from .native import transform_eri_block as transform_native_block
        return transform_native_block((c, d, e, f), direct_basis,
            max_memory=max_memory, reserved_bytes=reserved_bytes)
    nao = c.shape[0]
    representation, source = _array_source(nao, eri, eri_pair_matrix, df_factors)

    coefficients = (c, d, e, f)
    shape = tuple(x.shape[1] for x in coefficients)
    dtype = jnp.result_type(source, *coefficients)
    block_size = _eri_block_size(source, coefficients, representation, max_memory,
                                 reserved_bytes=reserved_bytes)
    if not prod(shape) or (representation == "df" and source.shape[0] == 0):
        return jnp.zeros(shape, dtype=dtype)
    if representation == "dense":
        return jnp.einsum(
            "pqrs,pP,qQ,rR,sS->PQRS", source, c, d, e, f,
            precision="highest", optimize="optimal",
        )
    if representation == "packed":
        rows, cols, _, _ = _metadata_arrays(nao, c.dtype)
        products = _mo_pair_products(c, d, rows, cols)
        other = _mo_pair_products(e, f, rows, cols)
        return jnp.einsum(
            "pqP,PQ,rsQ->pqrs", products, source, other, precision="highest"
        )
    return _transform_df_eri_block(source, coefficients, shape, dtype, block_size)


def transform_eri_blocks(
    coefficients, requests, *, eri=None, eri_pair_matrix=None, df_factors=None,
    direct_basis=None, max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0,
):
    """Transform static four-space-ID requests within one shared MB budget.

    ``coefficients`` contains each real AO/MO space once. ``requests`` contains
    four static integer IDs per output, returned in the supplied order. Native
    integrals share shell traversal and first/second/half contractions locally;
    physical ERI symmetry is used only in that branch. Array inputs preserve
    their raw input/AD chart and reuse only identical ordered requests.

    ``reserved_bytes`` charges other caller-owned live arrays against the same
    ``max_memory`` after its single 20% reserve. Forward outputs and scratch are
    planned together; complete solver/AD process storage remains caller-owned.
    """
    max_memory = validate_memory_mb(max_memory)
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    cs = tuple(map(jnp.asarray, coefficients))
    if any(c.ndim != 2 or c.shape[0] != cs[0].shape[0] for c in cs):
        raise ValueError('Coefficient matrices must share the AO dimension')
    if any(jnp.iscomplexobj(c) for c in cs):
        raise NotImplementedError('MO block transformation requires real coefficients')
    try:
        requests = tuple(tuple(request) for request in requests)
    except TypeError as error:
        raise ValueError('Requests must contain four static integer space IDs') from error
    if any(len(r) != 4 or any(isinstance(i, bool) or not isinstance(i, Integral)
                              or not 0 <= i < len(cs) for i in r) for r in requests):
        raise ValueError('Each request must contain four valid static integer space IDs')
    if sum(x is not None for x in (eri, eri_pair_matrix, df_factors, direct_basis)) != 1:
        raise ValueError('Supply exactly one of eri, eri_pair_matrix, df_factors, direct_basis')
    if not requests:
        return ()
    if direct_basis is not None:
        from .native import transform_eri_blocks as native_blocks
        return native_blocks(cs, requests, direct_basis, max_memory=max_memory,
                             reserved_bytes=reserved_bytes)

    representation, source = _array_source(cs[0].shape[0], eri, eri_pair_matrix, df_factors)
    unique = tuple(dict.fromkeys(requests))
    selected = {r: tuple(cs[i] for i in r) for r in unique}
    sizes = {r: prod(c.shape[1] for c in selected[r]) *
                jnp.result_type(source, *selected[r]).itemsize for r in unique}
    output_buffers = 3 if representation == 'df' else 2
    resident = array_storage_bytes(source, *cs)
    reservations = {r: reserved_bytes + resident - array_storage_bytes(source, *selected[r])
                       + output_buffers * (sum(sizes.values()) - sizes[r]) for r in unique}
    # Preflight every request before the first contraction. Passing the original
    # MB value plus byte reservations avoids applying the runtime reserve twice.
    for r in unique:
        _eri_block_size(source, selected[r], representation, max_memory,
                        reserved_bytes=reservations[r])
    options = {{'dense': 'eri', 'packed': 'eri_pair_matrix', 'df': 'df_factors'}[representation]: source}
    values = {r: transform_eri_block(selected[r], **options, max_memory=max_memory,
                                    reserved_bytes=reservations[r]) for r in unique}
    return tuple(values[r] for r in requests)


def _array_source(nao, eri, eri_pair_matrix, df_factors):
    """Validate source shape before either singular or batch memory planning."""
    if eri is not None:
        source = jnp.asarray(eri)
        if source.shape != (nao,) * 4:
            raise ValueError("AO eri must have shape (nao, nao, nao, nao)")
        return "dense", source
    if eri_pair_matrix is not None:
        source = jnp.asarray(eri_pair_matrix)
        npair = nao * (nao + 1) // 2
        if source.shape != (npair, npair):
            raise ValueError("eri_pair_matrix must be a square s4 AO pair matrix")
        return "packed", source
    source = jnp.asarray(df_factors)
    if source.ndim != 3 or source.shape[1:] != (nao, nao):
        raise ValueError("df_factors must have shape (naux, nao, nao)")
    return "df", source


def _eri_block_size(source, coefficients, representation, max_memory, *, reserved_bytes=0):
    """Check static forward-array storage and choose the DF auxiliary tile."""
    arrays = (source, *coefficients)
    device = None
    for array in arrays:
        device = getattr(array, "device", None)
        if callable(device):
            device = device()
        if device is not None:
            break
    if device is None:
        device = getattr(jax.config, "jax_default_device", None)
    available = workspace_bytes(
        max_memory, resident_bytes=array_storage_bytes(*arrays) + reserved_bytes, device=device,
    )
    nao = coefficients[0].shape[0]
    shape = tuple(x.shape[1] for x in coefficients)
    dtype = jnp.result_type(*arrays)
    itemsize = dtype.itemsize
    # DF can simultaneously retain the old accumulator, a tile contribution
    # and the new accumulator, including during eager tail accumulation.
    output_buffers = 3 if representation == "df" else 2
    available -= output_buffers * prod(shape) * itemsize
    if available < 0:
        raise ValueError("AO->MO output buffers exceed the working-memory budget")
    if not prod(shape) or (representation == "df" and source.shape[0] == 0):
        return 0
    # Promotion may create coefficient copies inside contractions. In the DF
    # branch the AO factors themselves are promoted only after taking a tile.
    coefficient_copies = sum(x.size * itemsize for x in coefficients
                             if x.dtype != dtype)
    available -= coefficient_copies

    if representation == "dense":
        # Bound the three successive one-leg intermediates of the existing
        # optimized contraction, plus a possible promoted resident-AO copy.
        width = max(shape)
        scratch = itemsize * sum(nao ** (4 - n) * width ** n for n in (1, 2, 3))
        if source.dtype != dtype:
            scratch += source.size * itemsize
        if available < scratch:
            raise ValueError("Dense AO->MO scratch exceeds the working-memory budget")
        return None
    if representation == "packed":
        npair = source.shape[0]
        left_size, right_size = shape[0] * shape[1], shape[2] * shape[3]
        # Pair metadata, gathered coefficients, pair-product construction and
        # the larger pair-contraction intermediate remain explicit scratch.
        metadata = 8 * npair + 4 * nao ** 2 + itemsize * npair
        scratch = metadata + itemsize * npair * (
            2 * sum(shape) + 4 * (left_size + right_size)
            + max(left_size, right_size) + 1
        )
        if source.dtype != dtype:
            scratch += source.size * itemsize
        if available < scratch:
            raise ValueError("Packed AO->MO scratch exceeds the working-memory budget")
        return None
    per_aux = itemsize * (
        nao ** 2 + nao * (min(shape[:2]) + min(shape[2:]))
        + shape[0] * shape[1] + shape[2] * shape[3]
    )
    if any(x.dtype != dtype for x in arrays):
        # Retain room for original and promoted versions of tile-local
        # arrays without casting the full resident factor tensor.
        per_aux *= 2
    if available < per_aux:
        raise ValueError("Minimum DF AO->MO tile exceeds the working-memory budget")
    return min(source.shape[0], available // per_aux)


def _transform_df_eri_block(source, coefficients, shape, dtype, block_size):
    """Accumulate exact MO blocks using bounded, recomputed DF intermediates."""
    c, d, e, f = coefficients

    def pair_transform(block, left, right):
        if left.shape[1] <= right.shape[1]:
            half = jnp.einsum("Lpq,pP->LPq", block, left, precision="highest")
            return jnp.einsum("LPq,qQ->LPQ", half, right, precision="highest")
        half = jnp.einsum("Lpq,qQ->LpQ", block, right, precision="highest")
        return jnp.einsum("LpQ,pP->LPQ", half, left, precision="highest")

    def contribution(block, c, d, e, f):
        left = pair_transform(block, c, d)
        right = pair_transform(block, e, f)
        return jnp.einsum("Lpq,Lrs->pqrs", left, right, precision="highest")

    @jax.checkpoint
    def accumulate(index, value, factors, c, d, e, f):
        block = jax.lax.dynamic_slice_in_dim(
            factors, index * block_size, block_size, axis=0,
        )
        return value + contribution(block, c, d, e, f)

    nblocks, tail = divmod(source.shape[0], block_size)
    g = jax.lax.fori_loop(
        0, nblocks, lambda i, value: accumulate(i, value, source, c, d, e, f),
        jnp.zeros(shape, dtype=dtype),
    )
    if tail:
        g = g + jax.checkpoint(contribution)(
            source[nblocks * block_size:], c, d, e, f,
        )
    return g


def _mo_pair_products(left: Array, right: Array, rows: Array, cols: Array) -> Array:
    left_rows = left[rows]
    left_cols = left[cols]
    right_rows = right[rows]
    right_cols = right[cols]
    products = jnp.einsum(
        "Pi,Pa->iaP",
        left_rows,
        right_cols,
        precision=Precision.HIGHEST,
    )
    swapped = jnp.einsum(
        "Pi,Pa->iaP",
        left_cols,
        right_rows,
        precision=Precision.HIGHEST,
    )
    offdiag = (rows != cols).astype(products.dtype)
    return products + swapped * offdiag[None, None, :]
