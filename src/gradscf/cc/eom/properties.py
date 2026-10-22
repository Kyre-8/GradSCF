# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see ../LICENSE.pyscf.
# Endpoint contractions adapted from PySCF 2.13.0 cc/momgfccsd.py.
"""Restricted CCSD charged-state Dyson amplitudes in the active MO frame."""

import jax
import jax.numpy as jnp

from ...solvers.diagnostics import require_converged_derivative
from ..ground import METHODS
from .amplitudes import EOMAmplitudeSpace
from .types import DysonResult
from ..._memory import tree_storage_bytes, resolve_memory_mb, validate_reserved_bytes, workspace_bytes


def _endpoint(t1, t2, l1, l2, sector, p):
    """One physical bra/ket row, with no orbital identity or endpoint stack."""
    no, nv = t1.shape
    space = EOMAmplitudeSpace(no, nv, sector)

    def occupied(p):
        unit = jax.nn.one_hot(p, no, dtype=t1.dtype)
        if sector == "ip":
            e1 = unit-jnp.einsum("ie,e->i", l1, t1[p])
            e1 -= jnp.einsum("imef,mef->i", l2, 2*t2[p]-t2[p].swapaxes(1, 2))
            tmp = (-jnp.einsum("ijea,e->ija", l2, t1[p])
                   +jnp.einsum("ja,i->ija", l1, unit))
            e2 = 2*tmp-tmp.swapaxes(0, 1)
            b1, b2 = unit, jnp.zeros(space.shapes[1], dtype=t1.dtype)
        else:
            e1, e2 = -l1[p], -2*l2[p]+l2[:, p]
            b1, b2 = -t1[p], -t2[p]
        return packed(e1, e2, b1, b2)

    def virtual(p):
        a = p-no
        if sector == "ip":
            e1, e2 = l1[:, a], 2*l2[:, :, a]-l2[:, :, :, a]
            b1, b2 = t1[:, a], t2[:, :, a]
        else:
            unit = jax.nn.one_hot(a, nv, dtype=t1.dtype)
            e1 = unit-jnp.einsum("mb,m->b", l1, t1[:, a])
            e1 -= jnp.einsum("kmeb,kme->b", l2, 2*t2[:, :, :, a]-t2[:, :, a])
            tmp = (-jnp.einsum("ikba,k->iab", l2, t1[:, a])
                   +jnp.einsum("ib,a->iab", l1, unit))
            e2 = 2*tmp-tmp.swapaxes(1, 2)
            b1, b2 = unit, jnp.zeros(space.shapes[1], dtype=t1.dtype)
        return packed(e1, e2, b1, b2)

    def packed(e1, e2, b1, b2):
        return (space.pack(e1, e2).astype(jnp.result_type(t1, t2, l1, l2)),
                space.pack(b1, b2).astype(jnp.result_type(t1, t2)))

    # Particle-moment signs are already included in b1/b2 above.
    return jax.lax.cond(p < no, occupied, virtual, p)


def _endpoints(t1, t2, l1, l2, sector):
    """Explicit small-system reference/export; Dyson contracts bounded rows."""
    return jax.vmap(lambda p: _endpoint(t1, t2, l1, l2, sector, p))(
        jnp.arange(sum(t1.shape)))


def dyson_orbitals(ground, left, result, *, sector=None, max_elements=20_000_000,
                   max_memory=None, reserved_bytes=0):
    """Left/right one-spin Dyson columns and signed biorthogonal pole weights.

    ``left`` is the converged ground LambdaResult, not the charged left vector.
    Amplitudes use the active occupied-then-virtual spatial MO frame. Their
    products define pole residues; individual norms are not spectral weights.
    A spin-summed closed-shell spectrum multiplies ``weights`` by two. No
    photoionization matrix element or continuum orbital is included.

    Forward properties also accept stopped EOM vectors. Their derivatives are
    deliberately invalid unless the EOM result enabled vector_response. The
    valid flag checks the supplied states; response_valid additionally checks
    isolated-root response. Actual backward linear solves check their residuals.
    max_elements bounds simultaneous endpoint rows; max_memory budgets known
    input/output and row work arrays in decimal MB. Full endpoint matrices are
    unnecessary. Reverse mode recomputes row tiles rather than storing a tape.
    """
    if sector is None:
        sector = result.sector
    elif sector != result.sector:
        raise ValueError("Dyson sector disagrees with the EOM result")
    if sector not in {"ip", "ea"}:
        raise ValueError("Dyson sector must be ip or ea")
    if isinstance(ground.t1, (tuple, list)) or jnp.ndim(ground.t1) != 2:
        raise NotImplementedError("Dyson orbitals currently require restricted CCSD")
    t1, t2, l1, l2 = map(jnp.asarray, (ground.t1, ground.t2, left.l1, left.l2))
    no, nv = t1.shape
    space = EOMAmplitudeSpace(no, nv, sector)
    if not no or not nv or t2.shape != (no, no, nv, nv) or l1.shape != t1.shape or l2.shape != t2.shape:
        raise ValueError("Ground and Lambda amplitudes must match the active space")
    if any(jnp.iscomplexobj(x) for x in (t1, t2, l1, l2)):
        raise NotImplementedError("Dyson orbitals currently require real amplitudes")
    shape = (space.size, result.energies.size)
    if result.right_vectors.shape != shape or result.left_vectors.shape != shape:
        raise ValueError("Charged vectors do not match the sector and active space")
    if (not isinstance(max_elements, int) or isinstance(max_elements, bool)
            or max_elements < 2*space.size):
        raise ValueError("Dyson endpoints exceed max_elements")
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    dtype_left = jnp.result_type(t1, t2, l1, l2, result.right_vectors)
    dtype_right = jnp.result_type(t1, t2, result.left_vectors)
    output_bytes = (no+nv)*result.energies.size*(dtype_left.itemsize+dtype_right.itemsize)
    resident = tree_storage_bytes(ground, left, result, t1, t2, l1, l2)
    available = workspace_bytes(resolve_memory_mb(max_memory),
        resident_bytes=reserved_bytes+resident+3*output_bytes, device=getattr(t1, "device", None))
    # Include the occupied/virtual slices of T2 entering each endpoint: these
    # panels can exceed the packed IP dimension when nv is much larger than no.
    item = max(dtype_left.itemsize, dtype_right.itemsize)
    per_row = item*(12*space.size+4*max(no*nv*nv, no*no*nv)
                    +8*result.energies.size+2*(no+nv))
    block = min(32, no+nv, max_elements//(2*space.size), available//per_row)
    if block < 1:
        raise ValueError("Minimum Dyson row exceeds the working-memory budget")
    valid = (ground.converged & left.converged & result.ground_valid
             & (ground.method_id == METHODS.index("ccsd")) & result.converged)
    valid &= jnp.all(jnp.isfinite(t1)) & jnp.all(jnp.isfinite(t2))
    valid &= jnp.all(jnp.isfinite(l1)) & jnp.all(jnp.isfinite(l2))
    response = valid & result.response_valid & result.vector_response
    # A live zero makes even a purely stopped EOM-state contribution reject AD
    # when vector response was not requested. T/Lambda partials are not enough.
    def contract(index, outputs):
        start = jnp.minimum(index*block, no+nv-block)
        bra, ket = jax.vmap(lambda p: _endpoint(t1, t2, l1, l2, sector, p))(
            start+jnp.arange(block))
        rows = bra@result.right_vectors, ket@result.left_vectors
        return tuple(jax.lax.dynamic_update_slice_in_dim(out, row, start, axis=0)
                     for out, row in zip(outputs, rows))

    dl, dr = jax.lax.fori_loop(0, (no+nv+block-1)//block, jax.checkpoint(contract),
        (jnp.zeros((no+nv, result.energies.size), dtype_left),
         jnp.zeros((no+nv, result.energies.size), dtype_right)))
    if not result.vector_response:
        sentinel = require_converged_derivative(result.energies, False)
        zero = sentinel-jax.lax.stop_gradient(sentinel)
        dl, dr = dl+zero, dr+zero
    dl, dr = (require_converged_derivative(jnp.where(valid, x, jnp.nan), response)
              for x in (dl, dr))
    return DysonResult(dl, dr, jnp.sum(dl*dr, axis=0), valid, response)
