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


def _endpoints(t1, t2, l1, l2, sector):
    """Return physical bra/ket endpoint rows in the existing IP/EA packing."""
    no, nv = t1.shape
    space = EOMAmplitudeSpace(no, nv, sector)
    occupied, virtual = jnp.eye(no, dtype=t1.dtype), jnp.eye(nv, dtype=t1.dtype)
    bras, kets = [], []
    for p in range(no+nv):
        if sector == "ip":
            if p < no:
                e1 = occupied[p]-jnp.einsum("ie,e->i", l1, t1[p])
                e1 -= jnp.einsum("imef,mef->i", l2, 2*t2[p]-t2[p].swapaxes(1, 2))
                tmp = (-jnp.einsum("ijea,e->ija", l2, t1[p])
                       +jnp.einsum("ja,i->ija", l1, occupied[p]))
                e2 = 2*tmp-tmp.swapaxes(0, 1)
                b1, b2 = occupied[p], jnp.zeros(space.shapes[1], dtype=t1.dtype)
            else:
                a = p-no
                e1, e2 = l1[:, a], 2*l2[:, :, a]-l2[:, :, :, a]
                b1, b2 = t1[:, a], t2[:, :, a]
        else:
            if p < no:
                e1, e2 = -l1[p], -2*l2[p]+l2[:, p]
                b1, b2 = -t1[p], -t2[p]
            else:
                a = p-no
                e1 = virtual[a]-jnp.einsum("mb,m->b", l1, t1[:, a])
                e1 -= jnp.einsum("kmeb,kme->b", l2, 2*t2[:, :, :, a]-t2[:, :, a])
                tmp = (-jnp.einsum("ikba,k->iab", l2, t1[:, a])
                       +jnp.einsum("ib,a->iab", l1, virtual[a]))
                e2 = 2*tmp-tmp.swapaxes(1, 2)
                b1, b2 = virtual[a], jnp.zeros(space.shapes[1], dtype=t1.dtype)
            # The upstream particle-moment outer minus sign is included in
            # b1/b2 here, giving the physical +I virtual endpoint at T=0.
        bras.append(space.pack(e1, e2))
        kets.append(space.pack(b1, b2))
    return jnp.stack(bras), jnp.stack(kets)


def dyson_orbitals(ground, left, result, *, sector=None, max_elements=20_000_000):
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
            or max_elements < 1 or 2*(no+nv)*space.size > max_elements):
        raise ValueError("Dyson endpoints exceed max_elements")
    valid = (ground.converged & left.converged & result.ground_valid
             & (ground.method_id == METHODS.index("ccsd")) & result.converged)
    valid &= jnp.all(jnp.isfinite(t1)) & jnp.all(jnp.isfinite(t2))
    valid &= jnp.all(jnp.isfinite(l1)) & jnp.all(jnp.isfinite(l2))
    response = valid & result.response_valid & result.vector_response
    bra, ket = _endpoints(t1, t2, l1, l2, sector)
    # A live zero makes even a purely stopped EOM-state contribution reject AD
    # when vector response was not requested. T/Lambda partials are not enough.
    dl = bra @ result.right_vectors
    dr = ket @ result.left_vectors
    if not result.vector_response:
        sentinel = require_converged_derivative(result.energies, False)
        zero = sentinel-jax.lax.stop_gradient(sentinel)
        dl, dr = dl+zero, dr+zero
    dl, dr = (require_converged_derivative(jnp.where(valid, x, jnp.nan), response)
              for x in (dl, dr))
    return DysonResult(dl, dr, jnp.sum(dl*dr, axis=0), valid, response)
