"""CI input adaptation and selected normal-ordered integral kernels."""
from itertools import product
from math import comb, prod
from typing import NamedTuple

import jax
import jax.numpy as jnp

from gradscf.ao2mo import validate_integrals, transform_eri_blocks
from gradscf._memory import (DEFAULT_MAX_MEMORY_MB, array_storage_bytes,
                             resolve_memory_mb, validate_reserved_bytes, workspace_bytes)
from ..scf.reference import reference_from_source as prepare_reference
from .types import CIReference
from .space import UCISpace
from .direct import _topology, _normal_order, _needed, _kernels, _diagonal


class CIIntegrals(NamedTuple):
    """Live normal-ordered kernels; no complete MO or spin ERI is retained."""
    kernels: dict
    constant: object
    diagonal: object
    fock_diagonal: object


def from_orbitals(coefficients, focks, reference_energy, space, *, eri=None,
                  eri_pair_matrix=None, df_factors=None, direct_basis=None,
                  max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0):
    """Prepare retained-rank CI from full supplied C/F/reference-energy data.

    Focks and energy include every occupied electron; frozen orbitals are only
    removed from correlation. No SCF or canonicalization is performed. Raw
    unrestricted beta-alpha blocks use the existing pair transpose of ab.
    One shared AO2MO batch supplies only the group blocks used by the existing
    compressed normal-ordering and diagonal formulas. The selected four-virtual
    block can remain dense, but complete total-MO/spin tensors are not formed.
    This plans known preparation arrays, not complete solver/AD residency.
    """
    maximum = resolve_memory_mb(max_memory)
    reserved = validate_reserved_bytes(reserved_bytes)
    unrestricted = isinstance(space, UCISpace)
    cs = tuple(map(jnp.asarray, coefficients)) if unrestricted else (jnp.asarray(coefficients),)*2
    fs = tuple(map(jnp.asarray, focks)) if unrestricted else (jnp.asarray(focks),)*2
    energy = jnp.asarray(reference_energy)
    if (len(cs) != 2 or len(fs) != 2 or any(c.ndim != 2 or c.shape[1] != space.nmo for c in cs)
            or cs[0].shape != cs[1].shape or any(f.shape != (space.nmo,)*2 for f in fs)
            or energy.shape != ()):
        raise ValueError('Supply complete real AO/MO frames, Focks and scalar reference energy')
    if any(jnp.iscomplexobj(a) for a in (*cs, *fs, energy)):
        raise NotImplementedError('Selected CI requires real orbital frames, Focks and energy')
    if any(not jnp.issubdtype(a.dtype, jnp.floating) for a in (*cs, *fs, energy)):
        raise ValueError('Selected CI inputs must have floating-point dtype')
    topology = _topology(space)
    groups = topology.orbitals
    sizes = tuple(map(len, groups))
    quartic, keys = [], set()
    for types in product(range(4), repeat=4):
        if any(not groups[g] for g in types) or types[0] % 2 != types[1] % 2 or types[2] % 2 != types[3] % 2:
            continue
        m, l, _, _ = _normal_order(types, (0, 2, 3, 1), (True, False, True, False))
        if any(r > sizes[g % 4] for g, r in enumerate(m+l)) or not _needed(topology, m, l, None, None):
            continue
        quartic.append(types)
        keys.add((m, l))
    # Only group pairs coexisting in a retained determinant contribute. In a
    # singles space no determinant has two virtual particles; requesting vvvv
    # merely to extract its unused diagonal defeats selected integral supply.
    pairs = {(g, h) for g in range(4) for h in range(g, 4)
             if any(b.counts[g] > 0 and b.counts[h] > (g == h) for b in topology.blocks)}
    diagonal_types = [(g, g, h, h) for g, h in sorted(pairs)]
    diagonal_types += [(g, h, h, g) for g, h in sorted(pairs) if g % 2 == h % 2]
    def request(types):
        return types[2:]+types[:2] if unrestricted and types[0] % 2 > types[2] % 2 else types
    requests = tuple(dict.fromkeys(request(t) for t in quartic+diagonal_types))
    dtype = jnp.dtype(jnp.float64) if direct_basis is not None else jnp.result_type(*cs, *fs, energy,
        *jax.tree.leaves((eri, eri_pair_matrix, df_factors)))
    kernel_sizes = [prod(comb(sizes[g % 4], r) for g, r in enumerate(m+l)) for m, l in keys]
    # Final compressed kernels and three construction versions, plus all
    # group-interaction matrices and diagonal immutable updates, coexist with
    # the shared transform outputs. Source/copies are charged by AO2MO itself.
    extra = dtype.itemsize*(sum(kernel_sizes)+3*max(kernel_sizes, default=0)+6*space.nmo**2
        +2*sum(sizes[g]*sizes[h] for g in range(4) for h in range(g, 4))+4*space.size)
    resident = reserved+array_storage_bytes(*cs, *fs, energy)+extra
    workspace_bytes(maximum, resident_bytes=resident, device=getattr(cs[0], 'device', None))
    selected = tuple(cs[g % 2][:, jnp.asarray(indices, dtype=jnp.int32)] for g, indices in enumerate(groups))
    values = transform_eri_blocks(selected, requests, eri=eri, eri_pair_matrix=eri_pair_matrix,
        df_factors=df_factors, direct_basis=direct_basis, max_memory=maximum,
        reserved_bytes=resident)
    lookup = dict(zip(requests, values))
    fs = tuple(f.astype(dtype) for f in fs)
    def block(types):
        value = lookup[request(types)]
        return value.transpose(2, 3, 0, 1) if request(types) != types else value
    dummy = ((None, None), (None, None))
    kernels = _kernels(fs, dummy, topology, block_source=block)
    interactions = {}
    for g in range(4):
        for h in range(g, 4):
            if (g, h) not in pairs:
                interactions[g, h] = jnp.zeros((sizes[g], sizes[h]), dtype)
                continue
            value = jnp.einsum('ppqq->pq', block((g, g, h, h)))
            if g % 2 == h % 2:
                value -= jnp.einsum('pqqp->pq', block((g, h, h, g)))
            interactions[g, h] = value
    diagonal = _diagonal(fs, dummy, energy, topology, space.size, interactions=interactions)
    fock_diagonal = _diagonal(fs,None,jnp.zeros((),dtype),topology,space.size)
    return CIIntegrals(kernels, energy.astype(jnp.float64) if direct_basis is not None else energy,
                       diagonal,fock_diagonal)


def reference_from_source(source, *, max_memory=None):
    max_memory = resolve_memory_mb(max_memory, source)
    if isinstance(source, CIReference):
        h,g = validate_integrals(source.h1,source.eri)
        if not isinstance(source.nocc,int) or not 0 <= source.nocc <= h.shape[0]:
            raise ValueError("Invalid restricted nocc")
        return CIReference(h,g,source.nocc,source.nuclear_repulsion,source.mo_energy)
    ref = prepare_reference(source, max_memory=max_memory)
    return CIReference(ref.h1,ref.eri,ref.nocc,ref.nuclear_repulsion,ref.mo_energy)
