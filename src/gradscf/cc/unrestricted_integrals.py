"""Six selected UCC spin blocks from a supplied real UHF orbital frame."""
from math import prod

import jax
import jax.numpy as jnp

from gradscf._memory import (DEFAULT_MAX_MEMORY_MB, array_storage_bytes,
                             resolve_memory_mb, validate_reserved_bytes, workspace_bytes)
from gradscf.ao2mo import transform_eri_blocks, unrestricted_frozen_indices
from gradscf.ao2mo.blocks import _array_source
from .uccsd import SpinCCIntegrals


_SPACES = ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1),
           (0, 1, 0, 1), (0, 1, 1, 1), (1, 1, 1, 1))


def _spin_terms(sizes):
    """Static nonzero terms in <pq||rs>=(pr|qs)-(ps|qr).

    The beta-alpha pair block is the transpose of alpha-beta, matching the raw
    UCC reference chart without assuming AO permutation symmetry.
    """
    terms = []
    for block, spaces in enumerate(_SPACES):
        for axes, sign in (((0, 2, 1, 3), 1), ((0, 3, 1, 2), -1)):
            for left, right in ((0, 0), (0, 1), (1, 0), (1, 1)):
                spins = [None]*4
                for axis, spin in zip(axes, (left, left, right, right)):
                    spins[axis] = spin
                if any(sizes[space][spin] == 0 for space, spin in zip(spaces, spins)):
                    continue
                ordered = axes[2:]+axes[:2] if left > right else axes
                request = tuple(2*spins[axis]+spaces[axis] for axis in ordered)
                permutation = tuple(ordered.index(axis) for axis in range(4))
                target = tuple(slice(0, sizes[space][0]) if spin == 0
                               else slice(sizes[space][0], sum(sizes[space]))
                               for space, spin in zip(spaces, spins))
                terms.append((block, request, permutation, target, sign))
    return terms


def from_orbitals(coefficients, focks, nocc, reference_energy, *, frozen=None,
                  eri=None, eri_pair_matrix=None, df_factors=None, direct_basis=None,
                  max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0):
    """Prepare the existing (SpinCCIntegrals, nocc, nvir) UCC payload.

    Inputs are two AO/MO coefficient matrices, two full MO Focks and a supplied
    total HF energy. Frozen occupied electrons remain in those Focks and energy.
    No SCF, J/K, canonicalization or symmetrization occurs. Array inputs retain
    their raw AO/Fock chart; native inputs promote Fock/energy to float64.

    One AO2MO request batch shares ``max_memory`` (decimal MB) with known inputs,
    final six spin blocks, assembly scratch and caller ``reserved_bytes``.
    Complete spatial/spin ERIs are not formed. The spin vvvv block remains
    dense; downstream amplitudes, solver workspace and AD state are not capped.
    """
    max_memory = resolve_memory_mb(max_memory)
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    if len(coefficients) != 2 or len(focks) != 2:
        raise ValueError('Supply two unrestricted coefficient and full MO Fock matrices')
    cs, fs = tuple(map(jnp.asarray, coefficients)), tuple(map(jnp.asarray, focks))
    energy = jnp.asarray(reference_energy)
    if (any(c.ndim != 2 for c in cs) or cs[0].shape != cs[1].shape
            or any(f.shape != (c.shape[1],)*2 for c, f in zip(cs, fs)) or energy.ndim != 0):
        raise ValueError('Supply equally sized AO/MO frames, full MO Focks and scalar reference_energy')
    if any(jnp.iscomplexobj(x) for x in (*cs, *fs, energy)):
        raise NotImplementedError('Unrestricted CC requires real orbitals, Focks and reference energy')
    if any(not jnp.issubdtype(x.dtype, jnp.floating) for x in (*cs, *fs, energy)):
        raise ValueError('Unrestricted CC inputs must have floating-point dtype')
    if not isinstance(nocc, (tuple, list)):
        raise ValueError('nocc must be (nalpha, nbeta) with integer occupations')
    frozen = unrestricted_frozen_indices(cs[0].shape[1], nocc, frozen)
    options = dict(eri=eri, eri_pair_matrix=eri_pair_matrix, df_factors=df_factors,
                   direct_basis=direct_basis)
    if sum(x is not None for x in options.values()) != 1:
        raise ValueError('Supply exactly one AO integral representation')
    source = None
    if direct_basis is None:
        keyword = next(key for key, value in options.items() if value is not None)
        _, source = _array_source(cs[0].shape[0], eri, eri_pair_matrix, df_factors)
        if jnp.iscomplexobj(source):
            raise NotImplementedError('Unrestricted CC requires real AO integrals')
        if not jnp.issubdtype(source.dtype, jnp.floating):
            raise ValueError('Unrestricted CC AO integrals must have floating-point dtype')
        options = {keyword: source}
    occ = tuple(tuple(i for i in range(no) if i not in fr) for no, fr in zip(nocc, frozen))
    vir = tuple(tuple(i for i in range(no, cs[0].shape[1]) if i not in fr)
                for no, fr in zip(nocc, frozen))
    sizes = tuple(tuple(map(len, space)) for space in (occ, vir))
    dtype = jnp.dtype(jnp.float64) if direct_basis is not None else jnp.result_type(*cs, *fs, source)
    shapes = tuple(tuple(sum(sizes[space]) for space in block) for block in _SPACES)
    spin_bytes = sum(prod(shape) for shape in shapes)*dtype.itemsize
    # Keep the final blocks plus old/new scatter buffers in the shared budget.
    assembly_bytes = spin_bytes+2*max(map(prod, shapes))*dtype.itemsize
    nactive = sum(map(sum, sizes))
    planned = (jax.ShapeDtypeStruct((nactive, nactive), dtype),
               jax.ShapeDtypeStruct((nactive, nactive), dtype),
               *(jax.ShapeDtypeStruct((c.shape[0], len(o)+len(v)), c.dtype)
                 for c, o, v in zip(cs, occ, vir)))
    device = getattr(cs[0], 'device', None)
    device = device() if callable(device) else device
    if device is None:
        device = getattr(jax.config, 'jax_default_device', None)
    workspace_bytes(max_memory, device=device, resident_bytes=reserved_bytes+assembly_bytes+
        array_storage_bytes(*cs, *fs, energy, source, *jax.tree.leaves(direct_basis), *planned)
        +8*nactive)
    terms = _spin_terms(sizes)
    requests = tuple(dict.fromkeys(term[1] for term in terms))
    selections = tuple(jnp.asarray(indices, dtype=jnp.int32) for spin in range(2)
                       for indices in (occ[spin], vir[spin]))
    spaces = tuple(c[:, selection] for c, pair in zip(cs, (selections[:2], selections[2:]))
                   for selection in pair)
    no, nv = map(sum, sizes)
    fock = jnp.zeros((nactive, nactive), dtype)
    for spin, f in enumerate(fs):
        active = jnp.concatenate((selections[2*spin], selections[2*spin+1]))
        target = jnp.concatenate((jnp.arange(sum(sizes[0][:spin]), sum(sizes[0][:spin+1])),
                                  no+jnp.arange(sum(sizes[1][:spin]), sum(sizes[1][:spin+1]))))
        fock = fock.at[jnp.ix_(target, target)].set(f[jnp.ix_(active, active)].astype(dtype))
    selected_energy = energy.astype(dtype) if direct_basis is not None else energy
    resident = array_storage_bytes(*cs, *fs, energy, fock, selected_energy, *selections)
    blocks = transform_eri_blocks(spaces, requests, **options, max_memory=max_memory,
        reserved_bytes=reserved_bytes+resident+assembly_bytes+2*nactive*nactive*dtype.itemsize)
    values = dict(zip(requests, blocks))
    result = [jnp.zeros(shape, dtype) for shape in shapes]
    for block, request, permutation, target, sign in terms:
        result[block] = result[block].at[target].add(sign*values[request].transpose(permutation))
    return SpinCCIntegrals(fock, *result, selected_energy), sizes[0], sizes[1]
