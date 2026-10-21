"""Experimental exact excitation-block contractions in HF quasiparticle coordinates.

Groups are (virtual alpha, virtual beta, occupied alpha holes, occupied beta holes).
Independent increasing-index coefficients are orthonormal. Dense antisymmetric
blocks are expanded only under an explicit capacity bound. No determinant
connection table, dense Hamiltonian or bespoke numerical solver is constructed.
"""
from dataclasses import dataclass
from itertools import combinations, permutations, product
from math import comb, factorial, prod

import jax.numpy as jnp
import numpy as np

from .integrals import _active_orbitals
from gradscf.ao2mo import frozen_indices, unrestricted_frozen_indices


def _parity(order):
    return (-1)**sum(order[i] > order[j] for i in range(len(order)) for j in range(i+1, len(order)))


@dataclass(frozen=True)
class Block:
    excitation: tuple
    counts: tuple
    shape: tuple
    full_shape: tuple
    indices: tuple
    offset: int
    size: int


@dataclass(frozen=True)
class TensorSpace:
    nmo: int
    nocc: tuple
    unrestricted: bool
    orbitals: tuple
    blocks: tuple
    maps: tuple
    size: int
    dense_elements: int

    @property
    def ranks(self):
        return tuple(sum(b.excitation) for b in self.blocks for _ in range(b.size))

    def spin_ranks(self):
        return tuple(np.concatenate([np.full(b.size, b.excitation[s], dtype=np.int32)
                                     for b in self.blocks]) for s in range(2))

    def amplitudes(self, first):
        tensors = dict(zip((b.excitation for b in self.blocks), self.expand(first, max_rank=2)))
        va, vb, oa, ob = self.orbitals
        def doubles(key, shape):
            value = tensors.get(key)
            return jnp.zeros(shape, first.dtype) if value is None else -value.transpose(2, 3, 0, 1)
        ab = doubles((1, 1), (len(oa), len(ob), len(va), len(vb)))
        if not self.unrestricted:
            return ab
        return (doubles((2, 0), (len(oa), len(oa), len(va), len(va))), ab,
                doubles((0, 2), (len(ob), len(ob), len(vb), len(vb))))

    def expand(self, vector, *, max_rank=None):
        result = []
        for block, maps in zip(self.blocks, self.maps):
            if max_rank is not None and sum(block.excitation) > max_rank:
                result.append(None)
                continue
            if any(item is None for item in maps):
                raise ValueError("This packed-only space does not expand high-rank tensors")
            packed = vector[block.offset:block.offset+block.size].reshape(block.shape)
            positions, phase, axis = [], 1, 0
            for count, (index, sign) in zip(block.counts, maps):
                shape = [1]*len(block.full_shape)
                shape[axis:axis+count] = index.shape
                positions.append(jnp.asarray(index).reshape(shape))
                phase = phase*jnp.asarray(sign).reshape(shape)
                axis += count
            result.append(packed[tuple(positions)]*phase)
        return tuple(result)

    def pack(self, tensors):
        result = []
        for block, tensor in zip(self.blocks, tensors):
            if tensor is None:
                result.append(jnp.zeros(block.size, tensors[0].dtype))
                continue
            positions = []
            for group, choices in enumerate(block.indices):
                for column in range(block.counts[group]):
                    shape = [1]*4
                    shape[group] = len(choices)
                    positions.append(jnp.asarray(choices[:, column]).reshape(shape))
            value = tensor[tuple(positions)] if positions else tensor
            result.append(value.reshape(-1))
        return jnp.concatenate(result)

    def gaps(self, focks):
        fa, fb = focks if self.unrestricted else (focks[0], focks[0])
        va, vb, oa, ob = self.orbitals
        energies = (jnp.diag(fa)[jnp.asarray(va, dtype=jnp.int32)],
                    jnp.diag(fb)[jnp.asarray(vb, dtype=jnp.int32)],
                    -jnp.diag(fa)[jnp.asarray(oa, dtype=jnp.int32)],
                    -jnp.diag(fb)[jnp.asarray(ob, dtype=jnp.int32)])
        result = []
        for block in self.blocks:
            gap = jnp.zeros(block.shape, fa.dtype)
            for group, (eps, choices) in enumerate(zip(energies, block.indices)):
                shape = [1]*4
                shape[group] = len(choices)
                gap = gap+jnp.sum(eps[jnp.asarray(choices)], axis=1).reshape(shape)
            result.append(gap.reshape(-1))
        return jnp.concatenate(result)

    def determinant_map(self, determinants):
        """Physical determinant indices/signs for the packed wedge chart."""
        from ..fci.cistring import excite
        n = self.nmo
        na, nb = self.nocc
        reference = ((1 << na)-1) | (((1 << nb)-1) << n)
        lookup = {d: i for i, d in enumerate(determinants)}
        va, vb, oa, ob = self.orbitals
        indices, signs = [], []
        for block in self.blocks:
            rank = sum(block.excitation)
            phase = (-1)**(rank*(rank-1)//2)
            for selected in product(*block.indices):
                particles = tuple(va[i] for i in selected[0])+tuple(n+vb[i] for i in selected[1])
                holes = tuple(oa[i] for i in selected[2])+tuple(n+ob[i] for i in selected[3])
                determinant, sign = excite(reference, holes, particles)
                indices.append(lookup[determinant])
                signs.append(phase*sign)
        return np.asarray(indices, dtype=np.int32), np.asarray(signs, dtype=np.int8)


def _antisymmetry_map(n, rank):
    if rank == 0:
        return np.asarray(0, dtype=np.int32), np.asarray(1, dtype=np.int8)
    indices = np.zeros((n,)*rank, dtype=np.int32)
    signs = np.zeros((n,)*rank, dtype=np.int8)
    for index, chosen in enumerate(combinations(range(n), rank)):
        for order in permutations(range(rank)):
            pos = tuple(chosen[i] for i in order)
            indices[pos], signs[pos] = index, _parity(order)
    return indices, signs


def make_tensor_space(nmo, nocc, *, max_excitation, frozen=None,
                      max_determinants=5000, max_tensor_elements=2_000_000,
                      packed_only=False):
    unrestricted = isinstance(nocc, tuple)
    ns = nocc if unrestricted else (nocc, nocc)
    if unrestricted:
        fr = unrestricted_frozen_indices(nmo, ns, frozen)
    else:
        shared = frozen_indices(nmo, nocc, frozen)
        fr = (shared, shared)
    oa, va = _active_orbitals(nmo, ns[0], fr[0])
    ob, vb = _active_orbitals(nmo, ns[1], fr[1])
    orbitals = tuple(tuple(x) for x in (va, vb, oa, ob))
    dims = tuple(map(len, orbitals))
    ranks = sorted((a, b) for a in range(min(dims[0], dims[2], max_excitation)+1)
                   for b in range(min(dims[1], dims[3], max_excitation)+1)
                   if a+b <= max_excitation)
    ranks.sort(key=lambda x: (sum(x), x))
    descriptors, size, dense = [], 0, 0
    for a, b in ranks:
        counts = (a, b, a, b)
        shape = tuple(comb(n, r) for n, r in zip(dims, counts))
        full = tuple(n for n, r in zip(dims, counts) for _ in range(r))
        descriptors.append((a, b, counts, shape, full))
        size += prod(shape)
        dense += prod(full)
    if size > max_determinants:
        raise ValueError(f"Tensor space requires {size} independent coefficients; max_determinants={max_determinants}")
    expanded = (sum(prod(full) for a, b, _, _, full in descriptors if a+b <= 2)
                if packed_only else dense)
    if expanded > max_tensor_elements:
        raise ValueError(f"Dense tensor expansion requires {expanded} elements; max_tensor_elements={max_tensor_elements}")
    blocks, maps, offset, cache = [], [], 0, {}
    for a, b, counts, shape, full in descriptors:
        indices = tuple(np.asarray(list(combinations(range(n), r)), dtype=np.int32).reshape((comb(n, r), r))
                        for n, r in zip(dims, counts))
        block_size = prod(shape)
        blocks.append(Block((a, b), counts, shape, full, indices, offset, block_size))
        group_maps = []
        for n, r in zip(dims, counts):
            if packed_only and a+b > 2:
                group_maps.append(None)
                continue
            if (n, r) not in cache:
                cache[(n, r)] = _antisymmetry_map(n, r)
            group_maps.append(cache[(n, r)])
        maps.append(tuple(group_maps))
        offset += block_size
    return TensorSpace(nmo, tuple(ns), unrestricted, orbitals, tuple(blocks), tuple(maps), size, dense)


def _normal_order(types, order, daggers):
    create = {i:(types[i] < 2) == daggers[i] for i in order}
    arranged = sorted(order, key=lambda i: (0, types[i]) if create[i] else (1, -types[i]))
    sign = _parity([order.index(i) for i in arranged])
    creators = tuple(i for i in arranged if create[i])
    contracted = tuple(reversed([i for i in arranged if not create[i]]))
    m = tuple(sum(types[i] == t for i in creators) for t in range(4))
    l = tuple(sum(types[i] == t for i in contracted) for t in range(4))
    return m, l, creators+contracted, sign


def _shuffles(counts, created):
    choices = [tuple(combinations(range(n), m)) for n, m in zip(counts, created)]
    result = []
    for selected in product(*choices):
        order, sign, offset = [], 1, 0
        for n, chosen in zip(counts, selected):
            perm = chosen+tuple(i for i in range(n) if i not in chosen)
            sign *= _parity(perm)
            order.extend(offset+i for i in perm)
            offset += n
        result.append((sign, tuple(np.argsort(order))))
    return tuple(result)


@dataclass(frozen=True)
class TensorAction:
    space: TensorSpace
    kernels: tuple
    plans: tuple

    def __call__(self, vector):
        return self.apply(vector)

    def apply(self, vector, *, max_input_rank=None, max_output_rank=None):
        """Apply W with exact static input/output excitation-rank projections.

        Discarded output blocks are skipped before contraction, not zeroed
        after a full dense antisymmetric tensor has already been produced.
        The packed vector's full layout is preserved for Taylor recursion.
        """
        tensors = self.space.expand(vector, max_rank=max_input_rank)
        if max_input_rank is not None:
            max_output_rank = (max_input_rank+2 if max_output_rank is None
                               else min(max_output_rank, max_input_rank+2))
        result = [jnp.zeros(b.full_shape, vector.dtype)
                  if max_output_rank is None or sum(b.excitation) <= max_output_rank else None
                  for b in self.space.blocks]
        for source, target, kernel, axes, order, sign, shuffles in self.plans:
            if tensors[source] is None or result[target] is None:
                continue
            value = jnp.tensordot(self.kernels[kernel], tensors[source],
                                 axes=(tuple(range(self.kernels[kernel].ndim-len(axes), self.kernels[kernel].ndim)), axes))
            value = value.transpose(order)
            for phase, permutation in shuffles:
                result[target] = result[target]+sign*phase*value.transpose(permutation)
        return self.space.pack(result)


def build_tensor_action(eri, focks, space):
    """Build (F-offdiag)_N + V_N from one-/two-body normal-ordering rules."""
    groups = space.orbitals
    fs = focks if space.unrestricted else (focks[0], focks[0])
    offdiag = tuple(f-jnp.diag(jnp.diag(f)) for f in fs)
    if space.unrestricted:
        aa, ab, bb = eri
        gs = ((aa, ab), (ab.transpose(2, 3, 0, 1), bb))
    else:
        gs = ((eri, eri), (eri, eri))
    kernels, leaves = {}, {}
    lookup = {b.counts:i for i,b in enumerate(space.blocks)}

    def needed(types, order, daggers):
        m,l,_,_ = _normal_order(types,order,daggers)
        return any(all(s >= a for s,a in zip(b.counts,l))
                   and tuple(s-a+c for s,a,c in zip(b.counts,l,m)) in lookup
                   for b in space.blocks)

    def leaf(first, second, labels):
        tensor = gs[first][second]
        key = ((first, second) if space.unrestricted else (0, 0), tuple(groups[t] for t in labels))
        if key not in leaves:
            leaves[key] = tensor[jnp.ix_(*(jnp.asarray(groups[t], dtype=jnp.int32) for t in labels))]
        return leaves[key]

    def add(types, tensor, order, daggers, prefactor):
        m, l, axes, sign = _normal_order(types, order, daggers)
        key = (m, l)
        value = prefactor*sign*prod(factorial(x) for x in m)*tensor.transpose(axes)
        kernels[key] = kernels.get(key, 0.)+value

    for types in product(range(4), repeat=2):
        if types[0] % 2 != types[1] % 2 or any(not groups[t] for t in types):
            continue
        if not needed(types,(0,1),(True,False)):
            continue
        f = offdiag[types[0] % 2]
        add(types, f[jnp.ix_(*(jnp.asarray(groups[t], dtype=jnp.int32) for t in types))],
            (0, 1), (True, False), 1.)
    for types in product(range(4), repeat=4):
        spins = tuple(t % 2 for t in types)
        if sorted(spins[:2]) != sorted(spins[2:]) or any(not groups[t] for t in types):
            continue
        if not needed(types,(0,1,3,2),(True,True,False,False)):
            continue
        value = 0.
        if spins[0] == spins[2] and spins[1] == spins[3]:
            value = leaf(spins[0], spins[1], (types[0], types[2], types[1], types[3])).transpose(0, 2, 1, 3)
        if spins[0] == spins[3] and spins[1] == spins[2]:
            value = value-leaf(spins[0], spins[1], (types[0], types[3], types[1], types[2])).transpose(0, 2, 3, 1)
        add(types, value, (0, 1, 3, 2), (True, True, False, False), .25)

    keys = tuple(kernels)
    plans = []
    for source, block in enumerate(space.blocks):
        for index, (m, l) in enumerate(keys):
            remaining = tuple(s-a for s,a in zip(block.counts, l))
            if min(remaining) < 0:
                continue
            counts = tuple(r+a for r,a in zip(remaining, m))
            if counts not in lookup:
                continue
            axes, offset = [], 0
            for count, contracted in zip(block.counts, l):
                axes.extend(range(offset, offset+contracted))
                offset += count
            order, new_offset, old_offset = [], 0, sum(m)
            for created, old in zip(m, remaining):
                order.extend(range(new_offset, new_offset+created))
                order.extend(range(old_offset, old_offset+old))
                new_offset += created
                old_offset += old
            sign = (-1)**sum(remaining[t]*(l[u]+m[u]) for t in range(4) for u in range(t+1, 4))
            plans.append((source, lookup[counts], index, tuple(axes), tuple(order), sign, _shuffles(counts, m)))
    return TensorAction(space, tuple(kernels[k] for k in keys), tuple(plans))
