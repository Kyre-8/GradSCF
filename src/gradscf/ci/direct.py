"""Direct one-/two-body actions on compact particle/hole string blocks.

The four groups are virtual alpha/beta and occupied alpha/beta holes. Each
group stores increasing-index strings, never a dense antisymmetric tensor.
Removing/adding at most two indices factorizes the action into small string
maps and integral contractions. Projection applies to the complete operator;
no projected single-excitation operator is multiplied by another one.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations, permutations, product
from math import comb, prod
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.ao2mo import validate_integrals, validate_unrestricted_integrals
from ..fci.cistring import excite
from ..solvers import LinearOperator
from ..solvers.memory import apply_columns
from .._memory import array_storage_bytes, validate_reserved_bytes, workspace_bytes
from .space import CISpace, UCISpace


def _parity(order):
    return (-1)**sum(a > b for i, a in enumerate(order) for b in order[i+1:])


@lru_cache(maxsize=128)
def _choices(n, rank):
    return np.asarray(list(combinations(range(n), rank)), dtype=np.int32).reshape(comb(n, rank), rank)


@dataclass(frozen=True)
class _Block:
    counts: tuple
    shape: tuple
    indices: object
    signs: object


@dataclass(frozen=True)
class _Topology:
    orbitals: tuple
    blocks: tuple
    nocc: tuple


@lru_cache(maxsize=8)
def _topology(space):
    if not isinstance(space, (CISpace, UCISpace)):
        raise TypeError("Direct CI requires a CISpace or UCISpace")
    n = space.nmo
    ns = space.nocc if isinstance(space, UCISpace) else (space.nocc,)*2
    frozen = space.frozen if isinstance(space, UCISpace) else (space.frozen,)*2
    occupied = tuple(tuple(i for i in range(no) if i not in fr) for no, fr in zip(ns, frozen))
    virtual = tuple(tuple(i for i in range(no, n) if i not in fr) for no, fr in zip(ns, frozen))
    groups = virtual + occupied
    reference = ((1 << ns[0])-1) | (((1 << ns[1])-1) << n)
    lookup = {d: i for i, d in enumerate(space.determinants)}
    ranks = sorted({(((d & ((1 << n)-1)) >> ns[0]).bit_count(),
                      ((d >> n) >> ns[1]).bit_count()) for d in space.determinants},
                   key=lambda x: (sum(x), x))
    blocks = []
    for a, b in ranks:
        counts = (a, b, a, b)
        choices = tuple(_choices(len(g), r) for g, r in zip(groups, counts))
        indices, signs = [], []
        for selected in product(*choices):
            det, phase = reference, 1
            for group in reversed(range(4)):
                for i in reversed(selected[group]):
                    p = groups[group][i] + (group % 2)*n
                    phase *= -1 if (det & ((1 << p)-1)).bit_count() % 2 else 1
                    det ^= 1 << p
            if det not in lookup:
                raise ValueError("Direct CI requires complete retained excitation-rank blocks")
            indices.append(lookup[det])
            signs.append(phase)
        blocks.append(_Block(counts, tuple(map(len, choices)),
                             np.asarray(indices, dtype=np.int32), np.asarray(signs, dtype=np.int8)))
    if sum(len(b.indices) for b in blocks) != space.size:
        raise ValueError("CI determinants do not match the requested frozen excitation space")
    return _Topology(groups, tuple(blocks), tuple(ns))


@lru_cache(maxsize=256)
def _annihilation_map(n, rank, removed):
    """(removed string, spectator string) -> source string and fermion sign."""
    holes, spectators = _choices(n, removed), _choices(n, rank-removed)
    hole_lookup = {tuple(c): i for i, c in enumerate(holes)}
    spectator_lookup = {sum(1 << int(p) for p in c): i for i, c in enumerate(spectators)}
    indices = np.zeros((len(holes), len(spectators)), dtype=np.int32)
    signs = np.zeros_like(indices, dtype=np.int8)
    for source, occupied in enumerate(_choices(n, rank)):
        string = sum(1 << int(p) for p in occupied)
        for chosen in combinations(map(int, occupied), removed):
            target, phase = excite(string, chosen, ())
            row, col = hole_lookup[chosen], spectator_lookup[target]
            indices[row, col], signs[row, col] = source, phase
    return indices, signs


def _normal_order(types, order, daggers):
    create = {i: daggers[i] != (types[i] >= 2) for i in order}
    arranged = sorted(order, key=lambda i: (0, types[i]) if create[i] else (1, -types[i]))
    creators = tuple(i for i in arranged if create[i])
    annihilators = tuple(reversed([i for i in arranged if not create[i]]))
    m = tuple(sum(types[i] == g for i in creators) for g in range(4))
    l = tuple(sum(types[i] == g for i in annihilators) for g in range(4))
    return m, l, creators+annihilators, _parity([order.index(i) for i in arranged])


def _within(counts, limit):
    return limit is None or sum(counts[:2]) <= limit


def _needed(topology, m, l, max_input_rank, max_output_rank):
    targets = {b.counts for b in topology.blocks}
    for b in topology.blocks:
        if not _within(b.counts, max_input_rank):
            continue
        remaining = tuple(r-a for r, a in zip(b.counts, l))
        target = tuple(r+c for r, c in zip(remaining, m))
        if min(remaining) >= 0 and target in targets and _within(target, max_output_rank):
            return True
    return False


def _compressed_kernel(tensor, types, m, l, axes, groups):
    """Antisymmetrize only the operator indices, then take increasing strings."""
    counts = m+l
    choices = tuple(_choices(len(groups[g % 4]), r) for g, r in enumerate(counts))
    positions = []
    for g, r in enumerate(counts):
        for column in range(r):
            positions.append((g, column))
    value = 0.
    for orders in product(*(tuple(permutations(range(r))) for r in counts)):
        indices = [None]*len(types)
        for label, (group, column) in zip(axes, positions):
            shape = [1]*8
            shape[group] = len(choices[group])
            selected = choices[group][:, orders[group][column]]
            indices[label] = np.asarray(groups[group % 4])[selected].reshape(shape)
        phase = prod(_parity(order) for order in orders)
        value = value + phase*tensor[tuple(indices)]
    return value.reshape(prod(map(len, choices[:4])), prod(map(len, choices[4:])))


def _kernels(focks, gs, topology, block_diagonal=False, max_input_rank=None,
             max_output_rank=None, cross_only=False, block_source=None):
    groups = topology.orbitals
    kernels = {}
    for degree in ((2, 4) if gs is not None else (2,)):
        for types in product(range(4), repeat=degree):
            if any(not groups[g] for g in types):
                continue
            if degree == 2:
                if types[0] % 2 != types[1] % 2:
                    continue
                if block_diagonal and types[0] != types[1]:
                    continue
                if cross_only and types[0] == types[1]:
                    continue
                tensor, factor = focks[types[0] % 2], 1.
                order, daggers = (0, 1), (True, False)
            else:
                if types[0] % 2 != types[1] % 2 or types[2] % 2 != types[3] % 2:
                    continue
                tensor, factor = gs[types[0] % 2][types[2] % 2], .5
                order, daggers = (0, 2, 3, 1), (True, False, True, False)
            m, l, axes, phase = _normal_order(types, order, daggers)
            if (any(r > len(groups[g % 4]) for g, r in enumerate(m+l)) or
                    not _needed(topology, m, l, max_input_rank, max_output_rank)):
                continue
            selected_groups = groups
            if degree == 4 and block_source is not None:
                tensor = block_source(types)
                selected_groups = tuple(tuple(range(len(group))) for group in groups)
            value = factor*phase*_compressed_kernel(tensor, types, m, l, axes, selected_groups)
            key = (m, l)
            kernels[key] = kernels.get(key, 0.) + value
    return kernels


def _strides(shape):
    return tuple(prod(shape[i+1:]) for i in range(len(shape)))


def _map_indices(maps, rows, spectators, strides):
    shape = (rows[0].size, spectators[0].size)
    indices, signs = jnp.zeros(shape, dtype=jnp.int32), jnp.ones(shape, dtype=jnp.int8)
    for g, ((source, phase), stride) in enumerate(zip(maps, strides)):
        index = (rows[g][:, None], spectators[g][None, :])
        indices += stride*jnp.asarray(source)[index]
        signs *= jnp.asarray(phase)[index]
    return indices, signs


def _contract(kernel, coefficients, source, target, m, l, dimensions, block_size,
               rematerialize=False):
    remaining = tuple(r-a for r, a in zip(source.counts, l))
    left = tuple(comb(n, r) for n, r in zip(dimensions, l))
    right = tuple(comb(n, r) for n, r in zip(dimensions, m))
    spectators = tuple(comb(n, r) for n, r in zip(dimensions, remaining))
    remove = tuple(_annihilation_map(n, r, a) for n, r, a in zip(dimensions, source.counts, l))
    add = tuple(_annihilation_map(n, r, a) for n, r, a in zip(dimensions, target.counts, m))
    rows = jnp.unravel_index(jnp.arange(prod(left)), left)
    width = coefficients.shape[1]
    nt, nr = prod(spectators), prod(right)
    tc, rc = min(block_size, nt), min(block_size, nr)
    cross_phase = (-1)**sum(remaining[g]*(l[h]+m[h]) for g in range(4) for h in range(g+1, 4))

    def spectator_block(t, result):
        columns = t*tc+jnp.arange(tc)
        spectator = jnp.unravel_index(jnp.minimum(columns, nt-1), spectators)
        index, sign = _map_indices(remove, rows, spectator, _strides(source.shape))
        rhs = coefficients[index]*sign[:, :, None]*(columns < nt)[None, :, None]
        rhs = rhs.reshape(prod(left), tc*width)

        def created_block(r, current):
            created = r*rc+jnp.arange(rc)
            choice = jnp.unravel_index(jnp.minimum(created, nr-1), right)
            index, sign = _map_indices(add, choice, spectator, _strides(target.shape))
            values = (kernel[jnp.minimum(created, nr-1)]@rhs).reshape(rc, tc, width)
            mask = (created < nr)[:, None] & (columns < nt)[None, :]
            return current.at[index].add(values*(cross_phase*sign*mask)[:, :, None])

        body = jax.checkpoint(created_block) if rematerialize else created_block
        return jax.lax.fori_loop(0, (nr+rc-1)//rc, body, result)

    initial = jnp.zeros((prod(target.shape), width), dtype=coefficients.dtype)
    body = jax.checkpoint(spectator_block) if rematerialize else spectator_block
    return jax.lax.fori_loop(0, (nt+tc-1)//tc, body, initial)


def _reference(hs, gs, nocc):
    focks = list(hs)
    constant = sum(jnp.trace(h[:no, :no]) for h, no in zip(hs, nocc))
    if gs is None:
        return tuple(focks), constant
    for spin in (0, 1):
        for other in (0, 1):
            no = nocc[other]
            g = gs[spin][other]
            potential = jnp.einsum("pqii->pq", g[:, :, :no, :no])
            if spin == other:
                potential -= jnp.einsum("piiq->pq", g[:, :no, :no, :])
            focks[spin] = focks[spin]+potential
            constant += .5*jnp.trace(potential[:nocc[spin], :nocc[spin]])
    return tuple(focks), constant


def _diagonal(focks, gs, constant, topology, size, max_input_rank=None, max_output_rank=None,
              interactions=None):
    result = jnp.zeros(size, dtype=jnp.result_type(*focks))
    groups = topology.orbitals
    for block in topology.blocks:
        if not _within(block.counts, max_input_rank) or not _within(block.counts, max_output_rank):
            continue
        choices = tuple(_choices(len(g), r) for g, r in zip(groups, block.counts))
        orbitals = tuple(np.asarray(g, dtype=np.int32)[c] for g, c in zip(groups, choices))
        value = jnp.full(block.shape, constant)
        for g, chosen in enumerate(orbitals):
            shape = [1]*4
            shape[g] = len(chosen)
            term = jnp.sum(jnp.diag(focks[g % 2])[chosen], axis=1)
            value += (1 if g < 2 else -1)*term.reshape(shape)
        if gs is not None:
            for g in range(4):
                for h in range(g, 4):
                    if interactions is None:
                        tensor = gs[g % 2][h % 2]
                        interaction = jnp.einsum("ppqq->pq", tensor)
                        if g % 2 == h % 2:
                            interaction -= jnp.einsum("pqqp->pq", tensor)
                        p, q = orbitals[g], orbitals[h]
                    else:
                        interaction = interactions[g, h]
                        p, q = choices[g], choices[h]
                    if g == h:
                        term = .5*jnp.sum(interaction[p[:, :, None], p[:, None, :]], axis=(1, 2))
                        shape = [1]*4
                        shape[g] = len(p)
                    else:
                        term = jnp.sum(interaction[p[:, :, None, None], q[None, None, :, :]], axis=(1, 3))
                        term *= (1 if g < 2 else -1)*(1 if h < 2 else -1)
                        shape = [1]*4
                        shape[g], shape[h] = len(p), len(q)
                    value += term.reshape(shape)
        result = result.at[block.indices].set(value.reshape(-1))
    return result


@lru_cache(maxsize=16)
def _action(space, block_size, keys, max_input_rank, max_output_rank,
             rematerialize=False):
    """Share the numerical sigma graph across Hamiltonians on one topology."""
    topology = _topology(space)
    lookup = {b.counts: i for i, b in enumerate(topology.blocks)}
    dimensions = tuple(map(len, topology.orbitals))
    plans = []
    for source, block in enumerate(topology.blocks):
        if not _within(block.counts, max_input_rank):
            continue
        for index, (m, l) in enumerate(keys):
            remaining = tuple(r-a for r, a in zip(block.counts, l))
            counts = tuple(r+a for r, a in zip(remaining, m))
            if min(remaining) >= 0 and counts in lookup and _within(counts, max_output_rank):
                plans.append((source, lookup[counts], m, l, index))

    @jax.jit
    def columns(kernels, constant, vectors):
        vectors = vectors.astype(jnp.result_type(vectors, constant, *kernels))
        states = tuple(vectors[b.indices]*jnp.asarray(b.signs)[:, None]
                       if _within(b.counts, max_input_rank) else None for b in topology.blocks)
        outputs = [jnp.zeros((len(b.indices), vectors.shape[1]), vectors.dtype)
                   if _within(b.counts, max_output_rank) else None for b in topology.blocks]
        for i, state in enumerate(states):
            if state is not None and outputs[i] is not None:
                outputs[i] += constant*state
        for source, target, m, l, index in plans:
            outputs[target] += _contract(kernels[index], states[source], topology.blocks[source],
                                         topology.blocks[target], m, l, dimensions, block_size,
                                         rematerialize)
        result = jnp.zeros_like(vectors)
        for b, value in zip(topology.blocks, outputs):
            if value is None:
                continue
            result = result.at[b.indices].add(value*jnp.asarray(b.signs)[:, None])
        return result

    return columns


def _kernel_bytes(topology, two_body, dtype, max_input_rank, max_output_rank):
    """Static compressed-kernel payload, before any numerical contraction."""
    groups, sizes, keys = topology.orbitals, tuple(map(len, topology.orbitals)), set()
    for degree in ((2, 4) if two_body else (2,)):
        for types in product(range(4), repeat=degree):
            if any(not groups[g] for g in types) or any(types[i] % 2 != types[i+1] % 2 for i in range(0,degree,2)):
                continue
            order, daggers = ((0,1),(True,False)) if degree == 2 else ((0,2,3,1),(True,False,True,False))
            m,l,_,_ = _normal_order(types,order,daggers)
            if any(r > sizes[g % 4] for g,r in enumerate(m+l)) or not _needed(topology,m,l,max_input_rank,max_output_rank):
                continue
            keys.add((m,l))
    sizes = [prod(comb(sizes[g % 4],r) for g,r in enumerate(m+l)) for m,l in keys]
    return (sum(sizes)+3*max(sizes,default=0))*jnp.dtype(dtype).itemsize


def _action_layout(space, keys, max_input_rank=None, max_output_rank=None):
    """Static maxima and distinct map constants for the existing sigma graph."""
    topology = _topology(space)
    dimensions = tuple(map(len, topology.orbitals))
    counts = {b.counts for b in topology.blocks}
    maximum_left = maximum_spectators = maximum_created = 1
    tables = set()
    for block in topology.blocks:
        if not _within(block.counts,max_input_rank):
            continue
        for m,l in keys:
            remaining = tuple(r-a for r,a in zip(block.counts,l))
            target = tuple(r+a for r,a in zip(remaining,m))
            if min(remaining) < 0 or target not in counts or not _within(target,max_output_rank):
                continue
            maximum_left = max(maximum_left,prod(comb(n,a) for n,a in zip(dimensions,l)))
            maximum_created = max(maximum_created,prod(comb(n,a) for n,a in zip(dimensions,m)))
            maximum_spectators = max(maximum_spectators,prod(comb(n,a) for n,a in zip(dimensions,remaining)))
            tables.update((n,r,a) for n,r,a in zip(dimensions,block.counts,l))
            tables.update((n,r,a) for n,r,a in zip(dimensions,target,m))
    maps = 10*sum(comb(n,a)*comb(n,r-a) for n,r,a in tables)+10*space.size
    return maximum_left,maximum_spectators,maximum_created,maps


def _budgeted_columns(space, kernels, constant, vectors, block_size, max_memory,
                       reserved_bytes, max_input_rank=None, max_output_rank=None):
    """Select creator/spectator and RHS widths from one forward-array budget."""
    if vectors.shape[1] == 0:
        return jnp.zeros_like(vectors)
    maximum_left,maximum_spectators,maximum_created,maps = _action_layout(
        space,tuple(kernels),max_input_rank,max_output_rank)
    dtype = jnp.result_type(vectors,constant,*kernels.values())
    item = dtype.itemsize
    resident = reserved_bytes+maps+array_storage_bytes(vectors,constant,*kernels.values())
    resident += 6*space.size*vectors.shape[1]*item
    available = workspace_bytes(max_memory,resident_bytes=resident,device=getattr(vectors,'device',None))
    best = None
    widths = [1]
    while widths[-1] < block_size:
        widths.append(min(block_size,2*widths[-1]))
    for width in widths:
        tc,rc = min(width,maximum_spectators),min(width,maximum_created)
        fixed = (maximum_left*tc+rc*tc)*24+rc*maximum_left*item
        per_rhs = item*(2*maximum_left*tc+3*rc*tc)
        batch = min(vectors.shape[1],(available-fixed)//max(1,per_rhs))
        if batch >= 1:
            score = ((maximum_spectators+tc-1)//tc*(maximum_created+rc-1)//rc
                     *((vectors.shape[1]+batch-1)//batch),-batch,-width)
            if best is None or score < best[0]:
                best = score,width,int(batch)
    if best is None:
        raise ValueError('Minimum direct CI string/RHS tile exceeds the working-memory budget')
    _,width,batch = best
    columns = _action(space,width,tuple(kernels),max_input_rank,max_output_rank,True)
    values = tuple(kernels.values())
    op = LinearOperator((space.size,)*2,dtype,
        lambda x: columns(values,constant,x[:,None])[:,0],
        matmat=lambda x: columns(values,constant,x))
    return apply_columns(op,vectors,batch)


def _operator(hs, gs, space, block_size, block_diagonal=False,
              max_input_rank=None, max_output_rank=None, fluctuation=False,
              max_memory=None, reserved_bytes=0):
    if not isinstance(block_size, Integral) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    topology = _topology(space)
    if max_memory is not None:
        reserved_bytes = validate_reserved_bytes(reserved_bytes)
        reserved_bytes += array_storage_bytes(*hs,*jax.tree.leaves(gs))
        workspace_bytes(max_memory,resident_bytes=reserved_bytes+
            _kernel_bytes(topology,gs is not None,jnp.result_type(*hs,*jax.tree.leaves(gs)),
                          max_input_rank,max_output_rank)+32*space.nmo**2+64*space.size)
    focks, constant = _reference(hs, gs, topology.nocc)
    if fluctuation:
        cross = []
        for fock, no in zip(focks, topology.nocc):
            occupied = jnp.arange(fock.shape[0]) < no
            cross.append(jnp.where(occupied[:, None] != occupied[None, :], fock, 0.))
        focks, constant = tuple(cross), jnp.zeros_like(constant)
    kernels = _kernels(focks, gs, topology, block_diagonal, max_input_rank,
                       max_output_rank, cross_only=fluctuation)
    columns = _action(space, int(block_size), tuple(kernels), max_input_rank, max_output_rank)
    values = tuple(kernels.values())

    def matvec(vector):
        if max_memory is not None:
            return _budgeted_columns(space,kernels,constant,vector[:,None],block_size,
                max_memory,reserved_bytes,max_input_rank,max_output_rank)[:,0]
        return columns(values, constant, vector[:, None])[:, 0]

    def matmat(vectors):
        if max_memory is not None:
            return _budgeted_columns(space,kernels,constant,vectors,block_size,
                max_memory,reserved_bytes,max_input_rank,max_output_rank)
        if vectors.shape[1] == 0:
            return jnp.zeros_like(vectors)
        return apply_columns(unbatched, vectors, 4)

    dtype = jnp.result_type(*focks)
    unbatched = LinearOperator((space.size, space.size), dtype, matvec,
        matmat=lambda vectors: columns(values, constant, vectors), transpose_matvec=matvec)
    return replace(unbatched, matmat=matmat,
        diagonal=_diagonal(focks, gs, constant, topology, space.size,
                           max_input_rank, max_output_rank))


def from_prepared(integrals, space, *, block_size=128,max_memory=None,reserved_bytes=0,
                   part='hamiltonian',max_input_rank=None,max_output_rank=None):
    """Apply H, normal-ordered W or block-diagonal F_N from shared kernels.

    Rank projectors and transposes use the original compact sigma action. W
    drops the reference constant and same-group one-body F_N kernels; it keeps
    every quartic and occupied/virtual cross-Fock term. No new normal-ordering
    or perturbative partition is introduced.
    """
    if not isinstance(block_size, Integral) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError('block_size must be a positive integer')
    if integrals.diagonal.shape != (space.size,):
        raise ValueError('Prepared CI diagonal does not match the determinant space')
    if part not in ('hamiltonian','fluctuation','fock'):
        raise ValueError('part must be hamiltonian, fluctuation or fock')
    for value in (max_input_rank,max_output_rank):
        if value is not None and (not isinstance(value,Integral) or isinstance(value,bool) or value < 0):
            raise ValueError('Rank limits must be nonnegative integers or None')
    kernels = integrals.kernels
    constant,diagonal = integrals.constant,integrals.diagonal
    if part != 'hamiltonian':
        fock = lambda key: sum(key[0])+sum(key[1]) == 2 and key[0] == key[1]
        kernels = {key:value for key,value in kernels.items() if fock(key) == (part == 'fock')}
        constant = jnp.zeros_like(constant)
        diagonal = (integrals.fock_diagonal if part == 'fock' else
                    integrals.diagonal-integrals.constant-integrals.fock_diagonal)
    if max_input_rank is not None or max_output_rank is not None:
        ranks = jnp.asarray(space.ranks)
        keep = ((ranks <= max_input_rank) if max_input_rank is not None else True)
        keep = keep & ((ranks <= max_output_rank) if max_output_rank is not None else True)
        diagonal = jnp.where(keep,diagonal,0.)
    columns = _action(space,int(block_size),tuple(kernels),max_input_rank,max_output_rank)
    values = tuple(kernels.values())
    dtype = jnp.result_type(constant, *values)
    def apply(x):
        if max_memory is not None:
            return _budgeted_columns(space,kernels,constant,x,block_size,
                max_memory,reserved_bytes,max_input_rank,max_output_rank)
        return columns(values,constant,x)
    matvec = lambda x: apply(x[:,None])[:,0]
    unbatched = LinearOperator((space.size,)*2, dtype, matvec,
        diagonal=diagonal, matmat=lambda x: columns(values, constant, x),
        transpose_matvec=matvec)
    operator = replace(unbatched, matmat=apply if max_memory is not None else
                       lambda x: apply_columns(unbatched, x, 4))
    if max_input_rank is not None or max_output_rank is not None:
        return replace(operator,transpose_matvec=lambda x: from_prepared(integrals,space,
            block_size=block_size,max_memory=max_memory,reserved_bytes=reserved_bytes,part=part,
            max_input_rank=max_output_rank,max_output_rank=max_input_rank).matvec(x))
    return operator


def _from_integrals(h1, eri, space, block_size, max_input_rank, max_output_rank,
                    *, fluctuation=False, max_memory=None, reserved_bytes=0):
    """Shared validation and transpose assembly for H and its fluctuation."""
    if isinstance(space, UCISpace):
        hs, (aa, ab, bb) = validate_unrestricted_integrals(h1, eri)
        if hs[0].shape[0] != space.nmo:
            raise ValueError("MO integral dimensions do not match the MO space")
        gs = ((aa, ab), (ab.transpose(2, 3, 0, 1), bb))
    else:
        h, g = validate_integrals(h1, eri, space.nmo)
        hs, gs = (h, h), ((g, g), (g, g))
    limits = []
    for name, value in (("max_input_rank", max_input_rank), ("max_output_rank", max_output_rank)):
        if value is not None and (not isinstance(value, Integral) or isinstance(value, bool) or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer or None")
        limits.append(None if value is None else int(value))
    max_input_rank, max_output_rank = limits
    operator = _operator(hs, gs, space, block_size, max_input_rank=max_input_rank,
                          max_output_rank=max_output_rank, fluctuation=fluctuation,
                          max_memory=max_memory,reserved_bytes=reserved_bytes)
    if max_input_rank is None and max_output_rank is None:
        return operator
    def transpose(vector):
        return _operator(hs, gs, space, block_size, max_input_rank=max_output_rank,
                         max_output_rank=max_input_rank, fluctuation=fluctuation,
                         max_memory=max_memory,reserved_bytes=reserved_bytes).matvec(vector)
    return replace(operator, transpose_matvec=transpose)


def build_direct_hamiltonian(h1, eri, space, *, block_size=128,
                             max_input_rank=None, max_output_rank=None,
                             max_memory=None, reserved_bytes=0):
    """Physical real CI Hamiltonian without determinant-pair connections.

    Frozen electrons enter the reference constant and Fock matrix. ``block_size``
    bounds creator/spectator string tiles; it does not truncate the CI space.
    All integral contractions and the resulting action remain differentiable.
    Optional static rank limits return P_out H P_in in the same vector layout.
    Its transpose exchanges these projectors. No amplitude threshold is used.
    """
    return _from_integrals(h1, eri, space, block_size, max_input_rank, max_output_rank,
                          max_memory=max_memory,reserved_bytes=reserved_bytes)


def build_fluctuation(h1, eri, space, *, block_size=128,
                      max_input_rank=None, max_output_rank=None,
                      max_memory=None, reserved_bytes=0):
    """Normal-ordered W = H - E_ref - blockdiag(F)_N, in the same CI space.

    Retains cross occupied/virtual Fock terms and normal-ordered quartic terms.
    Frozen electrons contribute to F before the active-space projection; the
    complete reference constant is removed. No same-block Fock kernels or
    cancelling copies of F_N are constructed. Rank projections and their
    transposes use the same shared direct contraction graph as physical H.
    """
    return _from_integrals(h1, eri, space, block_size, max_input_rank, max_output_rank,
                           fluctuation=True,max_memory=max_memory,reserved_bytes=reserved_bytes)


def build_one_body(h1, space, *, block_size=128, block_diagonal=False):
    """Physical sum h[pq] a_p^dagger a_q in the same retained CI coordinates.

    Includes the occupied reference constant. Subtract that constant when a
    normal-ordered Fock operator is required. No two-electron tensor is built.
    A nonsymmetric real matrix is allowed; its transpose action uses h.T.
    ``block_diagonal=True`` explicitly projects h onto its occupied/virtual
    diagonal blocks, removing the cross-block kernels from the sigma graph.
    """
    hs = tuple(map(jnp.asarray, h1)) if isinstance(space, UCISpace) else (jnp.asarray(h1),)*2
    if len(hs) != 2 or any(h.shape != (space.nmo, space.nmo) for h in hs):
        raise ValueError("One-body matrices must match the CI orbital dimensions")
    if any(jnp.iscomplexobj(h) for h in hs):
        raise NotImplementedError("Direct CI requires real one-body matrices")
    if any(not jnp.issubdtype(h.dtype, jnp.floating) for h in hs):
        raise ValueError("One-body matrices must have floating-point dtype")
    if not isinstance(block_diagonal, (bool, np.bool_)):
        raise TypeError("block_diagonal must be a static boolean")
    operator = _operator(hs, None, space, block_size, block_diagonal)
    def transpose(vector):
        return _operator(tuple(h.T for h in hs), None, space, block_size, block_diagonal).matvec(vector)
    return replace(operator, transpose_matvec=transpose)
