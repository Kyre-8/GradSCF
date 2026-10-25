"""Budgeted small-density reference formula, shared with the sigma fallback."""
from itertools import combinations
from math import comb

import jax.numpy as jnp
import numpy as np

from .._cache import array_cache
from .._memory import workspace_bytes
from ..fci.cistring import excite
from .space import UCISpace


_CACHE_BYTES = 32_000_000


def _connection_bound(space, rank):
    n = space.nmo
    a,b = space.nocc if isinstance(space,UCISpace) else (space.nocc,)*2
    if rank == 1:
        per_state = a*(n-a+1)+b*(n-b+1)
    else:
        per_state = (4*comb(a,2)*comb(n-a+2,2)+4*comb(b,2)*comb(n-b+2,2)
                     +a*b*(n-a+1)*(n-b+1))
    return space.size*per_state


def affordable(space, rank, dtype, max_memory, *, resident_bytes, device=None):
    """Check table, construction and output bounds before enumerating links."""
    links = _connection_bound(space,rank)
    table = links*(2*rank+4)*np.dtype(np.int32).itemsize
    if links > 4_000_000 or table > _CACHE_BYTES:
        return False
    item = np.dtype(dtype).itemsize
    # 512 bytes/row covers host tuple/list/boxed-index construction. Retain
    # host/backend tables, gathered coefficients, values and output versions.
    known = resident_bytes+links*(512+4*item)+2*table
    known += 4*(rank+1)*space.nmo**(2*rank)*item
    try:
        workspace_bytes(max_memory,resident_bytes=known,device=device)
    except ValueError:
        return False
    return True


@array_cache(maxsize=8, max_bytes=_CACHE_BYTES)
def _density_connections(space, rank):
    n = space.nmo
    lookup = {d: i for i, d in enumerate(space.determinants)}
    links = []
    for ket, det in enumerate(space.determinants):
        occupied = [p for p in range(2*n) if det & (1 << p)]
        for holes in combinations(occupied, rank):
            remaining, _ = excite(det, holes, ())
            available = [p for p in range(2*n) if not remaining & (1 << p)]
            for particles in combinations(available, rank):
                if sum(p//n for p in particles) != sum(q//n for q in holes):
                    continue
                target, sign = excite(det, holes, particles)
                bra = lookup.get(target)
                if bra is None:
                    continue
                if rank == 1:
                    p, q = particles[0], holes[0]
                    links.append((p//n, bra, ket, q % n, p % n, sign))
                else:
                    p, r = particles
                    q, s = holes
                    block = p//n + r//n
                    p, q, r, s = p % n, q % n, r % n, s % n
                    links.append((block, bra, ket, p, q, r, s, sign))
                    if block != 1:
                        links.extend(((block, bra, ket, r, q, p, s, -sign),
                                      (block, bra, ket, p, s, r, q, -sign),
                                      (block, bra, ket, r, s, p, q, sign)))
                if len(links) > 4_000_000:
                    raise ValueError("CI density connection limit exceeded; reduce the orbital space")
    return np.asarray(links, dtype=np.int32).reshape(-1, 2*rank+4)


def density(coefficients, space, rank, dtype):
    """Original fermionic connection/scatter formula; coefficients stay live."""
    c = jnp.asarray(coefficients).astype(dtype)
    norm = jnp.vdot(c,c).real
    links = _density_connections(space,rank)
    blocks = jnp.zeros((rank+1,)+(space.nmo,)*(2*rank),c.dtype)
    if len(links):
        block,bra,ket,*indices,signs = links.T
        values = signs*c[bra]*c[ket]/jnp.where(norm > 0,norm,1.)
        blocks = blocks.at[(block,*indices)].add(values)
    blocks = jnp.where((norm > 0) & jnp.isfinite(norm),blocks,jnp.nan)
    if isinstance(space,UCISpace):
        return tuple(blocks[i] for i in range(rank+1))
    if rank == 1:
        return blocks[0]+blocks[1]
    return blocks[0]+blocks[1]+blocks[1].transpose(2,3,0,1)+blocks[2]
