"""Slater–Condon connections and differentiable real Hamiltonian actions.

H = sum h[p,q] a_p^+ a_q + 1/2 sum (pq|rs) a_p^+ a_r^+ a_s a_q.
The host builds integer connections once. All integral contractions use JAX.

Slater (1929), doi:10.1103/PhysRev.34.1293; Condon (1930),
doi:10.1103/PhysRev.36.1121. See ci/REFERENCES.md for scope and attribution.
"""
from itertools import combinations
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .space import UCISpace
from ..fci.cistring import excite
from gradscf.ao2mo import validate_integrals, spin_orbital_integrals
from .direct import build_direct_hamiltonian, build_one_body, build_fluctuation
from .._cache import array_cache


@array_cache(maxsize=8, max_bytes=32_000_000)
def _connections(space):
    n = space.nmo
    unrestricted = isinstance(space, UCISpace)
    frozen = (set(space.frozen[0]) | {p+n for p in space.frozen[1]} if unrestricted
              else set(space.frozen) | {p+n for p in space.frozen})
    orbital = (lambda p: p) if unrestricted else (lambda p: p % n)
    lookup = {d: i for i, d in enumerate(space.determinants)}
    rows, cols, one, two = [], [], [], []

    def link(row, col):
        if len(rows) >= 1_000_000:
            raise ValueError("CI connection limit (1000000) exceeded; reduce the orbital space")
        rows.append(row)
        cols.append(col)
        return len(rows) - 1

    def h(edge, p, q, sign=1):
        one.append((edge, orbital(p), orbital(q), sign))

    def g(edge, p, q, r, s, sign=1):
        two.append((edge, orbital(p), orbital(q), orbital(r), orbital(s), sign))

    # Diagonals occupy the first space.size connection slots.
    for j, det in enumerate(space.determinants):
        occ = [p for p in range(2 * n) if det & (1 << p)]
        edge = link(j, j)
        for p in occ:
            h(edge, p, p)
        for p, q in combinations(occ, 2):
            g(edge, p, p, q, q)
            if p // n == q // n:
                g(edge, p, q, q, p, -1)
    for j, det in enumerate(space.determinants):
        occ = [p for p in range(2 * n) if det & (1 << p)]
        holes = [p for p in occ if p not in frozen]
        vir = [p for p in range(2 * n) if not det & (1 << p) and p not in frozen]
        for q in holes:
            for p in vir:
                if p // n != q // n:
                    continue
                new, sign = excite(det, (q,), (p,))
                i = lookup.get(new, -1)
                if i <= j:
                    continue
                edge = link(i, j)
                h(edge, p, q, sign)
                for k in occ:
                    if k == q:
                        continue
                    g(edge, p, q, k, k, sign)
                    if p // n == k // n:
                        g(edge, p, k, k, q, -sign)
        for q, s in combinations(holes, 2):
            for p, r in combinations(vir, 2):
                if p // n + r // n != q // n + s // n:
                    continue
                new, sign = excite(det, (q, s), (p, r))
                i = lookup.get(new, -1)
                if i <= j:
                    continue
                edge = link(i, j)
                if p // n == q // n and r // n == s // n:
                    g(edge, p, q, r, s, sign)
                if p // n == s // n and r // n == q // n:
                    g(edge, p, s, r, q, -sign)
    return (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32),
            np.asarray(one, dtype=np.int32).reshape(-1, 4),
            np.asarray(two, dtype=np.int32).reshape(-1, 6))


class CIHamiltonian(NamedTuple):
    diagonal: object
    rows: object
    cols: object
    values: object

    def __call__(self, vectors):
        vectors = jnp.asarray(vectors)
        if vectors.ndim not in (1, 2) or vectors.shape[0] != self.diagonal.size:
            raise ValueError("CI vectors must have shape (ndet,) or (ndet, nvec)")
        if vectors.ndim == 1:
            return self(vectors[:, None])[:, 0]
        result = self.diagonal[:, None] * vectors
        result = result.at[self.rows].add(self.values[:, None] * vectors[self.cols])
        return result.at[self.cols].add(self.values[:, None] * vectors[self.rows])


def build_hamiltonian(h1, eri, space):
    if isinstance(space, UCISpace):
        h1, eri = spin_orbital_integrals(h1, eri)
        if h1.shape[0] != 2*space.nmo:
            raise ValueError("MO integral dimensions do not match the MO space")
    else:
        h1, eri = validate_integrals(h1, eri, space.nmo)
    rows, cols, one, two = _connections(space)
    weights = jnp.zeros((len(rows),), dtype=jnp.result_type(h1, eri))
    if len(one):
        e, p, q, sign = one.T
        weights = weights.at[e].add(sign * h1[p, q])
    if len(two):
        e, p, q, r, s, sign = two.T
        weights = weights.at[e].add(sign * eri[p, q, r, s])
    return CIHamiltonian(weights[:space.size], jnp.asarray(rows[space.size:]),
                         jnp.asarray(cols[space.size:]), weights[space.size:])


def hamiltonian_action(h1, eri, space, vectors):
    return build_hamiltonian(h1, eri, space)(vectors)


def hamiltonian_matrix(h1, eri, space):
    """Small-system oracle path. Normal Davidson solves never construct this."""
    if space.size > 2048:
        raise ValueError("Dense CI is limited to 2048 determinants; use Davidson")
    op = build_hamiltonian(h1, eri, space)
    return op(jnp.eye(space.size, dtype=op.diagonal.dtype))
