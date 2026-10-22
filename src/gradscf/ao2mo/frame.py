"""Explicit orbital frames for differentiable fixed orbital selections."""
from numbers import Integral

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def project_frame(coefficients, anchor, overlap, nocc):
    """Project fixed anchor O/V blocks into the current stationary subspaces.

    For each block A, Q=C_A(C_A.T S A0) and C_frame=Q chol(Q.T S Q)^-T.
    This fixes the otherwise arbitrary rotations inside equal-occupation
    spaces. It preserves their projectors and every continuous input's AD.
    A rank-losing anchor projection yields nonfinite Cholesky results; no
    clipping, canonicalization or fallback to another frame is performed.
    """
    c, a, s = map(jnp.asarray, (coefficients, anchor, overlap))
    if c.ndim != 2 or a.shape != c.shape or s.shape != (c.shape[0],)*2:
        raise ValueError('Orbital frame requires matching (nao,nmo) C/anchor and square overlap')
    if any(jnp.iscomplexobj(x) or not jnp.issubdtype(x.dtype, jnp.floating) for x in (c,a,s)):
        raise ValueError('Orbital frame requires real floating-point arrays')
    if not isinstance(nocc, Integral) or isinstance(nocc, bool) or not 0 <= nocc <= c.shape[1]:
        raise ValueError('nocc must select a static occupied/virtual partition')
    blocks = []
    for selection in (slice(None,nocc), slice(nocc,None)):
        current, initial = c[:,selection], a[:,selection]
        if not current.shape[1]:
            blocks.append(current)
            continue
        q = current@(current.T@s@initial)
        gram = q.T@s@q
        lower = jnp.linalg.cholesky(.5*(gram+gram.T))
        blocks.append(solve_triangular(lower, q.T, lower=True).T)
    return jnp.concatenate(blocks, axis=1)
