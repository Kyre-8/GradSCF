"""Checked forward-only complex TDA, using the shared Hermitian Davidson."""
from math import isfinite
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ...solvers.eigen.davidson import _davidson_lowest_symmetric
from .kernel import _eager


class TDAResult(NamedTuple):
    energies: object
    vectors: object
    residual_norms: object
    converged: object


def solve_tda(matrix, *, nstates=3, tol=1e-8, maxiter=100, max_subspace=None, seed=0):
    """Solve a bounded Hermitian matrix; vectors are columns with unit norm.

    The shared complex numerical Davidson is reused. The public shared
    implicit spectral API currently requires real data, so this entry point
    explicitly rejects AD instead of stopping gradients or duplicating a VJP.
    """
    _eager(matrix)
    a = jnp.asarray(matrix, dtype=jnp.complex128)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or not a.shape[0]
            or not np.isfinite(a).all() or not np.allclose(a, a.conj().T, atol=1e-10, rtol=1e-10)):
        raise ValueError('TDA requires a nonempty finite Hermitian matrix.')
    if isinstance(nstates, bool) or not isinstance(nstates, Integral) or not 1 <= nstates <= len(a):
        raise ValueError('nstates must be between one and the transition dimension.')
    if not isfinite(tol) or tol <= 0 or not isinstance(maxiter, Integral) or maxiter < 1:
        raise ValueError('Use positive finite tolerance and iteration limit.')
    size = len(a)
    width = min(size, max(40, 8*nstates)) if max_subspace is None else min(size, max_subspace)
    if not isinstance(width, Integral) or width < min(size, nstates+2):
        raise ValueError('max_subspace must leave room for correction vectors.')
    guesses = min(width, max(2*nstates, nstates+2))
    initial = jax.random.normal(jax.random.PRNGKey(seed), (size, guesses), dtype=jnp.complex128)/jnp.sqrt(size)
    diagonal = jnp.diag(a).real
    initial = initial.at[:, :nstates].add(jax.nn.one_hot(jnp.argsort(diagonal)[:nstates], size).T)
    apply = lambda vectors: a@vectors
    values, vectors, ok = _davidson_lowest_symmetric(apply, nroots=nstates, size=size,
        diag=diagonal.astype(jnp.complex128), tol=tol, max_iter=maxiter,
        max_subspace=width, initial_vectors=initial)
    values = values.real
    residual = jnp.linalg.norm(apply(vectors)-vectors*values, axis=0)
    converged = ok & jnp.isfinite(values) & jnp.isfinite(residual) & (residual <= tol*1.01)
    return TDAResult(values, vectors, residual, converged)
