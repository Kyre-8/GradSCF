from __future__ import annotations

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from gradscf.integrals.basis import CartesianBasis


def _packed_pair_indices(nao: int) -> tuple[np.ndarray, np.ndarray]:
    return np.tril_indices(int(nao))


def eri_pair_matrix_to_df_factors(
    pair_matrix: Array,
    *,
    nao: int,
    tol: float = 1e-10,
    max_rank: int | None = None,
    dtype: Array | None = None,
) -> Array:
    """Spectrally factorize packed native ERIs; no auxiliary basis is fitted."""

    pair_np = np.asarray(pair_matrix, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(0.5 * (pair_np + pair_np.T))
    keep = np.where(eigvals > float(tol))[0]
    target_dtype = jnp.asarray(pair_matrix if dtype is None else dtype).dtype
    if keep.size == 0:
        return jnp.zeros((0, int(nao), int(nao)), dtype=target_dtype)
    if max_rank is not None and int(max_rank) > 0:
        keep = keep[-int(max_rank) :]
    vals = eigvals[keep]
    vecs = eigvecs[:, keep]
    packed_factors = (vecs * np.sqrt(vals)[None, :]).T
    rows, cols = _packed_pair_indices(int(nao))
    factors = np.zeros((packed_factors.shape[0], int(nao), int(nao)), dtype=packed_factors.dtype)
    factors[:, rows, cols] = packed_factors
    factors[:, cols, rows] = packed_factors
    return jnp.asarray(factors, dtype=target_dtype)


def eri_pair_matrix_to_df_factors_traceable(
    pair_matrix: Array,
    *,
    nao: int,
    tol: float = 1e-10,
    max_rank: int | None = None,
) -> Array:
    """Traceable AO-pair factors, without an auxiliary fitting basis.

    With no rank cap, return the symmetric principal square root, with small
    eigenvalues removed at ``tol``. Its first matrix-function derivative is
    invariant within repeated auxiliary eigenvalue clusters. The active space
    must stay separated from the cutoff; this first-order interface does not
    promise higher derivatives at degeneracy.

    An explicit positive ``max_rank`` retains the legacy rectangular spectral
    factors. Their individual eigenvectors require isolated eigenvalues for AD.
    Both gauges reconstruct the same retained pair matrix.
    """

    pair = jnp.asarray(pair_matrix)
    sym_pair = 0.5 * (pair + pair.T)
    if max_rank is not None and int(max_rank) > 0:
        eigvals, eigvecs = jnp.linalg.eigh(sym_pair)
        eigvals = eigvals[-int(max_rank) :]
        eigvecs = eigvecs[:, -int(max_rank) :]
        threshold = jnp.asarray(tol, dtype=eigvals.dtype)
        scales = jnp.where(eigvals > threshold, jnp.sqrt(jnp.maximum(eigvals, threshold)), 0.)
        packed_factors = (eigvecs * scales[None, :]).T
    else:
        from gradscf.solvers.eigen.spectral import matrix_function

        def root(values, threshold):
            retained = values > threshold
            return jnp.where(retained, jnp.sqrt(jnp.where(retained, values, 1.)), 0.)

        def divided_difference(values, threshold):
            retained = values > threshold
            scales = root(values, threshold)
            both = retained[:, None] & retained[None, :]
            cross = retained[:, None] != retained[None, :]
            # sqrt(x)-sqrt(y) over x-y = 1/(sqrt(x)+sqrt(y)).
            # Across the cutoff, use the exact secant of the truncated function.
            sums = scales[:, None]+scales[None, :]
            gaps = values[:, None]-values[None, :]
            mixed = (scales[:, None]-scales[None, :])/jnp.where(cross, gaps, 1.)
            return jnp.where(both, 1/jnp.where(both, sums, 1.), jnp.where(cross, mixed, 0.))

        packed_factors = matrix_function(sym_pair, jnp.asarray(tol, dtype=pair.dtype), root,
                                         divided_difference=divided_difference)
    rows_np, cols_np = _packed_pair_indices(int(nao))
    rows = jnp.asarray(rows_np)
    cols = jnp.asarray(cols_np)
    factors = jnp.zeros((packed_factors.shape[0], int(nao), int(nao)), dtype=pair.dtype)
    factors = factors.at[:, rows, cols].set(packed_factors)
    factors = factors.at[:, cols, rows].set(packed_factors)
    return factors


def eri_to_df_factors(
    eri: Array,
    *,
    tol: float = 1e-10,
    max_rank: int | None = None,
) -> Array:
    """Build a spectral factorization from a full AO ERI tensor.

    This compresses the full ERI and is not auxiliary-basis density fitting.

    The factorization is performed in the AO-pair space:
    (pq|rs) ~= sum_Q B_Q[p,q] B_Q[r,s]
    """

    eri_np = np.asarray(eri, dtype=float)
    nao = int(eri_np.shape[0])
    rows, cols = _packed_pair_indices(nao)
    pair = eri_np[rows, cols][:, rows, cols]
    return eri_pair_matrix_to_df_factors(
        pair,
        nao=nao,
        tol=tol,
        max_rank=max_rank,
        dtype=jnp.asarray(eri),
    )


def eri_to_df_factors_from_basis(
    basis: CartesianBasis,
    *,
    tol: float = 1e-10,
    max_rank: int | None = None,
    engine: str = "auto",
) -> Array:
    from gradscf.integrals.molecular.eri import eri_pair_matrix_packed
    pair = eri_pair_matrix_packed(basis, engine=engine)
    return eri_pair_matrix_to_df_factors(
        pair,
        nao=basis.nao,
        tol=tol,
        max_rank=max_rank,
        dtype=pair,
    )
