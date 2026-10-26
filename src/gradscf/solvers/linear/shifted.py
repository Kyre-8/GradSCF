"""Repeated checked shifted solves with one local primal factorization."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..diagnostics import linear_residual


class ShiftedFactorization(NamedTuple):
    matrix: object
    values: object
    vectors: object
    inverse_vectors: object = None
    spectral_valid: object = True


def factor_shifted(matrix, *, symmetric=True):
    """Prepare M for (M-shift I) solves; eigenvectors are primal data only.

    The solve differentiates the original matrix equation, including at
    repeated eigenvalues. No eigenvalue gap division or clipping is used.
    symmetric=True keeps the transpose-symmetric contract; False prepares a
    general real/complex matrix without symmetrizing its physical operator.
    """
    matrix = jnp.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Expected a square matrix')
    if not jnp.issubdtype(matrix.dtype, jnp.inexact):
        raise ValueError('Expected a real or complex floating-point matrix')
    tolerance = 64*jnp.finfo(matrix.dtype).eps*jnp.maximum(1., jnp.linalg.norm(matrix))
    valid = jnp.all(jnp.isfinite(matrix))
    if symmetric:
        valid &= jnp.linalg.norm(matrix-matrix.T) <= tolerance
    matrix = jnp.where(valid, (matrix+matrix.T)*.5 if symmetric else matrix, jnp.nan)
    primal = jax.lax.stop_gradient(matrix)
    if symmetric and not jnp.iscomplexobj(matrix):
        values, vectors = jnp.linalg.eigh(primal)
        return ShiftedFactorization(matrix, values, vectors)
    values, vectors = jnp.linalg.eig(primal)
    inverse = jnp.linalg.inv(vectors)
    # General and complex symmetric matrices need not be diagonalizable.
    # Cache an eigenbasis only when its conditioning and reconstruction permit
    # accurate solves; defective/ill-conditioned bases use the dense equation.
    eps = jnp.finfo(matrix.dtype).eps
    condition = jnp.linalg.norm(vectors) * jnp.linalg.norm(inverse)
    error = jnp.linalg.norm((vectors * values) @ inverse - primal)
    spectral_valid = (jnp.isfinite(condition) & (condition < eps**(-.25))
                      & (error <= 128*eps*jnp.maximum(1., jnp.linalg.norm(primal))))
    return ShiftedFactorization(matrix, values, vectors, inverse, spectral_valid)


@jax.jit
def solve_shifted(state, rhs, shift, *, rtol=1e-11, atol=1e-13):
    """Solve a real/complex scalar shift for a vector or matrix RHS.

    Factors stay local to their matrix. custom_linear_solve supplies JVP/VJP
    from A dx = db - dA x, without differentiating the spectral basis.
    The explicit transpose solve is non-conjugating, as JAX requires even
    for complex operators. Both primal and transpose numerical solutions are
    checked against the original matrix equation inside the opaque solve.
    A failed residual gets one cached iterative refinement before rejection;
    neither the operator nor the requested residual tolerance is changed.
    """
    rhs, shift = jnp.asarray(rhs), jnp.asarray(shift)
    if rhs.ndim not in (1, 2) or rhs.shape[0] != state.matrix.shape[0] or shift.ndim:
        raise ValueError('Expected RHS (n,) or (n,k) and a scalar shift')
    dtype = jnp.result_type(state.matrix, state.values, state.vectors, rhs, shift)
    rhs = rhs.astype(dtype)
    def action(x):
        return state.matrix @ x - shift*x
    def numerical_solve(transpose):
        def solve(operator, b):
            inverse = state.vectors.T if state.inverse_vectors is None else state.inverse_vectors
            left, right = ((inverse.T, state.vectors.T) if transpose else (state.vectors, inverse))
            def spectral(value):
                denominator = state.values-shift
                if value.ndim == 2:
                    denominator = denominator[:, None]
                return left@((right@value)/denominator)
            matrix = state.matrix-shift*jnp.eye(state.matrix.shape[0], dtype=dtype)
            if transpose:
                matrix = matrix.T
            def raw(value):
                return (spectral(value) if state.inverse_vectors is None else
                    jax.lax.cond(state.spectral_valid, spectral,
                                 lambda rhs: jnp.linalg.solve(matrix, rhs), value))
            candidate = raw(b)
            _, valid = linear_residual(operator, candidate, b, rtol=rtol, atol=atol)
            def refine(value):
                improved = value+raw(b-operator(value))
                _, accepted = linear_residual(operator, improved, b, rtol=rtol, atol=atol)
                return improved, accepted
            candidate, valid = jax.lax.cond(valid,
                lambda value: (value, jnp.array(True)), refine, candidate)
            return jnp.where(valid, candidate, jnp.nan)
        return solve
    return jax.lax.custom_linear_solve(action, rhs, solve=numerical_solve(False),
                                      transpose_solve=numerical_solve(True))
