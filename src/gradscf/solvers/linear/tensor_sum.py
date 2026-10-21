"""Symmetric Kronecker-sum inverse with an implicit tensor response.

The spectral basis is only a numerical factorization. AD differentiates
A X = B, so repeated eigenvalues of an individual factor do not introduce
eigenvector denominators. The complete tensor-sum spectrum must be nonsingular.
"""
from math import isfinite

import jax
import jax.numpy as jnp

from ..diagnostics import linear_residual
from ..types import TensorSumResult


def _apply_axis(matrix, tensor, axis):
    return jnp.moveaxis(jnp.tensordot(matrix, tensor, axes=(1, axis)), 0, axis)


def solve_tensor_sum(factors, rhs, *, denominator_tol=1e-10, rtol=1e-10, atol=1e-12):
    """Solve sum_k F[k] acting on axis k of X = rhs, for real symmetric F.

    factors[k] has shape (rhs.shape[k], rhs.shape[k]); no full Kronecker matrix
    is formed. Storage is proportional to the RHS tensor plus factor matrices.
    Nonfinite/nonsymmetric factors, unresolved sums, or failed true-residual
    checks yield NaN solutions. The same checks apply to tangent/adjoint solves.
    """
    if any(not isfinite(x) or x <= 0 for x in (denominator_tol, rtol, atol)):
        raise ValueError("Tensor-sum tolerances must be finite and positive")
    rhs = jnp.asarray(rhs)
    factors = tuple(jnp.asarray(f) for f in factors)
    if not factors or len(factors) != rhs.ndim or any(
            f.shape != (rhs.shape[k], rhs.shape[k]) for k, f in enumerate(factors)):
        raise ValueError("Each factor shape must match its RHS axis")
    if not jnp.issubdtype(rhs.dtype, jnp.floating) or any(
            not jnp.issubdtype(f.dtype, jnp.floating) for f in factors):
        raise ValueError("Tensor-sum inputs require real floating-point dtype")
    dtype = jnp.result_type(rhs, *factors)
    rhs = rhs.astype(dtype)
    factors = tuple(f.astype(dtype) for f in factors)
    valid = jnp.asarray(True)
    for f in factors:
        tolerance = 32*jnp.finfo(dtype).eps*jnp.maximum(1., jnp.linalg.norm(f))
        valid &= jnp.all(jnp.isfinite(f)) & (jnp.linalg.norm(f-f.T) <= tolerance)
    if rhs.size == 0:
        return TensorSumResult(rhs, jnp.asarray(0., dtype), valid, jnp.asarray(jnp.inf, dtype))
    spectrum = jnp.zeros(rhs.shape, dtype)
    vectors = []
    for axis, f in enumerate(factors):
        # Diagonalization belongs to the common solver and is never differentiated.
        values, basis = jnp.linalg.eigh(jax.lax.stop_gradient(f))
        shape = [1]*rhs.ndim
        shape[axis] = f.shape[0]
        spectrum = spectrum + values.reshape(shape)
        vectors.append(basis)
    minimum = jnp.min(jnp.abs(spectrum))
    valid &= (minimum > denominator_tol) & jnp.isfinite(minimum)

    def operator(value):
        return sum(_apply_axis(f, value, axis) for axis, f in enumerate(factors))

    def solve(matvec, value):
        transformed = value
        for axis, basis in enumerate(vectors):
            transformed = _apply_axis(basis.T, transformed, axis)
        transformed /= jnp.where(jnp.abs(spectrum) > denominator_tol, spectrum, 1.)
        for axis, basis in enumerate(vectors):
            transformed = _apply_axis(basis, transformed, axis)
        norm, converged = linear_residual(matvec, transformed, value,
                                          rtol=rtol, atol=atol, converged=valid)
        return jnp.where(converged, transformed, jnp.nan)

    solution = jax.lax.custom_linear_solve(
        operator, rhs, solve=solve, symmetric=True)
    # JAX 0.10 can batch opaque-solve auxiliary diagnostics with tangent RHSs,
    # contaminating an outer jacfwd's unbatched primal. Check each numerical
    # solve above, then report the physical residual without auxiliary outputs.
    norm, converged = linear_residual(operator, solution, rhs,
                                      rtol=rtol, atol=atol, converged=valid)
    return TensorSumResult(solution, norm, converged, minimum)
