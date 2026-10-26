"""Complex systems through the shared checked real linear solver."""

import jax.numpy as jnp

from ..operators import LinearOperator, as_operator
from ..types import LinearResult
from .implicit import solve_linear


def solve_complex(matrix_or_operator, rhs, *, config=None):
    """Solve A x=b by the real representation [Re(A),-Im(A);Im(A),Re(A)].

    Accept a square real/complex operator and vector or block RHS. The
    representation stays matrix-free for GMRES, and reuses the real solver's
    primal/transpose residual checks and implicit AD. max_dense bounds 2*n.
    Failed solves retain their status and NaN solution, without a fallback.
    """
    op, rhs = as_operator(matrix_or_operator), jnp.asarray(rhs)
    n = op.shape[0]
    if op.shape != (n, n):
        raise ValueError("Solver requires a square operator")
    if rhs.ndim not in (1, 2) or rhs.shape[0] != n:
        raise ValueError("rhs shape must be (n,) or (n,nrhs)")
    if not all(jnp.issubdtype(dtype, jnp.inexact) for dtype in (op.dtype, rhs.dtype)):
        raise ValueError("Complex solve requires floating-point or complex data")
    dtype = jnp.result_type(op.dtype, rhs.dtype, 1j)
    rhs = rhs.astype(dtype)

    def apply(x):
        value = op.apply(x[:n] + 1j * x[n:])
        return jnp.concatenate((value.real, value.imag), axis=0)

    real = LinearOperator((2 * n, 2 * n), rhs.real.dtype, apply, matmat=apply)
    result = solve_linear(
        real, jnp.concatenate((rhs.real, rhs.imag), axis=0), config=config
    )
    return LinearResult(
        result.solution[:n] + 1j * result.solution[n:],
        result.residual_norm,
        result.converged,
        result.status,
    )
