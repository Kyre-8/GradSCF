"""Checked implicit linear solves, including their transpose and higher response."""
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from ..diagnostics import linear_residual
from ..operators import as_operator, validate_real_square
from ..types import LinearSolverConfig, LinearResult
from .direct import direct_solve
from .gmres import gmres_solve


def checked_linear_solve(matvec, rhs, *, config, converged=True,
                         preconditioner=None, transpose_preconditioner=None,
                         cholesky=None, diagonal_factor=None):
    """Internal vector/block kernel; validity is checked inside the opaque solve.

    Placing a nonlinear residual predicate in a linear tangent map would break
    transposition. Both primal and transposed numerical solves are checked here.
    """
    if diagonal_factor is not None:
        if cholesky is not None:
            raise ValueError("Supply either an exact diagonal or Cholesky factor")
        factor = jnp.asarray(diagonal_factor)
        if factor.shape != (rhs.shape[0],):
            raise ValueError("Exact diagonal factor must match the operator shape")
        if not jnp.issubdtype(factor.dtype, jnp.floating):
            raise NotImplementedError("Exact diagonal solves require real floating-point data")
        factor = jax.lax.stop_gradient(factor.astype(rhs.dtype))
        finite = jnp.isfinite(factor) & (factor != 0)
        safe = jnp.where(finite, factor, 1.).reshape((-1,)+(1,)*(rhs.ndim-1))
        action = matvec if rhs.ndim == 1 else lambda x: jax.vmap(
            matvec, in_axes=1, out_axes=1)(x)

        def solve(operator, value):
            candidate = value/safe
            norm, valid = linear_residual(operator, candidate, value,
                rtol=config.rtol, atol=config.atol,
                converged=jnp.asarray(converged) & jnp.all(finite))
            return jnp.where(valid, candidate, jnp.nan), (norm, valid)

        # The stopped factor is numerical only. AD differentiates the original
        # action, including RHS response at zero and all off-diagonal tangents.
        return jax.lax.custom_linear_solve(action, rhs, solve=solve,
            transpose_solve=solve, has_aux=True)

    if rhs.size == 0:
        return rhs, (jnp.asarray(0., dtype=rhs.real.dtype), jnp.asarray(converged))

    if cholesky is not None:
        if config.method != "direct" or rhs.shape[0] > config.max_dense:
            raise ValueError("A Cholesky factor requires a bounded direct solve")
        factor = jax.lax.stop_gradient(jnp.asarray(cholesky))
        if factor.shape != (rhs.shape[0], rhs.shape[0]):
            raise ValueError("Cholesky factor must match the operator shape")
        action = matvec if rhs.ndim == 1 else lambda x: jax.vmap(
            matvec, in_axes=1, out_axes=1)(x)

        def solve(operator, value):
            candidate = solve_triangular(factor.T,
                solve_triangular(factor, value, lower=True), lower=False)
            norm, valid = linear_residual(operator, candidate, value,
                rtol=config.rtol, atol=config.atol, converged=converged)
            return jnp.where(valid, candidate, jnp.nan), (norm, valid)

        return jax.lax.custom_linear_solve(action, rhs, solve=solve,
            transpose_solve=solve, has_aux=True)

    if rhs.ndim == 2 and config.method == "direct":
        if rhs.shape[0] > config.max_dense:
            raise ValueError("Direct solve exceeds max_dense; use GMRES")
        matrix = jax.vmap(matvec, in_axes=1, out_axes=1)(jnp.eye(rhs.shape[0], dtype=rhs.dtype))
        def block_solve(transpose):
            a = matrix.T if transpose else matrix
            def solve(operator, value):
                candidate = jnp.linalg.solve(a, value)
                norm, valid = linear_residual(operator, candidate, value, rtol=config.rtol,
                                               atol=config.atol, converged=converged)
                return jnp.where(valid, candidate, jnp.nan), (norm, valid)
            return solve
        return jax.lax.custom_linear_solve(lambda x: matrix @ x, rhs,
            solve=block_solve(False), transpose_solve=block_solve(True), has_aux=True)

    def factory(precond):
        def solve(operator, value):
            if config.method == "direct":
                result = direct_solve(operator, value, max_dense=config.max_dense)
            else:
                result = gmres_solve(operator, value, rtol=config.rtol, atol=config.atol,
                                     maxiter=config.maxiter, restart=config.restart,
                                     preconditioner=precond)
            norm, valid = linear_residual(operator, result, value, rtol=config.rtol,
                                           atol=config.atol, converged=converged)
            return jnp.where(valid, result, jnp.full_like(result, jnp.nan)), (norm, valid)
        return solve

    if rhs.ndim == 2:
        # Keep the column loop INSIDE the opaque linear solve. Mapping public
        # solves outside it breaks linear_transpose of the block action.
        transpose = jax.linear_transpose(matvec, jnp.zeros(rhs.shape[0], rhs.dtype))

        def block_factory(vector_operator, precond):
            def solve(operator, value):
                solution, (norms, valid) = jax.lax.map(
                    lambda column: factory(precond)(vector_operator, column), value.T
                )
                return solution.T, (jnp.linalg.norm(norms), jnp.all(valid))
            return solve

        return jax.lax.custom_linear_solve(
            lambda x: jax.vmap(matvec, in_axes=1, out_axes=1)(x), rhs,
            solve=block_factory(matvec, preconditioner),
            transpose_solve=block_factory(lambda x: transpose(x)[0], transpose_preconditioner),
            has_aux=True,
        )

    return jax.lax.custom_linear_solve(matvec, rhs, solve=factory(preconditioner),
        transpose_solve=factory(transpose_preconditioner), has_aux=True)


def solve_linear(matrix_or_operator, rhs, *, config=None, preconditioner=None,
                 transpose_preconditioner=None, cholesky=None):
    """Solve a real square system for a vector or (n,nrhs) block.

    maxiter counts GMRES restart cycles. True residuals certify convergence;
    the upstream GMRES info flag is never treated as a convergence certificate.
    An optional lower Cholesky factor reuses a real SPD matrix factorization;
    implicit response and both primal/transpose residual checks are retained.
    Explicit exact-diagonal operator structure bypasses dense/Krylov work, with
    the same true-residual checks and implicit response through the live action.
    """
    config = LinearSolverConfig() if config is None else config
    op = as_operator(matrix_or_operator)
    validate_real_square(op)
    rhs = jnp.asarray(rhs)
    if not (rhs.shape == (op.shape[0],) or
            (rhs.ndim == 2 and rhs.shape[0] == op.shape[0])):
        raise ValueError("rhs shape must be (n,) or (n,nrhs)")
    if not jnp.issubdtype(rhs.dtype, jnp.floating):
        raise ValueError("rhs must have real floating-point dtype")
    rhs = rhs.astype(jnp.result_type(rhs.dtype, op.dtype))
    solution, (norm, valid) = checked_linear_solve(op.apply, rhs, config=config,
        preconditioner=preconditioner, transpose_preconditioner=transpose_preconditioner,
        cholesky=cholesky, diagonal_factor=op.diagonal if op.exact_diagonal else None)
    return LinearResult(solution, norm, valid, jnp.where(valid, 0, 1))


def solve_implicit_linear_system(matvec, b_flat, *, tol, max_iter, restart=None, converged=True):
    """Compatibility contract used by implicit SCF and scGW responses."""
    config = LinearSolverConfig(rtol=float(tol), maxiter=max(1, int(max_iter)),
                                 restart=20 if restart is None else max(1, int(restart)))
    return checked_linear_solve(matvec, b_flat, config=config, converged=converged)[0]
