"""Dense or iterative real non-Hermitian pairs with first-order energy response."""

import jax
import jax.numpy as jnp
from ..types import LinearSolverConfig, NonHermitianSolverConfig, NonHermitianResult
from ..operators import as_operator, validate_real_square
from ..diagnostics import require_converged_derivative
from ._nonhermitian_ritz import paired_eigenvectors
from .nonhermitian_davidson import davidson_ritz
from ._nonhermitian_response import attach_pair_response


def solve_nonhermitian(matrix_or_operator, *, config=None, initial_vectors=None,
                      vector_response=False, response_config=None):
    """Lowest-real-part dense eigenpairs or iterative Ritz estimates.

    By default only isolated real eigenvalue JVP/VJP is exposed and left/right
    vectors are stopped numerical outputs. Opt-in ``vector_response=True``
    attaches normalized local left/right response using shared root and linear
    solves. ``response_config`` controls those solves; direct capacity is n+1
    per right root and n per left root, not the number of requested roots.
    Complex selected roots give NaN real outputs;
    raw_eigenvalues holds the dense spectrum or the final projected spectrum.
    Davidson uses bounded storage and true right/left and guard residuals; its
    gaps/order are local Ritz estimates, not a global isolation certificate.
    spectrum_complete distinguishes full coverage. Exceptional or poorly
    conditioned roots invalidate energy response; no dense fallback is used.
    """
    cfg = NonHermitianSolverConfig() if config is None else config
    if not isinstance(vector_response, bool):
        raise TypeError("vector_response must be a static bool")
    if response_config is not None and not isinstance(response_config, LinearSolverConfig):
        raise TypeError("response_config must be a LinearSolverConfig")
    if response_config is not None and not vector_response:
        raise ValueError("response_config requires vector_response=True")
    op = as_operator(matrix_or_operator)
    validate_real_square(op)
    n = op.shape[0]
    k = cfg.nroots
    if k > n:
        raise ValueError("nroots exceeds operator dimension")
    if cfg.method == "dense":
        if initial_vectors is not None:
            raise ValueError("initial_vectors applies only to Davidson")
        if n > cfg.max_dense:
            raise ValueError("Non-Hermitian dense reference exceeds max_dense")
        blocks = []
        for start in range(0, n, cfg.block_size):
            probes = jax.nn.one_hot(
                jnp.arange(start, min(start + cfg.block_size, n)), n, dtype=op.dtype
            ).T
            blocks.append(op.apply(probes))
        a = jax.lax.stop_gradient(jnp.concatenate(blocks, axis=1))
        wr, right, left = paired_eigenvectors(a, k)
        complete, iterations, dimension = (
            jnp.asarray(True),
            jnp.asarray(1),
            jnp.asarray(n),
        )
        guards = jnp.zeros((0,), dtype=op.dtype)
        numerical_valid = jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(wr))
    else:
        solved = davidson_ritz(op, cfg, initial_vectors)
        wr, right, left = solved.spectrum, solved.right, solved.left
        iterations, dimension = solved.iterations, solved.dimension
        complete = dimension == n
        guards = solved.guard_residuals
        numerical_valid = (
            solved.guard_valid
            & jnp.all(jnp.isfinite(right))
            & jnp.all(jnp.isfinite(left))
        )
    restarts = jnp.asarray(0) if cfg.method == "dense" else solved.restarts
    wr, right, left = jax.lax.stop_gradient((wr, right, left))
    r = right.real
    r /= jnp.maximum(jnp.linalg.norm(r, axis=0), jnp.finfo(r.dtype).tiny)
    l0 = left.real
    gram = l0.T @ r
    singular = jnp.linalg.svd(gram, compute_uv=False)
    pairing = jnp.all(jnp.isfinite(singular)) & (
        singular[-1] > 64 * jnp.finfo(r.dtype).eps * jnp.maximum(1.0, singular[0])
    )
    l = l0 @ jnp.linalg.solve(
        jnp.where(pairing, gram.T, jnp.eye(k, dtype=r.dtype)), jnp.eye(k, dtype=r.dtype)
    )
    energies = wr[:k].real
    right_res = jnp.linalg.norm(
        jax.lax.stop_gradient(op.apply(r)) - r * energies, axis=0
    )
    left_norm = jnp.linalg.norm(l, axis=0)
    left_res = jnp.linalg.norm(
        jax.lax.stop_gradient(op.T.apply(l)) - l * energies, axis=0
    ) / jnp.maximum(left_norm, 1.0)
    bio = jnp.linalg.norm(l.T @ r - jnp.eye(k, dtype=r.dtype))
    real = jnp.abs(wr[:k].imag) <= cfg.imaginary_tol
    conv = (
        jnp.all(jnp.isfinite(r))
        & pairing
        & numerical_valid
        & real
        & (right_res <= cfg.atol)
        & (left_res <= cfg.atol)
        & (bio <= 1e-8)
    )
    condition = jnp.where(pairing, left_norm, jnp.inf)
    present = jnp.arange(wr.size) < dimension
    distances = jnp.where(
        present[None, :], jnp.abs(wr[:k, None] - wr[None, :]), jnp.inf
    )
    distances = distances.at[jnp.arange(k), jnp.arange(k)].set(jnp.inf)
    margin = cfg.gap_atol + cfg.gap_rtol * jnp.maximum(
        jnp.abs(wr[:k, None]), jnp.where(present[None, :], jnp.abs(wr[None, :]), 0.0)
    )
    margin += 2 * condition[:, None] * jnp.maximum(right_res, left_res)[:, None]
    valid = (
        jnp.all(conv)
        & jnp.all(condition < cfg.max_condition)
        & jnp.all(distances > margin)
        & numerical_valid
    )
    r, l = jax.lax.stop_gradient(r), jax.lax.stop_gradient(l)
    if vector_response:
        linear = (LinearSolverConfig(rtol=1e-10, restart=40)
                  if response_config is None else response_config)
        values, r, l = attach_pair_response(op, energies, r, l,
                                            config=linear, valid=valid)
    else:
        applied = op.apply(r)
        live_zero = applied - jax.lax.stop_gradient(applied)
        values = energies + jnp.sum(l * live_zero, axis=0)
    values = require_converged_derivative(values, valid)
    values = require_converged_derivative(jnp.where(real, values, jnp.nan), valid)
    r = jnp.where(real[None, :], r, jnp.nan)
    l = jnp.where(real[None, :], l, jnp.nan)
    if vector_response:
        # A complex-root output mask is constant NaN in the primal. Guard after
        # masking as well, so its tangent cannot be mistaken for a valid zero.
        r = require_converged_derivative(r, valid)
        l = require_converged_derivative(l, valid)
    return NonHermitianResult(
        values,
        r,
        l,
        right_res,
        left_res,
        conv,
        jnp.broadcast_to(valid, (k,)),
        condition,
        bio,
        wr,
        complete,
        iterations,
        dimension,
        jnp.min(distances, axis=1),
        guards,
        restarts,
    )
