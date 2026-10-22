"""Dense or iterative real non-Hermitian pairs with first-order energy response."""

import jax
import jax.numpy as jnp
from ..types import LinearSolverConfig, NonHermitianSolverConfig, NonHermitianResult
from ..operators import as_operator, validate_real_square
from ..diagnostics import require_converged_derivative
from ._nonhermitian_ritz import paired_eigenvectors
from .nonhermitian_davidson import davidson_ritz
from ._nonhermitian_response import attach_pair_response
from ..memory import apply_columns, plan_nonhermitian
from gradscf._memory import array_storage_bytes, tree_storage_bytes, validate_reserved_bytes


def solve_nonhermitian(matrix_or_operator, *, config=None, initial_vectors=None,
                      vector_response=False, response_config=None,
                      max_memory=None, reserved_bytes=0, _batch_size=None,
                      _response_budget=None):
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
    Optional max_memory budgets known primal and pair-response arrays in decimal MB;
    reserved_bytes declares caller data not visible in a generic operator.
    The two phases share the original budget, without retaining the primal
    Krylov workspace in the response phase. Opaque operator/library scratch
    remains caller-owned. A private EOM relay preserves its existing primal plan.
    """
    cfg = NonHermitianSolverConfig() if config is None else config
    if not isinstance(vector_response, bool):
        raise TypeError("vector_response must be a static bool")
    if response_config is not None and not isinstance(response_config, LinearSolverConfig):
        raise TypeError("response_config must be a LinearSolverConfig")
    if response_config is not None and not vector_response:
        raise ValueError("response_config requires vector_response=True")
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    a = None if hasattr(matrix_or_operator, "matvec") else jnp.asarray(matrix_or_operator)
    op = as_operator(matrix_or_operator if a is None else a)
    validate_real_square(op)
    n = op.shape[0]
    k = cfg.nroots
    if k > n:
        raise ValueError("nroots exceeds operator dimension")
    response_budget = None if _response_budget is None else dict(_response_budget)
    source_arrays = ()
    if max_memory is not None:
        # Preserve caller arrays before dtype promotion and before dense `a`
        # is rebound to the stopped numerical assembly below.
        source_arrays = (a, op.diagonal, initial_vectors)
        if initial_vectors is not None:
            if cfg.method == "dense":
                raise ValueError("initial_vectors applies only to Davidson")
            if jnp.iscomplexobj(initial_vectors):
                raise NotImplementedError("Initial vectors must be real")
            initial_vectors = jnp.asarray(initial_vectors, dtype=op.dtype)
            if (initial_vectors.ndim != 2 or initial_vectors.shape[0] != n
                    or not 1 <= initial_vectors.shape[1] <= min(n, cfg.max_space)):
                raise ValueError("initial_vectors must have shape (n, 1..max_space)")
        source_bytes = reserved_bytes+array_storage_bytes(*source_arrays, initial_vectors)
        cfg, _, _batch_size = plan_nonhermitian(n, op.dtype, cfg, max_memory,
            resident_bytes=source_bytes,
            initial_columns=0 if initial_vectors is None else initial_vectors.shape[1],
            device=getattr(op.diagonal, "device", None))
        response_budget = dict(max_memory=max_memory, reserved_bytes=source_bytes,
            operator_min_bytes=0, operator_bytes_per_column=0, action_columns=_batch_size)
    apply = lambda v: apply_columns(op, v, _batch_size)
    apply_transpose = lambda v: apply_columns(op.T, v, _batch_size)
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
            blocks.append(apply(probes))
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
        solved = (davidson_ritz(op, cfg, initial_vectors) if _batch_size is None else
                  davidson_ritz(op, cfg, initial_vectors, batch_size=_batch_size))
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
        jax.lax.stop_gradient(apply(r)) - r * energies, axis=0
    )
    left_norm = jnp.linalg.norm(l, axis=0)
    left_res = jnp.linalg.norm(
        jax.lax.stop_gradient(apply_transpose(l)) - l * energies, axis=0
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
        if response_budget is not None:
            # Only retained arrays survive the primal phase, not its Arnoldi
            # capacities, eigensolver workspaces or complete trajectory.
            response_budget['reserved_bytes'] += tree_storage_bytes((
                a, wr, right, left, l0, gram, singular,
                right_res, left_norm, left_res, bio, real, conv, condition,
                present, distances, margin, valid, guards), exclude=source_arrays)
        values, r, l = attach_pair_response(op, energies, r, l,
            config=linear, valid=valid, **({} if response_budget is None else response_budget))
    else:
        applied = apply(r)
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
