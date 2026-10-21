"""One real Hermitian solve with explicitly selected differentiable observables."""

import jax
import jax.numpy as jnp
from ..operators import as_operator, validate_real_square
from ..diagnostics import require_converged_derivative
from ..types import (
    EigenSolverConfig,
    EigenResponseConfig,
    LinearSolverConfig,
    EigenResult,
)
from .primal import solve_ritz
from .response import attach_spectral_response, spectral_values
from ..memory import apply_columns, plan_hermitian
from gradscf._memory import array_storage_bytes, validate_reserved_bytes


def solve_hermitian(
    matrix_or_operator, *, config=None, response=None, probes=None, initial_vectors=None,
    max_memory=None, reserved_bytes=0, _batch_size=None,
):
    """Solve once; differentiate isolated eigenpairs or an isolated subspace.

    Subspace mode permits internal degeneracy and exposes only invariant AD
    outputs. Static targets keep JIT shapes fixed. Raw Ritz diagnostics include
    one available guard root; no automatic dimension expansion at a crossing.
    Legacy config AD controls remain honored when response is omitted (also
    used by RPA); combining nondefault legacy controls with response is an error.
    Optional max_memory is a decimal-MB budget for known primal working arrays;
    reserved_bytes declares additional resident caller data. Operator internals,
    implicit response and actual traced-device capacity need separate planning.
    """
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    cfg = EigenSolverConfig() if config is None else config
    if response is None:
        response = EigenResponseConfig(
            target=(
                "eigenpairs"
                if cfg.gradient_mode == "implicit_eigenvector"
                else "eigenvalues"
            ),
            linear_config=LinearSolverConfig(
                rtol=cfg.adjoint_tol, maxiter=cfg.adjoint_maxiter, restart=40
            ),
        )
    elif (
        cfg.gradient_mode != "eigenvalue_only"
        or cfg.adjoint_tol != 1e-10
        or cfg.adjoint_maxiter != 100
    ):
        raise ValueError("Use response controls or legacy config AD controls, not both")
    if not isinstance(response, EigenResponseConfig):
        raise TypeError("response must be an EigenResponseConfig")
    a = None if hasattr(matrix_or_operator, "matvec") else jnp.asarray(matrix_or_operator)
    op = as_operator(matrix_or_operator if a is None else a)
    validate_real_square(op)
    n, k = op.shape[0], cfg.nroots
    if probes is not None:
        if response.target != "subspace":
            raise ValueError("probes require a subspace response")
        probes = jnp.asarray(probes)
        if probes.ndim not in (1, 2) or probes.shape[0] != n:
            raise ValueError("Projector probes must have shape (n,) or (n,nvec)")
        if not jnp.issubdtype(probes.dtype, jnp.floating):
            raise NotImplementedError(
                "Projector probes must be real floating-point data"
            )
        probes = probes.astype(jnp.result_type(probes.dtype, op.dtype))
    if max_memory is not None:
        if initial_vectors is not None:
            initial_vectors = jnp.asarray(initial_vectors)
            if (initial_vectors.ndim != 2 or initial_vectors.shape[0] != n
                    or initial_vectors.shape[1] < min(k+1, n)):
                raise ValueError("initial_vectors must have shape (size, m), "
                                 "including requested roots and the boundary root")
        cfg, _, _batch_size = plan_hermitian(n, op.dtype, cfg, max_memory,
            resident_bytes=reserved_bytes + array_storage_bytes(
                a, op.diagonal, probes, initial_vectors),
            initial_columns=0 if initial_vectors is None else initial_vectors.shape[1],
            device=getattr(op.diagonal, "device", None))
    structure = jnp.asarray(True)
    if a is not None:
        symmetry_tol = (
            32 * jnp.finfo(a.dtype).eps * jnp.maximum(1.0, jnp.linalg.norm(a))
        )
        structure = jnp.all(jnp.isfinite(a)) & (
            jnp.linalg.norm(a - a.T) <= symmetry_tol
        )
    roots = solve_ritz(
        op, cfg, initial_vectors=initial_vectors, structure_valid=structure,
        batch_size=_batch_size,
    )
    x = roots.vectors[:, :k]
    if k < n:
        gap = roots.values[k] - roots.values[k - 1]
        threshold = (
            response.gap_atol
            + response.gap_rtol
            * jnp.maximum(jnp.abs(roots.values[k]), jnp.abs(roots.values[k - 1]))
            + roots.residual_norms[k]
            + roots.residual_norms[k - 1]
        )
        known_end = (~roots.present[k]) & roots.complete & jnp.all(roots.present[:k])
        gap = jnp.where(known_end, jnp.inf, gap)
        separated = known_end | (roots.present[k] & (gap > threshold))
    else:
        gap = jnp.asarray(jnp.inf, x.dtype)
        separated = jnp.asarray(True)
    guard_valid = jnp.all(roots.converged[k:] | (~roots.present[k:] & roots.complete))
    all_converged = jnp.all(roots.converged[:k]) & guard_valid
    valid = all_converged & separated
    if cfg.value_min is not None:
        lower_margin = (
            response.gap_atol
            + response.gap_rtol * (abs(cfg.value_min) + roots.lower_gap)
            + roots.lower_residual
        )
        valid &= (roots.lower_gap > lower_margin) & (roots.lower_residual <= cfg.atol)
    if response.target != "subspace":
        distance = jnp.abs(roots.values[:k, None] - roots.values[None, :])
        scale = jnp.maximum(
            jnp.abs(roots.values[:k, None]), jnp.abs(roots.values[None, :])
        )
        threshold = (
            response.gap_atol
            + response.gap_rtol * scale
            + roots.residual_norms[:k, None]
            + roots.residual_norms[None, :]
        )
        distance = distance.at[jnp.arange(k), jnp.arange(k)].set(jnp.inf)
        valid &= jnp.all((distance > threshold) | ~roots.present[None, :])

    # Validity is an explicit response input and is checked on every returned
    # observable. Capturing its nondifferentiable tracer inside this operator
    # bypasses closure_convert's differentiable-parameter hoisting and leaks
    # it across a nested JIT/higher derivative.
    apply = lambda v: apply_columns(op, v, _batch_size)

    projection = trace = None
    if response.target == "subspace":
        if probes is not None:
            if k == n:
                projection = probes
            else:
                y = attach_spectral_response(apply, x, config=response.linear_config, valid=valid)
                projection = y @ (y.T @ probes)
            projection = require_converged_derivative(projection, valid)
        trace = (jnp.sum(x*apply(x)) if k == n else
                 jnp.sum(spectral_values(apply, x, config=response.linear_config, valid=valid)))
        trace = require_converged_derivative(trace, valid)
        values = vectors = None
        converged = all_converged
        response_valid = valid
    else:
        # One matrix root, with separate rank-one charts for isolated columns.
        # An outer vmap of custom_root fails under nested JIT/third derivatives
        # on JAX 0.10; a scan also drops higher terms. Columnwise checked linear
        # solves stay inside the single implicit root instead.
        values = spectral_values(
            apply, x, config=response.linear_config, valid=valid, isolated=True)
        values = require_converged_derivative(values, valid)
        if response.target == "eigenpairs":
            vectors = attach_spectral_response(
                apply, x, config=response.linear_config, valid=valid, isolated=True)
            vectors = require_converged_derivative(vectors, valid)
        else:
            vectors = x
        converged = roots.converged[:k]
        response_valid = jnp.broadcast_to(valid, (k,))
    status = jnp.where(~converged, 1, jnp.where(response_valid, 0, 2))
    return EigenResult(
        values,
        vectors,
        roots.residual_norms[:k],
        converged,
        status,
        response_valid,
        gap,
        roots.values,
        roots.vectors,
        roots.residual_norms,
        roots.present,
        roots.lower_gap,
        roots.lower_residual,
        projection,
        trace,
    )
