"""Real matrix-free RPA with metric-constrained first-order vector response."""

from dataclasses import replace
from functools import partial
from math import isfinite
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..operators import as_operator, validate_real_square, LinearOperator
from ..types import EigenSolverConfig, LinearSolverConfig, RPAResult
from ..diagnostics import require_converged_derivative
from ..linear import solve_linear
from .api import solve_hermitian
from .rpa import _davidson_lowest_tdhf
from .response import _spectral_inputs
from .stable_rpa import solve_stable_rpa


def _initial(diagonal, width, count, seed):
    k = min(width, max(count + 1, 2 * count))
    canonical = jax.nn.one_hot(
        jnp.argsort(diagonal)[:count], diagonal.size, dtype=diagonal.dtype
    ).T
    random = jnp.asarray(
        np.random.default_rng(seed).normal(size=(diagonal.size, k - count)),
        dtype=diagonal.dtype,
    )
    return jnp.concatenate([canonical, random], axis=1)


def _metric(values):
    n = values.shape[0] // 2
    return jnp.concatenate([values[:n], -values[n:]], axis=0)


def _response_vectors(apply_h, values, vectors, diagonal, config, response_config):
    """Existing first-order metric-gauge expression, evaluated only under AD."""
    z0 = jax.lax.stop_gradient(vectors)
    w0 = jax.lax.stop_gradient(values)

    def root(k):
        z = z0[:, k]
        jz = _metric(z)

        # For z.T J z=1, K=H-omega J+(Jz)(Jz).T fixes the metric gauge.
        def augmented(v):
            return apply_h(v) - w0[k] * _metric(v) + jz * jnp.vdot(jz, v)

        hz = apply_h(z)
        dhz = hz - jax.lax.stop_gradient(hz)
        rhs = -dhz + jz * jnp.vdot(z, dhz)
        shifted = jnp.concatenate([diagonal - w0[k], diagonal + w0[k]]) + jz * jz
        floor = jnp.asarray(1e-8, dtype=z.dtype)
        denominator = jnp.where(
            jnp.abs(shifted) > floor, shifted, jnp.where(shifted < 0, -floor, floor)
        )
        precondition = lambda v: v / denominator
        op = LinearOperator((z.size, z.size), z.dtype, augmented)
        dz = solve_linear(
            op,
            rhs,
            config=response_config or LinearSolverConfig(
                rtol=config.adjoint_tol,
                maxiter=config.adjoint_maxiter,
                restart=min(20, z.size),
            ),
            preconditioner=precondition,
            transpose_preconditioner=precondition,
        ).solution
        return z + dz

    return jax.lax.map(root, jnp.arange(values.size)).T


@partial(jax.custom_jvp, nondiff_argnums=(0, 5, 6))
def _lazy_response(operator, parameters, values, vectors, diagonal, config, response_config, valid):
    return vectors


@_lazy_response.defjvp
def _lazy_response_jvp(operator, config, response_config, primals, tangents):
    parameters, values, vectors, diagonal, valid = primals
    dparameters, _, _, _, _ = tangents
    attached = lambda physical: _response_vectors(
        lambda v: require_converged_derivative(operator(v, *physical), valid),
        values, vectors, diagonal, config, response_config)
    tangent = jax.jvp(attached, (parameters,), (dparameters,))[1]
    return _lazy_response(operator, parameters, values, vectors, diagonal,
                          config, response_config, valid), tangent


def _attach_response(apply_h, values, vectors, diagonal, config, response_config, valid):
    if (response_config is not None and response_config.method == "direct"
            and vectors.shape[0] > response_config.max_dense):
        raise ValueError("Direct response solve exceeds max_dense (requires 2*n)")
    # The metric-gauge solver acts on vectors, so convert that input signature.
    # Keep validity explicit: captured bool tracers do not lift through nested JIT.
    operator, parameters, _ = _spectral_inputs(apply_h, vectors[:, 0])
    return _lazy_response(operator, parameters, values, jax.lax.stop_gradient(vectors),
                          diagonal, config, response_config, valid)


def solve_rpa(a, b, *, config=None, gap_tol=1e-8, stability_tol=1e-10, seed=0,
              response_config=None):
    """Real symmetric A/B RPA, matrix-free Davidson or bounded dense reference.

    Operator callers assert symmetry. Davidson screens A-B and A+B using
    independent lowest Ritz solves, with full-support deterministic guesses.
    Positive converged Ritz values are numerical diagnostics, NOT a rigorous
    global stability certificate: stability_certified is False for this path.
    Requested/guard physical residuals and positive metric norm are checked.
    First-order AD applies only to an entirely isolated requested root set.
    Vector-response solves are attached lazily and run only under AD.
    response_config optionally configures the shared response linear solve.
    Its direct capacity is n for dense reduction and 2*n for the structured
    metric equation; it does not change the forward eigensolver or gaps.
    """
    cfg = (
        EigenSolverConfig(gradient_mode="implicit_eigenvector")
        if config is None
        else config
    )
    if response_config is not None and not isinstance(response_config, LinearSolverConfig):
        raise TypeError("response_config must be a LinearSolverConfig")
    if cfg.value_min is not None:
        raise ValueError("value_min is a Hermitian interval control, not an RPA frequency window")

    if any(not isfinite(t) or t <= 0 for t in (gap_tol, stability_tol)):
        raise ValueError("RPA tolerances must be finite and positive")
    if not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ao, bo = as_operator(a), as_operator(b)
    validate_real_square(ao)
    validate_real_square(bo)
    if ao.shape != bo.shape:
        raise ValueError("RPA A and B must have matching shapes")
    n = ao.shape[0]
    if cfg.nroots > n:
        raise ValueError("nroots exceeds RPA dimension")
    dtype = jnp.result_type(ao.dtype, bo.dtype)
    if cfg.method == "dense":
        if n > cfg.max_dense:
            raise ValueError("Dense RPA exceeds max_dense")
        eye = jnp.eye(n, dtype=dtype)
        return solve_stable_rpa(
            ao.apply(eye),
            bo.apply(eye),
            config=cfg,
            gap_tol=gap_tol,
            stability_tol=stability_tol,
            response_config=response_config,
        )
    if ao.diagonal is None or bo.diagonal is None:
        raise ValueError("RPA Davidson requires diagonals of A and B")
    nsolve = min(cfg.nroots + 1, n)
    width = min(n, 40 if cfg.max_subspace is None else cfg.max_subspace)
    if width < min(n, nsolve + 2):
        raise ValueError(
            "max_subspace must include guard roots and two correction slots"
        )
    structure = jnp.asarray(True)
    for original in (a, b):
        if not isinstance(original, LinearOperator):
            original = jnp.asarray(original)
            tol = (
                32 * jnp.finfo(dtype).eps * jnp.maximum(1.0, jnp.linalg.norm(original))
            )
            structure &= jnp.all(jnp.isfinite(original)) & (
                jnp.linalg.norm(original - original.T) <= tol
            )
    margins, residuals, checks = [], [], []
    for sign in (-1, 1):
        diag = ao.diagonal + sign * bo.diagonal
        op = LinearOperator(
            ao.shape,
            dtype,
            lambda x: ao.apply(x) + sign * bo.apply(x),
            diagonal=diag,
            matmat=lambda x: ao.apply(x) + sign * bo.apply(x),
        )
        state = solve_hermitian(
            op,
            config=replace(
                cfg, nroots=1, gradient_mode="eigenvalue_only", max_subspace=width
            ),
            initial_vectors=_initial(diag, width, 1, seed),
        )
        state = jax.tree.map(jax.lax.stop_gradient, state)
        margins.append(state.values[0])
        residuals.append(state.residual_norms[0])
        checks.append(state.converged[0])
    margins, residuals = jnp.stack(margins), jnp.stack(residuals)
    stable = (
        structure
        & jnp.all(jnp.stack(checks))
        & jnp.all(margins > stability_tol + residuals)
    )

    def h(v):
        return jnp.concatenate(
            [ao.apply(v[:n]) + bo.apply(v[n:]), bo.apply(v[:n]) + ao.apply(v[n:])],
            axis=0,
        )

    def vind(rows):
        return _metric(h(rows.T)).T

    _, x, y, _ = _davidson_lowest_tdhf(
        lambda rows: jax.lax.stop_gradient(vind(rows)),
        nroots=nsolve,
        size=n,
        diag=ao.diagonal,
        tol=cfg.atol,
        max_iter=cfg.maxiter,
        max_subspace=width,
        orth_eps=min(1e-10, 0.01 * cfg.atol),
        initial_vectors=_initial(ao.diagonal, width, nsolve, seed),
    )
    metric_norm = jnp.sum(x * x - y * y, axis=0)
    z = (
        jnp.concatenate([x, y], axis=0)
        / jnp.sqrt(jnp.where(metric_norm > 0, metric_norm, 1.0))[None, :]
    )
    z = jax.lax.stop_gradient(z)
    applied = h(z)
    values = jnp.sum(z * applied, axis=0)
    physical_residuals = jnp.linalg.norm(applied - _metric(z) * values[None, :], axis=0)
    converged = (
        stable
        & (metric_norm > 0)
        & jnp.isfinite(values)
        & (values > gap_tol)
        & (physical_residuals <= cfg.atol)
    )
    distances = jnp.abs(values[: cfg.nroots, None] - values[None, :])
    distances = distances.at[jnp.arange(cfg.nroots), jnp.arange(cfg.nroots)].set(
        jnp.inf
    )
    isolated = jnp.all(
        distances
        > gap_tol + physical_residuals[: cfg.nroots, None] + physical_residuals[None, :]
    )
    response = jnp.all(converged) & isolated
    if cfg.gradient_mode == "implicit_eigenvector":
        z = _attach_response(
            h,
            values[: cfg.nroots],
            z[:, : cfg.nroots],
            ao.diagonal,
            cfg,
            response_config,
            response,
        )
        z = require_converged_derivative(z, response)
    else:
        z = z[:, : cfg.nroots]
    # Guards on both sides of masking are necessary: the outer guard invalidates
    # JVPs of the NaN branch; the inner one keeps its zero cotangent from silently
    # becoming a finite VJP after where discards the outer NaN cotangent.
    values = require_converged_derivative(values[: cfg.nroots], response)
    values = require_converged_derivative(jnp.where(stable, values, jnp.nan), response)
    z = jnp.where(stable, z, jnp.nan)
    if cfg.gradient_mode == "implicit_eigenvector":
        z = require_converged_derivative(z, response)
    return RPAResult(
        values,
        z[:n],
        z[n:],
        physical_residuals[: cfg.nroots],
        converged[: cfg.nroots],
        stable,
        jnp.broadcast_to(response, (cfg.nroots,)),
        margins,
        jnp.asarray(False),
        residuals,
    )
