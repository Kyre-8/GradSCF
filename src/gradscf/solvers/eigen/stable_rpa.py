"""Bounded stable real RPA with metric-normalized isolated-state response."""

from math import isfinite
from dataclasses import replace

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from ..diagnostics import require_converged_derivative
from ..types import (
    EigenSolverConfig,
    EigenResponseConfig,
    LinearSolverConfig,
    RPAResult,
)
from .api import solve_hermitian


def solve_stable_rpa(a, b, *, config=None, gap_tol=1e-8, stability_tol=1e-10,
                     response_config=None):
    """Solve [[A,B],[-B,-A]] for real symmetric A and B with A+/-B > 0.

    This dense reference path is bounded by config.max_dense. Cholesky of
    M=A-B reduces the problem to L.T@(A+B)@L; no matrix-function eigenvector
    derivative or doubled nonsymmetric eigendecomposition is needed. Returned
    X/Y are columns, with X.T@X-Y.T@Y=I. First-order isolated-root response
    includes both the reduction and amplitude reconstruction. Instabilities
    are reported as invalid, with the two actual stability eigenvalue minima;
    no unstable roots are discarded or converted to real frequencies.
    response_config optionally supplies shared LinearSolverConfig controls
    for the n-dimensional reduced-state response, independently of the
    forward eigensolver. None preserves the existing GMRES configuration.
    """
    cfg = (
        EigenSolverConfig(method="dense", gradient_mode="implicit_eigenvector")
        if config is None
        else config
    )
    if response_config is not None and not isinstance(response_config, LinearSolverConfig):
        raise TypeError("response_config must be a LinearSolverConfig")
    if cfg.value_min is not None:
        raise ValueError(
            "value_min is a Hermitian interval control, not an RPA frequency window"
        )

    if cfg.method != "dense":
        raise NotImplementedError("Stable RPA currently requires method='dense'")
    a, b = jnp.asarray(a), jnp.asarray(b)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != a.shape:
        raise ValueError("RPA requires square A and B blocks of the same shape")
    n = a.shape[0]
    if n > cfg.max_dense:
        raise ValueError("Stable RPA exceeds max_dense")
    if not 1 <= cfg.nroots <= n:
        raise ValueError("Invalid number of RPA roots")
    if any(not isfinite(x) or x <= 0 for x in (gap_tol, stability_tol)):
        raise ValueError("RPA gap and stability tolerances must be finite and positive")
    if any(not jnp.issubdtype(x.dtype, jnp.floating) for x in (a, b)):
        raise ValueError("Stable RPA requires real floating-point blocks")
    dtype = jnp.result_type(a, b)
    a, b = a.astype(dtype), b.astype(dtype)
    identity = jnp.eye(n, dtype=dtype)
    symmetry_tol = (
        32
        * jnp.finfo(dtype).eps
        * jnp.maximum(1.0, jnp.linalg.norm(a) + jnp.linalg.norm(b))
    )
    structure = (
        jnp.all(jnp.isfinite(a))
        & jnp.all(jnp.isfinite(b))
        & (jnp.linalg.norm(a - a.T) <= symmetry_tol)
        & (jnp.linalg.norm(b - b.T) <= symmetry_tol)
    )
    m, p = a - b, a + b
    margins = jnp.stack(
        [
            jnp.linalg.eigvalsh(
                jnp.where(structure, jax.lax.stop_gradient(x), identity)
            )[0]
            for x in (m, p)
        ]
    )
    margins = jnp.where(structure, margins, jnp.nan)
    stable = structure & jnp.all(margins > stability_tol)
    # Identity is only an internal finite workspace for invalid inputs. All
    # physical outputs are masked, and input derivatives explicitly invalidated.
    m = jnp.where(stable, require_converged_derivative(m, stable), identity)
    p = jnp.where(stable, require_converged_derivative(p, stable), identity)
    chol = jnp.linalg.cholesky(m)
    reduced = chol.T @ p @ chol
    roots = solve_hermitian(
        reduced,
        config=replace(
            cfg, gradient_mode="eigenvalue_only", adjoint_tol=1e-10, adjoint_maxiter=100
        ),
        response=EigenResponseConfig(
            target=(
                "eigenpairs"
                if cfg.gradient_mode == "implicit_eigenvector"
                else "eigenvalues"
            ),
            gap_atol=0.0,
            gap_rtol=32 * jnp.finfo(dtype).eps,
            linear_config=response_config or LinearSolverConfig(
                rtol=cfg.adjoint_tol, maxiter=cfg.adjoint_maxiter
            ),
        ),
    )
    omega = jnp.sqrt(jnp.concatenate([roots.values, roots.raw_values[cfg.nroots :]]))
    z = jnp.concatenate([roots.vectors, roots.raw_vectors[:, cfg.nroots :]], axis=1)
    reduced_converged = roots.raw_residual_norms <= cfg.atol
    plus = (chol @ z) / jnp.sqrt(omega)[None, :]
    minus = solve_triangular(chol.T, z, lower=False) * jnp.sqrt(omega)[None, :]
    x, y = (plus + minus) * 0.5, (plus - minus) * 0.5
    residuals = jnp.sqrt(
        jnp.sum(
            (a @ x + b @ y - x * omega) ** 2 + (b @ x + a @ y + y * omega) ** 2, axis=0
        )
    )
    metric = jnp.sum(x * x - y * y, axis=0)
    converged = (
        stable
        & reduced_converged
        & jnp.isfinite(omega)
        & (residuals <= cfg.atol)
        & (
            jnp.abs(metric - 1)
            <= 64
            * jnp.finfo(dtype).eps
            * jnp.maximum(1.0, jnp.sum(x * x + y * y, axis=0))
        )
    )
    distance = jnp.abs(omega[: cfg.nroots, None] - omega[None, :])
    distance = distance.at[jnp.arange(cfg.nroots), jnp.arange(cfg.nroots)].set(jnp.inf)
    separated = jnp.all(
        distance > gap_tol + residuals[: cfg.nroots, None] + residuals[None, :], axis=1
    )
    response = (
        converged[: cfg.nroots]
        & roots.response_valid
        & jnp.all(converged)
        & separated
        & (omega[: cfg.nroots] > gap_tol)
    )
    # NaN derivative guards for one returned root also poison a zero cotangent
    # of that root. Advertise the conservative whole-request contract equally
    # for JVP and VJP; callers can request a smaller isolated prefix instead.
    response = jnp.broadcast_to(jnp.all(response), (cfg.nroots,))
    values = require_converged_derivative(
        jnp.where(stable, omega[: cfg.nroots], jnp.nan), response
    )
    x, y = [jnp.where(stable, v[:, : cfg.nroots], jnp.nan) for v in (x, y)]
    if cfg.gradient_mode == "eigenvalue_only":
        x, y = jax.lax.stop_gradient(x), jax.lax.stop_gradient(y)
    else:
        x, y = [require_converged_derivative(v, response[None, :]) for v in (x, y)]
    return RPAResult(
        values,
        x,
        y,
        residuals[: cfg.nroots],
        converged[: cfg.nroots],
        stable,
        response,
        margins,
        stable,
        jnp.zeros((2,), dtype=dtype),
    )
