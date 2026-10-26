"""Shared numerical Ritz solve and diagnostics, with no differentiated outputs."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from .dense import dense_vectors
from .davidson import _davidson_lowest_symmetric, _davidson_max_subspace
from ..memory import apply_columns


class RitzResult(NamedTuple):
    values: object
    vectors: object
    residual_norms: object
    converged: object
    present: object
    complete: object
    lower_gap: object
    lower_residual: object


def solve_ritz(operator, config, *, initial_vectors=None, structure_valid=True,
               batch_size=None):
    n, k = operator.shape[0], config.nroots
    count = min(k + 1, n)
    if k > n:
        raise ValueError("nroots exceeds operator dimension")
    apply = lambda v: apply_columns(operator, v, batch_size)
    complete = jnp.asarray(False)
    present = jnp.ones(count, dtype=bool)
    lower_gap, lower_residual = jnp.asarray(jnp.inf, operator.dtype), jnp.asarray(
        0.0, operator.dtype
    )
    if config.method == "dense":
        if initial_vectors is not None:
            raise ValueError("initial_vectors applies only to Davidson")
        x = dense_vectors(
            operator,
            nroots=count if config.value_min is None else n,
            max_dense=config.max_dense,
            batch_size=batch_size,
        )
        complete = jnp.asarray(True)
        if config.value_min is not None:
            all_applied = apply(x)
            theta = jnp.sum(x * all_applied, axis=0)
            nearest = jnp.argmin(jnp.abs(theta - config.value_min))
            lower_gap = jnp.abs(theta[nearest] - config.value_min)
            lower_residual = jnp.linalg.norm(
                all_applied[:, nearest] - x[:, nearest] * theta[nearest]
            )
            order = jnp.argsort(jnp.where(theta > config.value_min, theta, jnp.inf))[
                :count
            ]
            present = theta[order] > config.value_min
            x = x[:, order] * present[None, :]
    else:
        if operator.diagonal is None:
            raise ValueError("Davidson requires an operator diagonal approximation")
        if config.max_subspace is not None and config.max_subspace < min(n, count + 2):
            raise ValueError(
                "max_subspace must include nroots plus one boundary root and two expansion slots (or the full dimension)"
            )
        width = _davidson_max_subspace(count, n, config.max_subspace)
        if initial_vectors is None:
            guess_count = min(
                width,
                max(
                    count,
                    (
                        2 * count
                        if config.initial_guess_count is None
                        else config.initial_guess_count
                    ),
                ),
            )
            guesses = jax.nn.one_hot(
                jnp.argsort(operator.diagonal)[:count], n, dtype=operator.dtype
            ).T
            initial_vectors = jax.random.normal(
                jax.random.PRNGKey(config.seed), (n, guess_count), dtype=operator.dtype
            ) / jnp.sqrt(float(n))
            initial_vectors = initial_vectors.at[:, :count].add(guesses)
        elif jnp.iscomplexobj(initial_vectors):
            raise NotImplementedError("Public solvers require real initial vectors")
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(operator.diagonal)))
        numerical = _davidson_lowest_symmetric(
            lambda v: jax.lax.stop_gradient(apply(v)),
            nroots=count,
            size=n,
            diag=operator.diagonal,
            tol=config.atol,
            max_iter=config.maxiter,
            max_subspace=width,
            collapse_subspace=config.collapse_subspace,
            max_trial_vectors=config.max_trial_vectors,
            initial_vectors=initial_vectors,
            orth_eps=jnp.minimum(1e-10, 0.01 * config.atol / scale),
            positive_eig_threshold=config.value_min,
            return_basis=config.value_min is not None,
        )
        if config.value_min is None:
            _, x, _ = numerical
        else:
            basis, abasis, active = numerical
            # Audit the final bounded basis, including the case where fewer
            # than k+1 eigenvalues exist in the selected interval. No second
            # Davidson run or full physical operator materialization.
            h = basis.T @ abasis
            h = 0.5 * (h + h.T)
            shift = (jnp.linalg.norm(h) + 1.0) * 1e6
            h = jnp.where(active[:, None] & active[None, :], h, 0.0) + jnp.diag(
                (~active) * shift
            )
            theta, coeff = jnp.linalg.eigh(h)
            actual = theta < 0.5 * shift
            nearest = jnp.argmin(
                jnp.where(actual, jnp.abs(theta - config.value_min), jnp.inf)
            )
            lower_gap = jnp.abs(theta[nearest] - config.value_min)
            lower_residual = jnp.linalg.norm(
                abasis @ coeff[:, nearest]
                - (basis @ coeff[:, nearest]) * theta[nearest]
            )
            eligible = (theta > config.value_min) & actual
            order = jnp.argsort(jnp.where(eligible, theta, jnp.inf))[:count]
            present = eligible[order]
            x = (basis @ coeff[:, order]) * present[None, :]
            complete = jnp.all(active) & (width == n)
    x = jax.lax.stop_gradient(x)
    ax = apply(x)
    values = jnp.sum(x * ax, axis=0)
    residual = jnp.linalg.norm(ax - x * values[None, :], axis=0)
    orth = jnp.linalg.norm(x.T @ x - jnp.diag(present.astype(x.dtype)))
    valid = (
        jnp.isfinite(values)
        & present
        & (residual <= config.atol)
        & structure_valid
        & (orth <= 128 * jnp.finfo(x.dtype).eps * max(n, 1))
    )
    return jax.tree.map(
        jax.lax.stop_gradient,
        RitzResult(
            values, x, residual, valid, present, complete, lower_gap, lower_residual
        ),
    )
