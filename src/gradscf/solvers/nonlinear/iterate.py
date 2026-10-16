"""Residual-driven preconditioned iterations and DIIS, with shared implicit AD."""

from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .diis import diis_push, diis_solve
from .root import attach_root


@dataclass(frozen=True)
class NonlinearConfig:
    maxiter: int = 100
    residual_tol: float = 1e-9
    energy_tol: float = 1e-10
    diis_space: int = 6
    diis_start: int = 2
    damping: float = 0.0

    def __post_init__(self):
        if self.maxiter < 1 or self.diis_space < 1 or self.diis_start < 1:
            raise ValueError("Iteration/history limits must be positive")
        if (
            any(not isfinite(x) or x <= 0 for x in (self.residual_tol, self.energy_tol))
            or not isfinite(self.damping)
            or not 0 <= self.damping < 1
        ):
            raise ValueError("Invalid nonlinear tolerance or damping")


class NonlinearResult(NamedTuple):
    solution: object
    iterations: object
    residual_norm: object
    energy_change: object
    converged: object


def solve_nonlinear(
    residual,
    initial,
    *,
    update,
    observable,
    config=None,
    linear_config=None,
    valid_inputs=True,
):
    """Converge a real vector residual, then differentiate its root, not DIIS.

    update(x,R) supplies a preconditioned trial vector. observable supplies the
    scalar convergence monitor. Both the true infinity-norm residual and scalar
    change must converge; damping/history do not alter the differentiated equation.
    """
    cfg = NonlinearConfig() if config is None else config
    initial = jnp.asarray(initial)
    if initial.ndim != 1:
        raise ValueError("Nonlinear iterates must be packed vectors")
    if not jnp.issubdtype(initial.dtype, jnp.floating):
        raise ValueError("Nonlinear iterates must be real floating-point vectors")
    if initial.size == 0:
        return NonlinearResult(
            initial,
            jnp.array(0),
            jnp.array(0.0),
            jnp.array(0.0),
            jnp.asarray(valid_inputs),
        )
    def numerical_forward(seed):
        zeros = jnp.zeros((cfg.diis_space, seed.size), dtype=seed.dtype)
        norm = lambda x: jnp.max(jnp.abs(residual(x)))
        state = (
            seed,
            observable(seed),
            norm(seed),
            jnp.array(jnp.inf, seed.dtype),
            jnp.array(0, jnp.int32),
            zeros,
            zeros,
            jnp.array(0, jnp.int32),
            jnp.array(0, jnp.int32),
        )

        def condition(s):
            done = (s[2] <= cfg.residual_tol) & (s[3] <= cfg.energy_tol)
            return (
                (~done)
                & (s[4] < cfg.maxiter)
                & jnp.isfinite(s[2])
                & jnp.asarray(valid_inputs)
            )

        def step(s):
            x, e, r, de, k, hist, err, head, count = s
            trial = update(x, residual(x))
            trial = (1 - cfg.damping) * trial + cfg.damping * x
            hist, err, head, count = diis_push(
                trial, residual(trial), hist, err, head, count
            )
            trial = jax.lax.cond(
                (count >= 2) & (k + 1 >= cfg.diis_start),
                lambda _: diis_solve(hist, err, count),
                lambda _: trial,
                None,
            )
            new_energy = observable(trial)
            return (
                trial,
                new_energy,
                norm(trial),
                jnp.abs(new_energy - e),
                k + 1,
                hist,
                err,
                head,
                count,
            )

        return jax.lax.while_loop(condition, step, state)

    # Stop every physical input before tracing derivatives of the numerical
    # trajectory. Stopping only the returned state is too late for nested AD:
    # residual/update/observable can close over another implicit root's tangent.
    forward, parameters = jax.closure_convert(numerical_forward, initial)
    state = forward(jax.lax.stop_gradient(initial),
                    *jax.tree.map(jax.lax.stop_gradient, parameters))
    x, _, r, de, k, *_ = jax.tree.map(jax.lax.stop_gradient, state)
    valid = (
        jnp.asarray(valid_inputs)
        & jnp.isfinite(r)
        & (r <= cfg.residual_tol)
        & (de <= cfg.energy_tol)
    )
    x = attach_root(residual, x, config=linear_config, converged=valid)
    return NonlinearResult(x, k, r, de, valid)
