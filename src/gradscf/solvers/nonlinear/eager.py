"""Checked eager fixed-point iteration for maps with host domain validation."""
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .diis import diis_push, diis_solve
from .iterate import NonlinearConfig


class EagerIterationResult(NamedTuple):
    solution: object
    iterations: int
    residual_norm: float
    converged: bool


def iterate_eager(physical_map, initial, *, config=None, valid_state=None,
                  check=None, callback=None):
    """Iterate a packed real vector using shared DIIS and true map residuals.

    Convergence requires max(abs(map(x)-x)) <= residual_tol at the returned
    state. Optional check(x,mapped,norm) can only add a convergence condition;
    callback(iteration,x,norm) observes it without changing that condition.
    NonlinearConfig.energy_tol is left to check because no physical energy is
    defined by a generic packed vector. diis_space=1 selects plain iteration.
    Invalid DIIS proposals use the damped plain map, then at most 16 halvings
    toward the current valid state. No expensive map is evaluated outside
    valid_state. This forward solver explicitly supplies no AD/JIT contract.
    """
    from jax._src import core
    if not core.trace_state_clean() or any(isinstance(x, jax.core.Tracer)
            for x in jax.tree_util.tree_leaves(initial)):
        raise NotImplementedError('iterate_eager is eager forward only; AD/JIT is unsupported')
    cfg = NonlinearConfig() if config is None else config
    state = jnp.asarray(initial)
    if state.ndim != 1 or not jnp.issubdtype(state.dtype, jnp.floating):
        raise ValueError('Use a packed real floating-point vector')
    def valid(value):
        return bool(jnp.all(jnp.isfinite(value))) and (valid_state is None or bool(valid_state(value)))
    if not valid(state):
        raise ValueError('The initial eager state is outside its finite physical domain')
    history = jnp.zeros((cfg.diis_space, state.size), dtype=state.dtype)
    errors = jnp.zeros_like(history)
    head, count = jnp.array(0), jnp.array(0)
    for iteration in range(cfg.maxiter+1):
        mapped = jnp.asarray(physical_map(state))
        if mapped.shape != state.shape or not jnp.issubdtype(mapped.dtype, jnp.floating):
            raise ValueError('The physical map must return a matching real floating-point vector')
        residual = mapped-state
        norm = float(jnp.max(jnp.abs(residual))) if state.size else 0.
        if callback is not None:
            callback(iteration, state, norm)
        done = norm <= cfg.residual_tol and (check is None or bool(check(state, mapped, norm)))
        if done or iteration == cfg.maxiter or not isfinite(norm):
            return EagerIterationResult(state, iteration, norm, done)
        plain = state+(1-cfg.damping)*residual
        history, errors, head, count = diis_push(plain, residual, history, errors, head, count)
        candidate = (diis_solve(history, errors, count)
                     if int(count) >= 2 and iteration+1 >= cfg.diis_start else plain)
        if not valid(candidate):
            candidate = plain
            for _ in range(16):
                if valid(candidate):
                    break
                candidate = .5*(candidate+state)
            if not valid(candidate):
                return EagerIterationResult(state, iteration, norm, False)
        state = candidate


__all__ = ['EagerIterationResult', 'iterate_eager']
