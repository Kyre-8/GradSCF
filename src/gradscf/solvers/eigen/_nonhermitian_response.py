"""Local isolated real eigenpairs using the shared implicit-root response."""

import jax
import jax.numpy as jnp

from ..diagnostics import require_converged_derivative
from ..linear.implicit import checked_linear_solve
from ..nonlinear.root import attach_root


def attach_pair_response(operator, values, right, left, *, config, valid):
    """Retain unit right norm and biorthogonal left norm on a simple branch.

    Each right root has n+1 unknowns (r, omega); each left root has n unknowns.
    Joint arrays keep one custom_root boundary while their tangent systems are
    solved columnwise. The direct capacity therefore bounds n+1, not (n+1)*k.
    There is no response through the numerical eigensolver or root selector.
    """
    n, k = right.shape
    if config.method == "direct" and n+1 > config.max_dense:
        raise ValueError("Non-Hermitian pair response exceeds max_dense (n+1)")

    def tangent_solve(action, rhs):
        def column(index, value):
            def apply(vector):
                block = jnp.zeros_like(rhs).at[:, index].set(vector)
                return action(block)[:, index]
            return checked_linear_solve(apply, value, config=config,
                                        converged=valid)[0]
        return jax.vmap(column, in_axes=(0, 1), out_axes=1)(jnp.arange(k), rhs)

    def right_residual(state):
        r, omega = state[:-1], state[-1]
        ar = require_converged_derivative(operator.apply(r), valid)
        norm = .5*(jnp.sum(r*r, axis=0, keepdims=True)-1.)
        residual = jnp.concatenate((ar-r*omega, norm), axis=0)
        return jnp.where(valid, residual, state)

    initial = jnp.concatenate((right, values[None, :]), axis=0)
    state = attach_root(right_residual, initial, config=config,
                        tangent_solve=tangent_solve, converged=valid)
    r, omega = state[:-1], state[-1]

    def left_residual(l):
        atl = require_converged_derivative(operator.T.apply(l), valid)
        residual = atl-l*omega+r*(jnp.sum(r*l, axis=0, keepdims=True)-1.)
        return jnp.where(valid, residual, l)

    l = attach_root(left_residual, left, config=config,
                    tangent_solve=tangent_solve, converged=valid)
    return tuple(require_converged_derivative(x, valid) for x in (omega, r, l))
