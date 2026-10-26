"""Local isolated real eigenpairs using the shared implicit-root response."""

import jax
import jax.numpy as jnp

from ..diagnostics import require_converged_derivative
from ..linear.implicit import checked_linear_solve
from ..nonlinear.root import attach_root
from ..memory import apply_columns, plan_linear
from gradscf._memory import (array_storage_bytes, jaxpr_temporary_bytes,
                             validate_reserved_bytes, workspace_bytes)


def attach_pair_response(operator, values, right, left, *, config, valid,
                         max_memory=None, reserved_bytes=0, operator_min_bytes=0,
                         operator_bytes_per_column=0, action_columns=None):
    """Retain unit right norm and biorthogonal left norm on a simple branch.

    Each right root has n+1 unknowns (r, omega); each left root has n unknowns.
    Joint arrays keep one custom_root boundary while their tangent systems are
    solved columnwise. The direct capacity therefore bounds n+1, not (n+1)*k.
    There is no response through the numerical eigensolver or root selector.
    A budget reuses one checked solve at a time with the same physical equations;
    the no-budget path retains its existing vectorized root-column graph.
    """
    n, k = right.shape
    if config.method == "direct" and n+1 > config.max_dense:
        raise ValueError("Non-Hermitian pair response exceeds max_dense (n+1)")
    reserved_bytes = validate_reserved_bytes(reserved_bytes)

    def tangent_solve(action, rhs):
        def column(index, value):
            def apply(vector):
                block = jnp.zeros_like(rhs).at[:, index].set(vector)
                return action(block)[:, index]
            return checked_linear_solve(apply, value, config=config,
                                        converged=valid)[0]
        return jax.vmap(column, in_axes=(0, 1), out_axes=1)(jnp.arange(k), rhs)

    def phase(residual, seed, *held):
        if max_memory is None:
            return tangent_solve, {}
        m, item = seed.shape[0], seed.dtype.itemsize
        resident = reserved_bytes+array_storage_bytes(values, right, left, seed, *held)
        # Full joint RHS/output arrays remain while one root's numerical
        # system runs. The root seed is explicit because custom tangent_solve
        # owns its accounting, rather than attach_root's default flat solver.
        resident += 3*seed.size*item
        width = m if config.method == 'direct' else 1
        columns = k if action_columns is None else min(k, action_columns)
        fixed = width*validate_reserved_bytes(operator_min_bytes)
        per_columns = width*columns*validate_reserved_bytes(operator_bytes_per_column)
        minimum = fixed+per_columns
        if minimum > workspace_bytes(max_memory, resident_bytes=resident):
            raise ValueError('Pair response physical minimum exceeds the memory budget')

        def column(point, vector, index):
            block = jnp.zeros_like(point).at[:, index].set(vector)
            return jax.jvp(residual, (point,), (block,))[1][:, index]

        def payload(point, vectors, index, transpose):
            action = lambda vector: column(point, vector, index)
            if transpose:
                original = action
                action = lambda vector: jax.linear_transpose(original, point[:, 0])(vector)[0]
            return (action(vectors) if vectors.ndim == 1 else
                    jax.vmap(action, in_axes=1, out_axes=1)(vectors))

        shape = (m, m) if config.method == 'direct' else (m,)
        vector = jax.ShapeDtypeStruct(shape, seed.dtype)
        index = jax.ShapeDtypeStruct((), jnp.result_type(int))
        # Reuse the declared-array counter for both orientations and the
        # actual direct assembly width. No empirical scaling or new planner.
        # A caller's independent fixed workspace (e.g. an FFI slab) is not
        # visible among declared graph outputs. Preserve it alongside the
        # larger of the two declared-array estimates.
        scratch = fixed+max(per_columns, sum(jaxpr_temporary_bytes(jax.make_jaxpr(
            lambda point, probes, i: payload(point, probes, i, transpose)
        )(seed, vector, index)) for transpose in (False, True)))
        planned, _ = plan_linear(m, seed.dtype, config, max_memory,
            resident_bytes=resident+m*item, operator_min_bytes=scratch)
        budget = dict(max_memory=max_memory, reserved_bytes=resident,
                      operator_min_bytes=scratch)

        def sequential(action, rhs):
            def solve(index):
                def apply(vector):
                    block = jnp.zeros_like(rhs).at[:, index].set(vector)
                    return action(block)[:, index]
                return checked_linear_solve(apply, rhs[:, index], config=planned,
                    converged=valid, **budget)[0]
            return jax.lax.map(solve, jnp.arange(k)).T
        return sequential, budget

    def right_residual(state):
        r, omega = state[:-1], state[-1]
        ar = require_converged_derivative(apply_columns(operator, r, action_columns), valid)
        norm = .5*(jnp.sum(r*r, axis=0, keepdims=True)-1.)
        residual = jnp.concatenate((ar-r*omega, norm), axis=0)
        return jnp.where(valid, residual, state)

    initial = jnp.concatenate((right, values[None, :]), axis=0)
    right_solve, right_budget = phase(right_residual, initial)
    state = attach_root(right_residual, initial, config=config,
                        tangent_solve=right_solve, converged=valid, **right_budget)
    r, omega = state[:-1], state[-1]

    def left_residual(l):
        atl = require_converged_derivative(apply_columns(operator.T, l, action_columns), valid)
        residual = atl-l*omega+r*(jnp.sum(r*l, axis=0, keepdims=True)-1.)
        return jnp.where(valid, residual, l)

    left_solve, left_budget = phase(left_residual, left, initial, state, r, omega)
    l = attach_root(left_residual, left, config=config,
                    tangent_solve=left_solve, converged=valid, **left_budget)
    return tuple(require_converged_derivative(x, valid) for x in (omega, r, l))
