"""JAX orbital iterations shared by implicit and unrolled SCF backward modes."""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
import optax
from ..memory import plan_linear
from ..types import LinearSolverConfig
from gradscf._memory import (jaxpr_temporary_bytes, tree_storage_bytes,
    validate_memory_mb, validate_reserved_bytes, workspace_bytes)


class OrbitalIterates(NamedTuple):
    coeff: jax.Array
    gradient_norm: jax.Array
    iterations: jax.Array
    evaluations: jax.Array
    stages: tuple
    polishing: tuple


def _safe_lbfgs_state(transform, x, gradient):
    # Optax 0.2.6 forms 1/<delta_x,delta_g> before discarding the first
    # secant pair. A nonzero dummy pair avoids 0*inf in reverse mode.
    state = transform.init(x)
    return state._replace(params=x-jnp.ones_like(x), updates=gradient-jnp.ones_like(gradient))


def _backtrack(value_grad, x, value, gradient, direction, *, correction=False, gradient_floor=0.):
    """Discrete halving search: reverse mode follows the actually accepted step."""
    allowance = 256*jnp.finfo(value.dtype).eps*jnp.maximum(1., jnp.abs(value))
    slope = jnp.vdot(gradient, direction).real
    old_norm = jnp.linalg.norm(gradient)

    def step(carry, index):
        def trial(_):
            candidate = x + jnp.exp2(-index.astype(value.dtype))*direction
            new_value, new_gradient = value_grad(candidate)
            new_norm = jnp.linalg.norm(new_gradient)
            reduced = (new_norm < old_norm) | (new_norm <= gradient_floor)
            resolved = new_value < value-allowance
            armijo = new_value <= value + 1e-4*jnp.exp2(-index.astype(value.dtype))*slope
            rounding = (old_norm <= 2e-3) & reduced & (new_value <= value+allowance)
            accepted = (rounding if correction else ((armijo & resolved) | rounding))
            accepted &= jnp.isfinite(new_value) & jnp.all(jnp.isfinite(new_gradient))
            return (jnp.where(accepted, candidate, x), jnp.where(accepted, new_value, value),
                    jnp.where(accepted, new_gradient, gradient), accepted, carry[4]+1)
        return jax.lax.cond(carry[3], lambda _: carry, trial, operand=None), None

    return jax.lax.scan(step, (x, value, gradient, jnp.asarray(False), jnp.asarray(0)),
                        jnp.arange(24))[0]


def _check_orbital_memory(args, coeff, energy, rotate, dimension, maximum, reserved, physical,
                          hessian_diagonal):
    """Preflight the fixed algorithm; no L-BFGS or primal GMRES capacity changes."""
    resident = reserved+tree_storage_bytes(args, coeff)
    workspace_bytes(maximum, resident_bytes=resident)
    vector = jax.ShapeDtypeStruct((dimension,), coeff.real.dtype)
    transform = optax.scale_by_lbfgs(memory_size=20, scale_init_precond=False)
    state = jax.eval_shape(lambda x, g: _safe_lbfgs_state(transform, x, g), vector, vector)
    # Old/new secant state, angle/gradient/search vectors, and the two stage /
    # ten polishing diagnostics remain mandatory in the original algorithm.
    resident += 2*tree_storage_bytes(state)+8*dimension*vector.dtype.itemsize+128*8
    with_diagonal = dimension > 32 and hessian_diagonal is not None
    if with_diagonal:
        # Gap, shifted denominator, right variable and recovered physical step.
        resident += 4*dimension*vector.dtype.itemsize
    workspace_bytes(maximum, resident_bytes=resident)
    value_grad = jax.value_and_grad(energy)
    value, gradient = jax.eval_shape(value_grad, vector, coeff, args)
    objective = lambda x: value_grad(x, coeff, args)
    graph = lambda f, *a: jaxpr_temporary_bytes(jax.make_jaxpr(f)(*a))
    diagonal_scratch = graph(hessian_diagonal, coeff, args) if with_diagonal else 0
    scratch = (graph(value_grad, vector, coeff, args)
               +graph(transform.update, vector, state, vector)
               +graph(lambda x, v, g, d: _backtrack(objective, x, v, g, d),
                      vector, value, gradient, vector)
               +graph(rotate, vector, coeff)+diagonal_scratch)
    workspace_bytes(maximum, resident_bytes=resident+physical+scratch)
    if dimension:
        gradient = lambda x: value_grad(x, coeff, args)[1]
        if dimension <= 32:
            action = graph(jax.jacfwd(gradient), vector)
            config = LinearSolverConfig(method='direct', max_dense=32)
            fixed = dimension*physical
        else:
            action = graph(lambda x, v: jax.jvp(gradient, (x,), (v,))[1], vector, vector)
            config = LinearSolverConfig(restart=min(dimension, 40), maxiter=10, rtol=1e-11, atol=0.)
            fixed = physical
        planned, _ = plan_linear(dimension, vector.dtype, config, maximum,
            resident_bytes=resident, operator_min_bytes=fixed+action+diagonal_scratch)
        if config.method == 'gmres' and planned.restart != config.restart:
            raise ValueError('Forward memory budget must retain the fixed primal GMRES restart')


@partial(jax.jit, static_argnames=('energy', 'rotate', 'dimension', 'max_iterations',
    'gradient_tolerance', 'max_memory', 'reserved_bytes', 'operator_min_bytes', 'hessian_diagonal'))
def solve_orbitals(args, coeff, *, energy, rotate, dimension, max_iterations, gradient_tolerance,
                   max_memory=None, reserved_bytes=0, operator_min_bytes=0, hessian_diagonal=None):
    """Bounded scans retain a genuine reverse-mode path through finite iterates.

    An optional approximate Hessian diagonal scales the large primal Newton
    solves on the right; their physical operator and residual checks stay live.
    """
    if max_memory is not None:
        _check_orbital_memory(args, coeff, energy, rotate, dimension,
            validate_memory_mb(max_memory), validate_reserved_bytes(reserved_bytes),
            validate_reserved_bytes(operator_min_bytes), hessian_diagonal)
    origin = jnp.zeros(dimension, dtype=coeff.real.dtype)
    value_grad = jax.value_and_grad(energy)
    iterations, evaluations = jnp.asarray(0), jnp.asarray(0)
    stages, polishing = [], []
    norm = jnp.asarray(jnp.inf, dtype=origin.dtype)
    transform = optax.scale_by_lbfgs(memory_size=20, scale_init_precond=False)

    for stage in range(2):
        base = coeff
        objective = lambda x: value_grad(x, base, args)
        value, grad = objective(origin)
        allowed = (norm > gradient_tolerance) & (iterations < max_iterations)
        state = _safe_lbfgs_state(transform, origin, grad)

        def body(carry, _):
            def update(c):
                x, value, gradient, state, count, calls, _active = c
                sy = jnp.vdot(x-state.params, gradient-state.updates).real
                state = jax.lax.cond((state.count > 0) & (sy <= 0.),
                    lambda _: _safe_lbfgs_state(transform, x, gradient), lambda _: state, None)
                preconditioned, next_state = transform.update(gradient, state, x)
                direction = -preconditioned
                direction = jnp.where(jnp.vdot(direction, gradient).real < 0., direction, -gradient)
                magnitude = jnp.linalg.norm(direction)
                direction = direction/jnp.maximum(1., magnitude)
                new_x, new_value, new_gradient, accepted, used = _backtrack(
                    objective, x, value, gradient, direction)
                active = accepted & (jnp.linalg.norm(new_gradient)*.5 > gradient_tolerance*.1)
                return new_x, new_value, new_gradient, next_state, count+1, calls+used, active
            active = carry[-1] & (carry[4] < max_iterations)
            return jax.lax.cond(active, update, lambda c: c, carry), None

        # Stationary seeds still receive the original Newton refinements below.
        initial_active = allowed & (jnp.linalg.norm(grad)*.5 > gradient_tolerance)
        carry, _ = jax.lax.scan(jax.checkpoint(body),
            (origin, value, grad, state, iterations, evaluations+1, initial_active),
            None, length=max_iterations)
        angles, _, _, _, new_iterations, evaluations, _ = carry
        coeff = rotate(angles, base)
        lbfgs_norm = jnp.linalg.norm(value_grad(origin, coeff, args)[1])*.5

        def polish(c, index):
            base, calls = c
            value, gradient = value_grad(origin, base, args)
            before = jnp.linalg.norm(gradient)*.5
            # A symmetric primal can converge before its parameter response.
            # Keep three actual differentiable Newton refinements at stationarity
            # instead of freezing an unconverged unrolled tangent trajectory.
            refining = before <= gradient_tolerance
            active = allowed & (before <= 1e-3) & ((~refining) | (index < 3))

            def correct(_):
                # For small orbital spaces, materialize once: this avoids deeply
                # nested Krylov/HVP adjoint compilations in unrolled DFT. Larger
                # spaces retain matrix-free actions and do not allocate a Hessian.
                if dimension <= 32:
                    hessian = jax.jacfwd(lambda x: value_grad(x, base, args)[1])(origin)
                    hessian = .5*(hessian+hessian.T)
                    hvp = lambda vector: hessian @ vector
                else:
                    def hvp(vector):
                        return jax.jvp(lambda x: value_grad(x, base, args)[1], (origin,), (vector,))[1]
                    diagonal = (None if hessian_diagonal is None else jax.lax.stop_gradient(
                        jnp.asarray(hessian_diagonal(base, args), dtype=origin.dtype)))

                def try_shift(carry, shift):
                    def solve(_):
                        operator = lambda v: hvp(v) + shift*v
                        if dimension <= 32:
                            step = jnp.linalg.solve(hessian+shift*jnp.eye(dimension), -gradient)
                        else:
                            scaled = operator
                            if diagonal is not None:
                                # This floor changes only the numerical scaling,
                                # never the physical Hessian or its shift.
                                d = diagonal+shift
                                floor = jnp.asarray(1e-8, dtype=origin.dtype)
                                d = jnp.where(jnp.abs(d) >= floor, d, jnp.where(d < 0., -floor, floor))
                                scaled = lambda vector: operator(vector/d)
                            step, _ = gmres(scaled, -gradient, tol=1e-11, atol=0.,
                                restart=min(dimension, 40), maxiter=10, solve_method='incremental')
                            if diagonal is not None:
                                step = step/d
                        residual = jnp.linalg.norm(operator(step)+gradient)/jnp.maximum(
                            jnp.linalg.norm(gradient), jnp.finfo(origin.dtype).tiny)
                        descent = jnp.vdot(gradient, step).real
                        valid = (jnp.all(jnp.isfinite(step)) & (residual <= .1)
                                 & ((descent < 0.) | refining) & (jnp.linalg.norm(step) <= .1))
                        return step, residual, shift, valid
                    return jax.lax.cond(carry[3], lambda _: carry, solve, None), None

                step, residual, shift, valid = jax.lax.scan(try_shift,
                    (origin, jnp.asarray(jnp.inf, origin.dtype), jnp.asarray(0., origin.dtype), jnp.asarray(False)),
                    jnp.stack([jnp.where(refining, 1e-6, 1e-3),
                        *[jnp.asarray(v, origin.dtype) for v in (0., 1e-7, 1e-5, .1, 1.)]]))[0]

                def search(_):
                    # Compare gradients in each candidate's own tangent frame.
                    def trial_gradient(x):
                        return value_grad(origin, rotate(x, base), args)
                    x, v, g, accepted, used = _backtrack(trial_gradient, origin, value, gradient,
                        step, correction=True, gradient_floor=jnp.where(refining, 2*gradient_tolerance, 0.))
                    return rotate(x, base), used, accepted, jnp.linalg.norm(g)*.5, v-value, jnp.linalg.norm(x)
                new_base, used, accepted, after, change, step_norm = jax.lax.cond(valid, search,
                    lambda _: (base, jnp.asarray(0), jnp.asarray(False), before,
                               jnp.asarray(0., origin.dtype), jnp.asarray(0., origin.dtype)), None)
                return new_base, used, accepted, after, change, step_norm, residual, shift

            new_base, used, accepted, after, change, step_norm, residual, shift = jax.lax.cond(active, correct,
                lambda _: (base, jnp.asarray(0), jnp.asarray(False), before,
                    jnp.asarray(0., origin.dtype), jnp.asarray(0., origin.dtype),
                    jnp.asarray(0., origin.dtype), jnp.asarray(0., origin.dtype)), None)
            info = dict(accepted=accepted, refinement=refining, gradient_before=before, gradient_after=after,
                energy_change=change, accepted_step_norm=step_norm, linear_relative_residual=residual,
                hessian_shift=shift, energy_allowance=256*jnp.finfo(value.dtype).eps*jnp.maximum(1.,jnp.abs(value)))
            return (new_base, calls+used), info

        (coeff, evaluations), history = jax.lax.scan(jax.checkpoint(polish), (coeff, evaluations), jnp.arange(5))
        norm = jnp.linalg.norm(value_grad(origin, coeff, args)[1])*.5
        polishing.extend(jax.tree.map(lambda a: a[i], history) for i in range(5))
        stages.append(dict(stage=stage, lbfgs_iterations=new_iterations-iterations,
                           gradient_after_lbfgs=lbfgs_norm, gradient_final=norm,
                           polishing_steps=jnp.sum(history['accepted'])))
        iterations = new_iterations
    return OrbitalIterates(coeff, norm, iterations, evaluations, tuple(stages), tuple(polishing))
