"""Bounded JAX L-BFGS minimization on the real unit sphere.

This is only a forward algorithm. Consumers attach their physical constrained
stationarity equations with the shared root/SCF response interfaces.
"""
from dataclasses import dataclass
from functools import partial
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from .minimize import _backtrack, _safe_lbfgs_state
from ..linear import solve_linear
from ..operators import LinearOperator
from ..types import LinearSolverConfig


@dataclass(frozen=True)
class SphereConfig:
    maxiter: int = 300
    tolerance: float = 1e-8
    memory: int = 10
    response_refinements: int = 0
    refine_stationary: bool = False

    def __post_init__(self):
        if self.maxiter < 1 or self.memory < 1 or self.response_refinements < 0 or not isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError('Positive iteration limits and finite positive tolerance required.')


class SphereResult(NamedTuple):
    solution: object
    energy: object
    residual_norm: object
    iterations: object
    converged: object


def _normalize(x):
    return x / jnp.sqrt(jnp.vdot(x, x).real)


@partial(jax.jit, static_argnames=('energy', 'config'))
def minimize_sphere(energy, initial, args, *, config=SphereConfig()):
    """Minimize energy(x,args) with x.T@x=1; scans also permit unrolled AD.

    A zero/nonfinite initial vector is invalid and yields nonfinite results.
    Convergence uses the tangential gradient, not an energy-change shortcut.
    """
    objective = jax.value_and_grad(lambda x: energy(_normalize(x), args))
    x = _normalize(jnp.asarray(initial))
    value, gradient = objective(x)
    transform = optax.scale_by_lbfgs(memory_size=config.memory, scale_init_precond=False)
    state = _safe_lbfgs_state(transform, x, gradient)

    def step(carry, _):
        def update(c):
            x, value, grad, state, count, active = c
            curvature = jnp.vdot(x-state.params, grad-state.updates).real
            state = jax.lax.cond((state.count > 0) & (curvature <= 0),
                lambda _: _safe_lbfgs_state(transform, x, grad), lambda _: state, None)
            direction, state = transform.update(grad, state, x)
            direction = -direction
            direction -= x*jnp.vdot(x, direction).real
            direction = jnp.where(jnp.vdot(direction, grad).real < 0, direction, -grad)
            direction /= jnp.maximum(1., jnp.linalg.norm(direction))
            trial, _, _, accepted, _ = _backtrack(objective, x, value, grad, direction)
            trial = _normalize(trial)
            new_value, new_grad = objective(trial)
            active = accepted & (jnp.linalg.norm(new_grad) > config.tolerance)
            return trial, new_value, new_grad, state, count+1, active
        return jax.lax.cond(carry[-1], update, lambda c: c, carry), None

    active = jnp.isfinite(value) & (jnp.linalg.norm(gradient) > config.tolerance)
    (x, value, gradient, _, count, _), _ = jax.lax.scan(jax.checkpoint(step),
        (x, value, gradient, state, jnp.array(0), active), None, length=config.maxiter)
    # A symmetric primal can already be stationary while its unrolled tangent
    # has never responded to the perturbation. Differentiable Newton corrections
    # retain that response even when the primal step is zero (ATLAS Eq. 33 with
    # AD HVPs). These are local refinements, not a substitute for minimization.
    def refine(x):
        raw_gradient = jax.grad(lambda y: energy(y,args))
        g = raw_gradient(x)
        eta2 = jnp.vdot(x,g)
        project = lambda v: v-x*jnp.vdot(x,v)
        def action(v):
            p = project(v)
            hp = jax.jvp(raw_gradient,(x,),(p,))[1]-eta2*p
            return project(hp)+x*jnp.vdot(x,v)
        # Bounded dense reference for small Hessians (32^2 entries maximum).
        # Larger problems retain only the HVP and Krylov workspace.
        method = 'direct' if x.size<=32 else 'gmres'
        correction = solve_linear(LinearOperator((x.size,x.size),x.dtype,action),
            -project(g),config=LinearSolverConfig(method=method,rtol=1e-10,atol=1e-14,
                                                 maxiter=20,restart=min(x.size,40),max_dense=32))
        return _normalize(x+correction.solution)
    for _ in range(config.response_refinements):
        norm = jnp.linalg.norm(objective(x)[1])
        active = (norm<=max(1e-3,config.tolerance)) & ((norm>config.tolerance) | config.refine_stationary)
        x = jax.lax.cond(active,
                         refine,lambda y:y,x)
    value, gradient = objective(x)
    norm = jnp.linalg.norm(gradient)
    return SphereResult(x, value, norm, count, jnp.isfinite(value) & (norm <= config.tolerance))
