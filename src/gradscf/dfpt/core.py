"""Response assembly on a supplied stationary state; no numerical solver copy."""
from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
from ..solvers import LinearOperator, LinearSolverConfig, solve_linear
from ..solvers.diagnostics import linear_residual
from ..solvers.nonlinear.taylor import root_taylor


class Response(NamedTuple):
    state: object
    values: object
    residual_norm: object
    converged: object

    @property
    def density(self):
        return self.values['density']

    @property
    def energy(self):
        return self.values['energy']


@dataclass(frozen=True, eq=False)
class Problem:
    """A fixed reference R(state, inputs)=0 and physical observables.

    response accepts an input PyTree tangent. series accepts a scalar path to
    inputs; its first axis holds normalized orders 1..n, not raw derivatives.
    potential=... is a convenience supplied by molecular/periodic adapters.
    """
    residual: object
    solution: object
    inputs: object
    quantities: object
    converged: object = True
    tolerance: float = 1e-7
    potential_tangent: object = None
    potential_path: object = None
    series_supported: bool = True

    def _base(self):
        if not isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError('Reference tolerance must be positive and finite')
        x = jnp.asarray(self.solution)
        if x.ndim != 1 or not jnp.issubdtype(x.dtype, jnp.floating):
            raise ValueError('Response state must be a real floating-point vector')
        r = self.residual(x, self.inputs)
        if r.shape != x.shape:
            raise ValueError('Response residual must be square')
        valid = self.converged & jnp.all(jnp.isfinite(r)) & (jnp.max(jnp.abs(r), initial=0.) <= self.tolerance)
        return x, valid

    def _tangent(self, tangent, potential):
        if (tangent is None) == (potential is None):
            raise ValueError('Supply exactly one of tangent or potential')
        if potential is not None:
            if self.potential_tangent is None:
                raise ValueError('This problem has no external-potential adapter')
            tangent = self.potential_tangent(potential)
        return tangent

    def response(self, tangent=None, *, potential=None, config=None):
        x, valid = self._base()
        tangent = self._tangent(tangent, potential)
        cfg = config or LinearSolverConfig(rtol=1e-10, atol=1e-12)
        _, action = jax.linearize(lambda y: self.residual(y, self.inputs), x)
        rhs = -jax.jvp(lambda a: self.residual(x, a), (self.inputs,), (tangent,))[1]
        op = LinearOperator((x.size, x.size), x.dtype, action)
        solved = solve_linear(op, rhs, config=cfg)
        norm, ok = linear_residual(action, solved.solution, rhs, rtol=cfg.rtol, atol=cfg.atol)
        valid &= ok & jnp.all(jnp.isfinite(solved.solution))
        state = solved.solution * jnp.where(valid, 1., jnp.nan)
        values = jax.jvp(self.quantities, (x, self.inputs), (state, tangent))[1]
        values = jax.tree.map(lambda a: a*jnp.where(valid, 1., jnp.nan), values)
        return Response(state, values, jax.lax.stop_gradient(norm), valid)

    def series(self, path=None, *, potential=None, order=2, config=None):
        if not self.series_supported:
            raise NotImplementedError('Taylor series is not available for this response model; use first response.')
        x, valid = self._base()
        if (path is None) == (potential is None):
            raise ValueError('Supply exactly one of path or potential')
        if potential is not None:
            tangent = self._tangent(None, potential)
            if self.potential_path is None:
                path = lambda t: jax.tree.map(lambda a, b: a+t*b, self.inputs, tangent)
            else:
                path = lambda t: self.potential_path(tangent, t)
        # A path is an expansion around the supplied reference, not a new SCF.
        base = path(jnp.array(0., x.dtype))
        if jax.tree.structure(base) != jax.tree.structure(self.inputs):
            raise ValueError('Path inputs must match the reference PyTree')
        for a, b in zip(jax.tree.leaves(base), jax.tree.leaves(self.inputs)):
            if jnp.shape(a) != jnp.shape(b):
                raise ValueError('Path input shapes must remain fixed')
            valid &= jnp.all(jnp.abs(a-b) <= self.tolerance)
        residual = lambda y, s: self.residual(y, path(s))
        result = root_taylor(residual, x, order=order, config=config, valid=valid)
        param_series = (jnp.array(1., x.dtype),) + (jnp.array(0., x.dtype),) * (order-1)
        base_values, terms = jet(lambda y, s: self.quantities(y, path(s)), (x, jnp.array(0., x.dtype)),
                        (tuple(result.coefficients), param_series), factorial_scaled=False)
        structure = jax.tree.structure(base_values)
        values = structure.unflatten([jnp.stack(leaf) for leaf in structure.flatten_up_to(terms)])
        values = jax.tree.map(lambda a: a*jnp.where(result.converged, 1., jnp.nan), values)
        return Response(result.coefficients, values, result.residual_norms, result.converged)
