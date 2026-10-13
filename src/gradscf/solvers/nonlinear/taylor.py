"""Normalized Taylor coefficients of a square, isolated real root."""
from numbers import Integral
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
from ..operators import LinearOperator
from ..linear import solve_linear
from ..types import LinearSolverConfig
from ..diagnostics import linear_residual


class TaylorResult(NamedTuple):
    coefficients: object
    residual_norms: object
    converged: object


def root_taylor(residual, solution, parameter=0., *, order, config=None, valid=True):
    """Solve J0*x_n = -[s**n]R(x0+sum_{j<n}x_j*s**j, parameter+s).

    Inputs must specify a converged root with fixed dimensions and a nonsingular
    Jacobian. This low-level routine checks linear solves; callers check roots
    and physical constraints. Coefficients include the 1/n! normalization.
    """
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
        raise ValueError('order must be a positive integer')
    x, s = jnp.asarray(solution), jnp.asarray(parameter)
    if x.ndim != 1 or s.shape != () or jnp.iscomplexobj(x) or jnp.iscomplexobj(s):
        raise ValueError('Taylor root requires a real vector and scalar parameter')
    cfg = config or LinearSolverConfig(rtol=1e-10, atol=1e-12)
    value, action = jax.linearize(lambda y: residual(y, s), x)
    if value.shape != x.shape:
        raise ValueError('Taylor residual must be square')
    op = LinearOperator((x.size, x.size), x.dtype, action)
    coefficients, norms = [], []
    ok = jnp.asarray(valid)
    for n in range(1, order + 1):
        series = (tuple(coefficients + [jnp.zeros_like(x)]),
                  (jnp.ones_like(s),) + (jnp.zeros_like(s),) * (n-1))
        _, terms = jet(residual, (x, s), series, factorial_scaled=False)
        rhs = -terms[-1]
        result = solve_linear(op, rhs, config=cfg)
        # Avoid custom_linear_solve auxiliary tangent-batch axes under jacfwd.
        norm, converged = linear_residual(action, result.solution, rhs, rtol=cfg.rtol, atol=cfg.atol)
        coefficients.append(result.solution)
        norms.append(jax.lax.stop_gradient(norm))
        ok &= converged & jnp.all(jnp.isfinite(result.solution))
    return TaylorResult(jnp.stack(coefficients) * jnp.where(ok, 1., jnp.nan), jnp.stack(norms), ok)
