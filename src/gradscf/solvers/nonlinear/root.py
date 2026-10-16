"""Implicit differentiation of a converged root supplied by a caller."""
import jax
from ..diagnostics import require_converged_derivative
from ..linear.implicit import checked_linear_solve
from ..types import LinearSolverConfig


def attach_root(residual, solution, *, config=None, tangent_solve=None, converged=True):
    """Attach root JVP/VJP and higher response without differentiating the trajectory.

    residual may close over differentiable parameters. Numerical forward
    iteration and the physical convergence criterion are supplied by the caller.
    An optional tangent_solve supports structured response operators.
    """
    cfg = LinearSolverConfig() if config is None else config
    seed = jax.lax.stop_gradient(solution)
    if tangent_solve is None:
        def tangent_solve(operator, rhs):
            shape = rhs.shape
            return checked_linear_solve(
                lambda v: operator(v.reshape(shape)).reshape(-1), rhs.reshape(-1),
                config=cfg, converged=converged)[0].reshape(shape)
    return jax.lax.custom_root(residual, seed,
        solve=lambda _residual, _initial: seed, tangent_solve=tangent_solve)


def stationary_value(function, solution, parameters, *, converged=True):
    """Evaluate a scalar stationary objective with its envelope differential.

    The caller establishes d(function)/d(solution)=0 in an unconstrained
    coordinate chart. Constraints depending on parameters must remain inside
    ``function``; raw constrained orbitals or densities are not such a chart.
    All differentiable physical inputs must be explicit in ``parameters``;
    the callback may close over static metadata, not differentiable values.
    The first derivative is the partial parameter derivative. The solution
    stays differentiable, retaining its implicit response at higher orders.
    The target is the selected stationary branch, not a discrete minimum
    selector. This does not define a response for a singular or nonconverged state.
    """
    @jax.custom_jvp
    def value(state, inputs):
        return function(state, inputs)

    @value.defjvp
    def value_jvp(primals, tangents):
        state, inputs = primals
        _, dinputs = tangents
        tangent = jax.jvp(lambda p: function(state, p), (inputs,), (dinputs,))[1]
        return value(state, inputs), tangent

    return require_converged_derivative(value(solution, parameters), converged)
