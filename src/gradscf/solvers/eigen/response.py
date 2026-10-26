"""Implicit invariant-subspace response; rank one is an isolated eigenstate."""

from typing import Literal, NamedTuple
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from ..nonlinear.root import attach_root
from ..linear.implicit import checked_linear_solve
from ..diagnostics import require_converged_derivative
from ..memory import plan_linear
from gradscf._memory import array_storage_bytes, jaxpr_temporary_bytes, tree_storage_bytes

EigenGradientMode = Literal["eigenvalue_only", "implicit_eigenvector"]


class _ResponseBudget(NamedTuple):
    max_memory: float
    reserved_bytes: int


def _normalize_frame(y, isolated):
    if isolated:
        return y/jnp.sqrt(jnp.sum(y*y, axis=0, keepdims=True))
    gram = y.T @ y
    lower = jnp.linalg.cholesky(.5*(gram+gram.T))
    return solve_triangular(lower, y.T, lower=True).T


def _response_plan(residual, seed, x, config, valid, isolated, budget):
    """Plan the existing tangent solve, including its actual action batch."""
    m = x.shape[0] if isolated else seed.size
    # The joint RHS, response output and cotangent coexist with one numerical
    # column solve. The stopped frame and seed are explicit retained owners.
    resident = budget.reserved_bytes+array_storage_bytes(x, seed, valid)+3*seed.nbytes
    rhs_bytes = m*seed.dtype.itemsize
    plan_linear(m, seed.dtype, config, budget.max_memory,
                resident_bytes=resident+rhs_bytes)

    def column(point, vector, index):
        tangent = jnp.zeros_like(point).at[:, index].set(vector) if isolated else vector
        result = jax.jvp(residual, (point,), (tangent,))[1]
        return result[:, index] if isolated else result

    def payload(point, vectors, index, transpose):
        action = lambda vector: column(point, vector, index)
        if transpose:
            original = action
            action = lambda vector: jax.linear_transpose(
                original, jnp.zeros((m,), dtype=seed.dtype))(vector)[0]
        return (action(vectors) if vectors.ndim == 1 else
                jax.vmap(action, in_axes=1, out_axes=1)(vectors))

    shape = (m, m) if config.method == 'direct' else (m,)
    probes = jax.ShapeDtypeStruct(shape, seed.dtype)
    index = jax.ShapeDtypeStruct((), jnp.result_type(int))
    scratch = sum(jaxpr_temporary_bytes(jax.make_jaxpr(
        lambda point, vectors, i: payload(point, vectors, i, transpose)
    )(seed, probes, index)) for transpose in (False, True))
    # Frame normalization follows the numerical solve and uses the same
    # original budget. Its declared arrays do not overlap the action phase.
    scratch = max(scratch, jaxpr_temporary_bytes(jax.make_jaxpr(
        lambda y: _normalize_frame(y, isolated))(x)))
    planned, _ = plan_linear(m, seed.dtype, config, budget.max_memory,
        resident_bytes=resident+rhs_bytes, operator_min_bytes=scratch)
    # attach_root's default flat tangent solve adds its seed itself. The
    # isolated custom tangent solve owns that charge directly.
    reserved = resident if isolated else resident-seed.nbytes
    return planned, dict(max_memory=budget.max_memory,
        reserved_bytes=reserved, operator_min_bytes=scratch)


def _implicit_frame(apply, basis, *, config, valid, isolated=False, budget=None):
    """Attach the exact local invariant-space root, including higher response.

    Hold a numerical orthonormal frame X fixed and write Y=X+QZ, Q=I-XX.T.
    Since X.T Y=I, invariance AY=YH implies H=X.T AY. The square residual is
    Q(AY-Y(X.T AY))+XX.T Z=0. Its first tangent is the usual projected
    Sylvester equation; its nonlinear terms retain the higher-order response.
    Normalize Y using its small Gram matrix after attaching the shared root.

    With ``isolated=True``, each column uses its own rank-one projector and
    normalization; all individual roots must be isolated. Otherwise the
    rank-k frame may only feed basis-invariant observables, and no division
    by gaps inside its selected cluster is introduced. The external gap,
    Euclidean metric and fixed selected rank are still required.
    """
    x = jax.lax.stop_gradient(basis)
    n, k = x.shape
    if isolated:
        # Independent rank-one charts retain rotations between selected roots.
        project = lambda z: x*jnp.sum(x*z, axis=0, keepdims=True)
    else:
        project = lambda z: x @ (x.T @ z)
    complement = lambda z: z - project(z)

    def residual(v):
        z = v.reshape(n, k)
        y = x + complement(z)
        ay = apply(y)
        if isolated:
            physical = complement(ay-y*jnp.sum(x*ay, axis=0, keepdims=True))+project(z)
        else:
            physical = (complement(ay - y @ (x.T @ ay)) + project(z)).reshape(-1)
        return jnp.where(valid, physical, v)

    # The converged numerical Ritz frame supplies the root Z=0. Differentiating
    # this root, rather than a one-step correction with frozen X/H, preserves
    # state dependence under nested JVP/VJP without unrolling the Ritz solver.
    seed = jnp.zeros_like(x) if isolated else jnp.zeros(n*k, dtype=x.dtype)
    planned, memory = ((config, {}) if budget is None else
        _response_plan(residual, seed, x, config, valid, isolated, budget))
    if isolated:
        def tangent_solve(operator, rhs):
            # The root's Jacobian has independent n-dimensional column blocks.
            # Keep the existing checked solve and max_dense bound per root.
            def column(index, value):
                def action(vector):
                    block = jnp.zeros_like(rhs).at[:, index].set(vector)
                    return operator(block)[:, index]
                return checked_linear_solve(action, value, config=planned,
                                            converged=valid, **memory)[0]
            if budget is None:
                return jax.vmap(column, in_axes=(0, 1), out_axes=1)(jnp.arange(k), rhs)
            # Keep this loop inside the original single custom_root tangent
            # solve: moving spectral roots into a scan loses higher response.
            return jax.lax.map(lambda i: column(i, rhs[:, i]), jnp.arange(k)).T

        correction = attach_root(residual, seed, config=planned,
            tangent_solve=tangent_solve, converged=valid, **memory)
    else:
        correction = attach_root(residual, seed,
                                 config=planned, converged=valid, **memory)
    y = x + complement(correction.reshape(n, k))
    return _normalize_frame(y, isolated)


@partial(jax.custom_jvp, nondiff_argnums=(0, 3, 5, 6))
def _frame(operator, parameters, reference, config, valid, isolated, budget):
    return reference


@_frame.defjvp
def _frame_jvp(operator, config, isolated, budget, primals, tangents):
    parameters, reference, valid = primals
    dparameters, _, _ = tangents
    attached = lambda p: _implicit_frame(
        # Apply the derivative guard after closure conversion. Keeping valid
        # explicit avoids a captured bool tracer, while this inner guard also
        # rejects VJPs through an invalid root whose physical tangent is zero.
        lambda z: require_converged_derivative(operator(z, *p), valid),
        reference, config=config, valid=valid, isolated=isolated, budget=budget)
    tangent = jax.jvp(attached, (parameters,), (dparameters,))[1]
    return _frame(operator, parameters, reference, config, valid, isolated, budget), tangent


@partial(jax.custom_jvp, nondiff_argnums=(0, 3, 5, 6))
def _values(operator, parameters, reference, config, valid, isolated, budget):
    return jnp.sum(reference*operator(reference, *parameters), axis=0)


@_values.defjvp
def _values_jvp(operator, config, isolated, budget, primals, tangents):
    parameters, reference, valid = primals
    dparameters, _, _ = tangents
    y = _frame(operator, parameters, reference, config, valid, isolated, budget)
    # Hold y fixed in this partial JVP; outer derivatives retain its response.
    day = jax.jvp(lambda p: operator(y, *p), (parameters,), (dparameters,))[1]
    return _values(operator, parameters, reference, config, valid, isolated, budget), jnp.sum(y*day, axis=0)


def _spectral_inputs(apply, basis):
    """Hoist physical inputs and retain a stopped, explicit numerical frame.

    A numerical forward or first energy derivative needs no response solve.
    Module-level rules avoid capturing a vmapped frame in a custom JVP that
    an outer derivative can invoke after the original batch/JIT scope ends.
    """
    x = jax.lax.stop_gradient(basis)
    operator, parameters = jax.closure_convert(apply, x)
    return operator, tuple(parameters), x


def _parameter_budget(budget, parameters):
    # Only static shape/PyTree metadata enters a hashable nondiff carrier.
    # Caller reservations and distinct traced parameter views can overlap
    # conservatively; this is not allocation or outer-AD-tape introspection.
    return (None if budget is None else budget._replace(
        reserved_bytes=budget.reserved_bytes+tree_storage_bytes(parameters)))


def attach_spectral_response(apply, basis, *, config, valid=True, isolated=False,
                             _budget=None):
    """Normalize isolated columns or a complete invariant-subspace frame."""
    operator, parameters, x = _spectral_inputs(apply, basis)
    return _frame(operator, parameters, x, config, valid, isolated,
                  _parameter_budget(_budget, parameters))


def spectral_values(apply, basis, *, config, valid=True, isolated=False, _budget=None):
    """Stationary eigenvalue differential, with response retained at higher order.

    ``isolated=True`` supplies separate isolated-root energies. For a complete
    cluster, only their sum is a basis-invariant energy observable.
    """
    operator, parameters, x = _spectral_inputs(apply, basis)
    return _values(operator, parameters, x, config, valid, isolated,
                   _parameter_budget(_budget, parameters))
