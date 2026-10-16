"""Implicit invariant-subspace response; rank one is an isolated eigenstate."""

from typing import Literal
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from ..nonlinear.root import attach_root
from ..linear.implicit import checked_linear_solve
from ..diagnostics import require_converged_derivative

EigenGradientMode = Literal["eigenvalue_only", "implicit_eigenvector"]


def _implicit_frame(apply, basis, *, config, valid, isolated=False):
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
    if isolated:
        def tangent_solve(operator, rhs):
            # The root's Jacobian has independent n-dimensional column blocks.
            # Keep the existing checked solve and max_dense bound per root.
            def column(index, value):
                def action(vector):
                    block = jnp.zeros_like(rhs).at[:, index].set(vector)
                    return operator(block)[:, index]
                return checked_linear_solve(action, value, config=config, converged=valid)[0]
            return jax.vmap(column, in_axes=(0, 1), out_axes=1)(jnp.arange(k), rhs)

        correction = attach_root(residual, jnp.zeros_like(x), config=config,
            tangent_solve=tangent_solve, converged=valid)
    else:
        correction = attach_root(residual, jnp.zeros(n*k, dtype=x.dtype),
                                 config=config, converged=valid)
    y = x + complement(correction.reshape(n, k))
    if isolated:
        return y/jnp.sqrt(jnp.sum(y*y, axis=0, keepdims=True))
    gram = y.T @ y
    lower = jnp.linalg.cholesky(.5*(gram+gram.T))
    return solve_triangular(lower, y.T, lower=True).T


@partial(jax.custom_jvp, nondiff_argnums=(0, 3, 5))
def _frame(operator, parameters, reference, config, valid, isolated):
    return reference


@_frame.defjvp
def _frame_jvp(operator, config, isolated, primals, tangents):
    parameters, reference, valid = primals
    dparameters, _, _ = tangents
    attached = lambda p: _implicit_frame(
        # Apply the derivative guard after closure conversion. Keeping valid
        # explicit avoids a captured bool tracer, while this inner guard also
        # rejects VJPs through an invalid root whose physical tangent is zero.
        lambda z: require_converged_derivative(operator(z, *p), valid),
        reference, config=config, valid=valid, isolated=isolated)
    tangent = jax.jvp(attached, (parameters,), (dparameters,))[1]
    return _frame(operator, parameters, reference, config, valid, isolated), tangent


@partial(jax.custom_jvp, nondiff_argnums=(0, 3, 5))
def _values(operator, parameters, reference, config, valid, isolated):
    return jnp.sum(reference*operator(reference, *parameters), axis=0)


@_values.defjvp
def _values_jvp(operator, config, isolated, primals, tangents):
    parameters, reference, valid = primals
    dparameters, _, _ = tangents
    y = _frame(operator, parameters, reference, config, valid, isolated)
    # Hold y fixed in this partial JVP; outer derivatives retain its response.
    day = jax.jvp(lambda p: operator(y, *p), (parameters,), (dparameters,))[1]
    return _values(operator, parameters, reference, config, valid, isolated), jnp.sum(y*day, axis=0)


def _spectral_inputs(apply, basis):
    """Hoist physical inputs and retain a stopped, explicit numerical frame.

    A numerical forward or first energy derivative needs no response solve.
    Module-level rules avoid capturing a vmapped frame in a custom JVP that
    an outer derivative can invoke after the original batch/JIT scope ends.
    """
    x = jax.lax.stop_gradient(basis)
    operator, parameters = jax.closure_convert(apply, x)
    return operator, tuple(parameters), x


def attach_spectral_response(apply, basis, *, config, valid=True, isolated=False):
    """Normalize isolated columns or a complete invariant-subspace frame."""
    operator, parameters, x = _spectral_inputs(apply, basis)
    return _frame(operator, parameters, x, config, valid, isolated)


def spectral_values(apply, basis, *, config, valid=True, isolated=False):
    """Stationary eigenvalue differential, with response retained at higher order.

    ``isolated=True`` supplies separate isolated-root energies. For a complete
    cluster, only their sum is a basis-invariant energy observable.
    """
    operator, parameters, x = _spectral_inputs(apply, basis)
    return _values(operator, parameters, x, config, valid, isolated)
