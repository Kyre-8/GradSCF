"""Invariant real spectral clusters via coupled implicit block equations.

The anchored charts AR=RT, L0.T R=I and A.T L=LU, R0.T L=I are
square systems of size n*k+k*k. Their Jacobians depend on separation from the
complement, never on a gap within the selected cluster. The live overlap inverse
in R solve(L.T R, L.T probes) is essential at second and higher orders.
"""
import jax
import jax.numpy as jnp

from ..diagnostics import require_converged_derivative
from ..nonlinear.root import attach_root
from ..memory import apply_columns, plan_linear
from ..types import NonHermitianResult, NonHermitianSubspaceResult
from gradscf._memory import (array_storage_bytes, jaxpr_temporary_bytes,
                             validate_reserved_bytes, workspace_bytes)


def full_subspace(operator, probes, *, block_size, max_memory=None,
                  reserved_bytes=0, operator_min_bytes=0,
                  operator_bytes_per_column=0, action_columns=None):
    """Full-span invariants without an eigenvector chart or dense A storage."""
    n, item = operator.shape[0], jnp.dtype(operator.dtype).itemsize
    block = min(n, block_size, n if action_columns is None else action_columns)
    if max_memory is not None:
        resident = reserved_bytes+array_storage_bytes(probes)
        available = workspace_bytes(max_memory, resident_bytes=resident)
        fixed = validate_reserved_bytes(operator_min_bytes)
        per = validate_reserved_bytes(operator_bytes_per_column)+4*n*item
        block = min(block, max(0, (available-fixed)//max(1, per)))
        if block < 1:
            raise ValueError('Full-space trace action exceeds the memory budget')

    def body(index, state):
        ids = index*block+jnp.arange(block)
        vectors = jax.nn.one_hot(ids, n, dtype=operator.dtype).T
        applied = apply_columns(operator, vectors, block)
        diagonal = applied[jnp.minimum(ids, n-1), jnp.arange(block)]
        trace, finite = state
        return (trace+jnp.sum(jnp.where(ids<n, diagonal, 0.)),
                finite & jnp.all(jnp.isfinite(applied)))

    trace, valid = jax.lax.fori_loop(0, (n+block-1)//block, body,
        (jnp.asarray(0., operator.dtype), jnp.asarray(True)))
    empty = jnp.zeros((0,), operator.dtype)
    frames = jnp.zeros((n, 0), operator.dtype)
    raw = NonHermitianResult(empty, frames, frames, empty, empty,
        jnp.zeros((0,), bool), jnp.zeros((0,), bool), empty,
        jnp.asarray(0., operator.dtype), empty.astype(jnp.result_type(operator.dtype, 1j)),
        jnp.asarray(False), jnp.asarray(0), jnp.asarray(0), empty, empty, jnp.asarray(0))
    projection = None if probes is None else require_converged_derivative(probes, valid)
    return NonHermitianSubspaceResult(projection, require_converged_derivative(trace, valid),
        jax.lax.stop_gradient(valid), jnp.asarray(jnp.inf, operator.dtype), raw)


def attach_subspace_response(operator, raw, probes, *, config, response,
                            numerical_valid, max_memory=None, reserved_bytes=0,
                            operator_min_bytes=0, operator_bytes_per_column=0,
                            action_columns=None):
    """Attach one joint right root and, only for projection, one joint left root."""
    r, l = raw.right_vectors, raw.left_vectors
    n, k = r.shape
    size = n*k+k*k
    linear = response.linear_config
    if linear.method == 'direct' and size > linear.max_dense:
        raise ValueError('Non-Hermitian coupled subspace response exceeds max_dense (n*k+k*k)')
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    qr, _ = jnp.linalg.qr(jnp.where(jnp.isfinite(r), r, 0.))
    ql, _ = jnp.linalg.qr(jnp.where(jnp.isfinite(l), l, 0.))
    sr, sl = jnp.linalg.svd(r, compute_uv=False), jnp.linalg.svd(l, compute_uv=False)
    gram = ql.T@qr
    singular = jnp.linalg.svd(gram, compute_uv=False)
    eps = 64*jnp.finfo(r.dtype).eps
    rank = ((sr[-1] > eps*jnp.maximum(1., sr[0])) &
            (sl[-1] > eps*jnp.maximum(1., sl[0])) &
            jnp.all(jnp.isfinite(singular)) & (singular[-1] > eps))
    condition = 1./jnp.maximum(singular[-1], jnp.finfo(r.dtype).tiny)
    left_anchor = ql@jnp.linalg.solve(jnp.where(rank, gram.T, jnp.eye(k, dtype=r.dtype)), jnp.eye(k, dtype=r.dtype))
    right_anchor, left_anchor = jax.lax.stop_gradient((qr, left_anchor))
    selected, outside = raw.raw_eigenvalues[:k], raw.raw_eigenvalues[k:]
    present = jnp.arange(k, raw.raw_eigenvalues.size) < raw.subspace_dimension
    distance = jnp.abs(selected[:, None]-outside[None, :])
    boundary = jnp.min(jnp.where(present[None, :], distance, jnp.inf))
    threshold = (response.gap_atol+response.gap_rtol*jnp.maximum(
        jnp.abs(selected[:, None]), jnp.abs(outside[None, :])))
    threshold += 2*condition*jnp.maximum(jnp.max(raw.residual_norms),
                                       jnp.max(raw.left_residual_norms))
    separated = jnp.any(present) & jnp.all((distance>threshold) | ~present[None, :])
    valid = jax.lax.stop_gradient(rank & (condition<config.max_condition) & separated &
        numerical_valid & jnp.all(raw.converged) &
        jnp.all(jnp.abs(selected.imag)<=config.imaginary_tol))

    def pack(frame, block):
        return jnp.concatenate((frame.reshape(-1), block.reshape(-1)))

    def unpack(state):
        return state[:n*k].reshape(n, k), state[n*k:].reshape(k, k)

    def residual(op, anchor):
        def equation(state):
            frame, block = unpack(state)
            action = require_converged_derivative(apply_columns(op, frame, action_columns), valid)
            out = pack(action-frame@block, anchor.T@frame-jnp.eye(k, dtype=frame.dtype))
            return jnp.where(valid, out, state)
        return equation

    def project(right, left, probe):
        return right@jnp.linalg.solve(left.T@right, left.T@probe)

    # Plan outputs and their active first JVP/VJP without materializing P.
    # Arbitrary outer derivative tapes remain outside this array budget.
    observable_bytes = 0
    if probes is not None and max_memory is not None:
        frame = jax.ShapeDtypeStruct((n, k), r.dtype)
        probe = jax.ShapeDtypeStruct(probes.shape, probes.dtype)
        def observable(r, l, p, dr, dl, dp):
            primal, tangent = jax.jvp(project, (r, l, p), (dr, dl, dp))
            _, pullback = jax.vjp(project, r, l, p)
            return primal, tangent, pullback(dp)
        observable_bytes = jaxpr_temporary_bytes(jax.make_jaxpr(observable)(
            frame, frame, probe, frame, frame, probe))

    def phase(equation, seed, *held):
        if max_memory is None:
            return linear, {}
        resident = reserved_bytes+array_storage_bytes(r, l, qr, ql, sr, sl, gram,
            singular, right_anchor, left_anchor, probes, seed, *held)+3*size*seed.dtype.itemsize
        resident += observable_bytes
        width = size if linear.method == 'direct' else 1
        columns = k if action_columns is None else min(k, action_columns)
        opaque = width*validate_reserved_bytes(operator_min_bytes)
        physical = width*columns*validate_reserved_bytes(operator_bytes_per_column)
        if opaque+physical > workspace_bytes(max_memory, resident_bytes=resident):
            raise ValueError('Subspace response physical minimum exceeds the memory budget')

        def payload(point, vector, transpose):
            action = lambda v: jax.jvp(equation, (point,), (v,))[1]
            if transpose:
                original = action
                action = lambda v: jax.linear_transpose(original, point)(v)[0]
            return action(vector) if vector.ndim == 1 else jax.vmap(action, in_axes=1, out_axes=1)(vector)

        shape = (size, size) if linear.method == 'direct' else (size,)
        vector = jax.ShapeDtypeStruct(shape, seed.dtype)
        scratch = opaque+max(physical, sum(jaxpr_temporary_bytes(jax.make_jaxpr(
            lambda point, v: payload(point, v, transpose))(seed, vector))
            for transpose in (False, True)))
        planned, _ = plan_linear(size, seed.dtype, linear, max_memory,
            resident_bytes=resident+array_storage_bytes(seed), operator_min_bytes=scratch)
        return planned, dict(max_memory=max_memory, reserved_bytes=resident,
                             operator_min_bytes=scratch)

    right_equation = residual(operator, left_anchor)
    seed_shape = jax.ShapeDtypeStruct((size,), r.dtype)
    right_config, right_budget = phase(right_equation, seed_shape)
    if probes is not None:
        left_equation = residual(operator.T, right_anchor)
        # Both phases must fit before even the numerical seed action. The
        # prospective right root and its output views remain during the left.
        left_shape = jax.ShapeDtypeStruct((size,), r.dtype)
        right_shape = jax.ShapeDtypeStruct((size,), r.dtype)
        left_config, left_budget = phase(left_equation, left_shape, seed_shape,
            right_shape, jax.ShapeDtypeStruct((n, k), r.dtype),
            jax.ShapeDtypeStruct((k, k), r.dtype), jax.ShapeDtypeStruct((), r.dtype))
    def checked_chart():
        # A small per-eigenvector residual need not survive a nearly singular
        # change of basis. Certify the actual block root after QR, not just the
        # original Ritz columns; discard these temporary actions before AD.
        applied = apply_columns(operator, right_anchor, action_columns)
        block = left_anchor.T@applied
        error = jnp.max(jnp.linalg.norm(applied-right_anchor@block, axis=0))
        if probes is not None:
            transposed = apply_columns(operator.T, left_anchor, action_columns)
            left_error = jnp.linalg.norm(transposed-left_anchor@block.T, axis=0)
            left_error /= jnp.maximum(1., jnp.linalg.norm(left_anchor, axis=0))
            error = jnp.maximum(error, jnp.max(left_error))
        gauge = jnp.linalg.norm(left_anchor.T@right_anchor-jnp.eye(k, dtype=r.dtype))
        return jax.lax.stop_gradient((block, error, gauge))

    block, chart_error, gauge_error = checked_chart()
    chart_threshold = (response.gap_atol+response.gap_rtol*jnp.maximum(
        jnp.abs(selected[:, None]), jnp.abs(outside[None, :])))+2*condition*chart_error
    valid = jax.lax.stop_gradient(valid & jnp.isfinite(chart_error) &
        (chart_error<=config.atol) & (gauge_error<=config.atol) &
        jnp.all((distance>chart_threshold) | ~present[None, :]))
    seed = pack(right_anchor, block)
    right_state = attach_root(right_equation, seed, config=right_config,
                             converged=valid, **right_budget)
    right, reduced = unpack(right_state)
    trace = require_converged_derivative(jnp.trace(reduced), valid)
    projection = None
    if probes is not None:
        left_seed = pack(left_anchor, block.T)
        left_state = attach_root(left_equation, left_seed, config=left_config,
                                 converged=valid, **left_budget)
        left, _ = unpack(left_state)
        projection = project(right, left, probes)
        projection = require_converged_derivative(projection, valid)
    return NonHermitianSubspaceResult(projection, trace, valid,
        jax.lax.stop_gradient(boundary), jax.lax.stop_gradient(raw))
