# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE.pyscf).
# Moments adapted from PySCF gccsd_t_slow; see NOTICE.md.
"""Triples response through second order with four-index moment panels.

Diagonalization supplies a numerical frame only. The derivative follows
D X = W, including the full occupied/virtual Fock matrix response. Neither
the forward loop nor its adjoint retains a six-index triples tensor.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..solvers import solve_tensor_sum
from ..solvers.diagnostics import require_converged_derivative


@jax.custom_jvp
def _second_order_context(values):
    return values


@_second_order_context.defjvp
def _reject_third_order(primals, tangents):
    raise NotImplementedError("Streamed triples currently support second-order AD only")


def _rotate_inputs(oo, vv, fock, ovvv, ooov, oovv, t1, t2):
    no = t1.shape[0]
    return (
        oo.T @ t1 @ vv,
        jnp.einsum("pi,qj,ra,sb,pqrs->ijab", oo, oo, vv, vv, t2),
        jnp.einsum("pi,qa,rb,sc,pqrs->iabc", oo, vv, vv, vv, ovvv),
        jnp.einsum("pi,qj,rk,sa,pqrs->ijka", oo, oo, oo, vv, ooov),
        jnp.einsum("pi,qj,ra,sb,pqrs->ijab", oo, oo, vv, vv, oovv),
        vv.T @ fock[no:, :no] @ oo,
    )


def _factor_inputs(fock, ovvv, ooov, oovv, t1, t2):
    no = t1.shape[0]
    eo, oo = jnp.linalg.eigh(fock[:no, :no])
    ev, vv = jnp.linalg.eigh(fock[no:, no:])
    return eo, ev, oo, vv, _rotate_inputs(oo, vv, fock, ovvv, ooov, oovv, t1, t2)


def _moment_panel(inputs, b, c):
    t1, t2, ovvv, ooov, oovv, fvo = inputs
    bcei = ovvv.transpose(3, 2, 1, 0)
    majk = ooov.transpose(2, 3, 0, 1)

    def base(a, b, c):
        w = (jnp.einsum("jke,ei->ijk", t2[:, :, a, :], bcei[b, c])
             - jnp.einsum("im,mjk->ijk", t2[:, :, b, c], majk[:, a]))
        v = (jnp.einsum("i,jk->ijk", t1[:, a], oovv[:, :, b, c])
             + jnp.einsum("i,jk->ijk", fvo[a], t2[:, :, b, c]))
        return w, v

    def virtual_antisymmetry(a):
        w, v = base(a, b, c)
        wb, vb = base(b, a, c)
        wc, vc = base(c, b, a)
        return w - wb - wc, v - vb - vc

    w, v = jax.vmap(virtual_antisymmetry)(jnp.arange(t1.shape[1]))
    occupied_antisymmetry = lambda a: a - a.swapaxes(1, 2) - a.swapaxes(1, 3)
    return occupied_antisymmetry(w), occupied_antisymmetry(v)


def _denominator(eo, ev, b, c, tolerance):
    eijk = eo[:, None, None] + eo[None, :, None] + eo[None, None, :]
    value = eijk[None, ...] - ev[:, None, None, None] - ev[b] - ev[c]
    return jnp.where(jnp.abs(value) > tolerance, value, 1.), jnp.min(jnp.abs(value))


def _forward(tolerance, *args):
    eo, ev, _, _, inputs = _factor_inputs(*args)
    nv = ev.shape[0]
    def body(index, carry):
        connected, singles, minimum = carry
        b, c = index // nv, index % nv
        w, v = _moment_panel(inputs, b, c)
        denominator, smallest = _denominator(eo, ev, b, c, tolerance)
        x = w / denominator
        return (connected + jnp.sum(w * x) / 36,
                singles + jnp.sum(v * x) / 36, jnp.minimum(minimum, smallest))
    zero = jnp.asarray(0., args[0].dtype)
    return jnp.stack(jax.lax.fori_loop(0, nv * nv, body, (zero, zero, zero + jnp.inf)))


def _panel_cotangents(inputs, b, c, x, y):
    _, pullback = jax.vjp(lambda a: _moment_panel(a, b, c), inputs)
    gc = pullback((2 * x / 36, jnp.zeros_like(y)))[0]
    gv = pullback((y / 36, x / 36))[0]
    # Each of the three occupied/virtual axes gives the same contraction.
    goo = -jnp.stack((jnp.einsum("aijk,aljk->il", x, x),
                      jnp.einsum("aijk,aljk->il", y, x))) / 12
    gvv = jnp.stack((jnp.einsum("aijk,bijk->ab", x, x),
                     jnp.einsum("aijk,bijk->ab", y, x))) / 12
    return gc, gv, goo, gvv


def _pullback_frame(oo, vv, args, gradients, goo, gvv):
    no = oo.shape[0]
    _, pullback = jax.vjp(lambda *a: _rotate_inputs(oo, vv, *a), *args)
    result = []
    for index, value in enumerate(gradients):
        g = list(pullback(value))
        g[0] = g[0].at[:no, :no].add(oo @ goo[index] @ oo.T)
        g[0] = g[0].at[no:, no:].add(vv @ gvv[index] @ vv.T)
        result.append(tuple(g))
    return tuple(result)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _value_and_gradients(tolerance, *args):
    eo, ev, oo, vv, inputs = _factor_inputs(*args)
    nv, no = len(ev), len(eo)
    zero = jnp.asarray(0., args[0].dtype)
    gradients = tuple(jnp.zeros_like(a) for a in inputs)
    initial = (zero, zero, zero + jnp.inf, gradients, gradients,
               jnp.zeros((2, no, no), zero.dtype), jnp.zeros((2, nv, nv), zero.dtype))

    def body(index, carry):
        connected, singles, minimum, gc, gv, goo, gvv = carry
        b, c = index // nv, index % nv
        w, v = _moment_panel(inputs, b, c)
        denominator, smallest = _denominator(eo, ev, b, c, tolerance)
        x, yv = w / denominator, v / denominator
        pc, pv, po, pvv = _panel_cotangents(inputs, b, c, x, yv)
        gc = tuple(a + b for a, b in zip(gc, pc))
        gv = tuple(a + b for a, b in zip(gv, pv))
        goo, gvv = goo + po, gvv + pvv
        return (connected + jnp.sum(w * x) / 36, singles + jnp.sum(v * x) / 36,
                jnp.minimum(minimum, smallest), gc, gv, goo, gvv)

    connected, singles, minimum, gc, gv, goo, gvv = jax.lax.fori_loop(
        0, nv * nv, body, initial)
    gradients = _pullback_frame(oo, vv, args, (gc, gv), goo, gvv)
    return jnp.stack((connected, singles, minimum)), gradients


def _local_fock_action(foo, fvv, value):
    """Four of the six tensor-sum axes stay within the current panel."""
    return (jnp.einsum("il,aljk->aijk", foo, value)
            + jnp.einsum("jl,ailk->aijk", foo, value)
            + jnp.einsum("kl,aijl->aijk", foo, value)
            - jnp.einsum("ab,bijk->aijk", fvv, value))


def _hessian_product(tolerance, primals, tangents):
    no = primals[-2].shape[0]
    fock, dfock = primals[0], tangents[0]
    # A fixed numerical frame is safe only because every axis of dot(D) below
    # enters dot(X) = D^-1 (dot(W) - dot(D) X). No eigenvector AD is used.
    eo, oo = jnp.linalg.eigh(jax.lax.stop_gradient(fock[:no, :no]))
    ev, vv = jnp.linalg.eigh(jax.lax.stop_gradient(fock[no:, no:]))
    inputs = _rotate_inputs(oo, vv, *primals)
    dinputs = _rotate_inputs(oo, vv, *tangents)
    dfoo = oo.T @ dfock[:no, :no] @ oo
    dfvv = vv.T @ dfock[no:, no:] @ vv
    nv = len(ev)
    gradients = tuple(jnp.zeros_like(a) for a in inputs)
    initial = (gradients, gradients,
               jnp.zeros((2, no, no), fock.dtype), jnp.zeros((2, nv, nv), fock.dtype))
    occupied = jnp.diag(eo)

    def body(index, carry):
        b, c = index // nv, index % nv
        (w, v), (dw, dv) = jax.jvp(
            lambda a: _moment_panel(a, b, c), (inputs,), (dinputs,))
        denominator, _ = _denominator(eo, ev, b, c, tolerance)
        x, y = w / denominator, v / denominator
        rhs = (dw - _local_fock_action(dfoo, dfvv, x),
               dv - _local_fock_action(dfoo, dfvv, y))

        def mix(d, state):
            # The fixed b,c axes mix with neighbouring panels. Complete
            # virtual antisymmetry gives X[:,b,d] = -X[:,d,b], also for Y.
            wc, vc = _moment_panel(inputs, d, c)
            wb, vb = _moment_panel(inputs, d, b)
            dc, _ = _denominator(eo, ev, d, c, tolerance)
            db, _ = _denominator(eo, ev, d, b, tolerance)
            return (state[0] + dfvv[b, d] * wc / dc - dfvv[c, d] * wb / db,
                    state[1] + dfvv[b, d] * vc / dc - dfvv[c, d] * vb / db)

        rhs = jax.lax.fori_loop(
            0, nv, jax.checkpoint(mix, prevent_cse=False), rhs)
        factors = (jnp.diag(-ev - ev[b] - ev[c]), occupied, occupied, occupied)
        dx, dy = (solve_tensor_sum(factors, r, denominator_tol=tolerance).solution
                  for r in rhs)
        _, response = jax.jvp(
            lambda a, u, z: _panel_cotangents(a, b, c, u, z),
            (inputs, x, y), (dinputs, dx, dy))
        gc, gv, goo, gvv = carry
        pc, pv, po, pvv = response
        return (tuple(a + b for a, b in zip(gc, pc)),
                tuple(a + b for a, b in zip(gv, pv)), goo + po, gvv + pvv)

    gc, gv, goo, gvv = jax.lax.fori_loop(
        0, nv * nv, jax.checkpoint(body, prevent_cse=False), initial)
    return _pullback_frame(oo, vv, primals, (gc, gv), goo, gvv)


def _contract_gradients(value, gradients, tangents):
    tangent = [sum(jnp.vdot(g, t) for g, t in zip(row, tangents))
               for row in gradients]
    return jnp.stack((*tangent, jnp.zeros_like(value[2])))


@_value_and_gradients.defjvp
def _value_and_gradients_jvp(tolerance, primals, tangents):
    primals = _second_order_context(primals)
    value, gradients = _value_and_gradients(tolerance, *primals)
    tangent = _contract_gradients(value, gradients, tangents)
    return (value, gradients), (tangent, _hessian_product(tolerance, primals, tangents))


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _components(tolerance, *args):
    return _forward(tolerance, *args)


@_components.defjvp
def _components_jvp(tolerance, primals, tangents):
    value, gradients = _value_and_gradients(tolerance, *primals)
    return value, _contract_gradients(value, gradients, tangents)


def _streamed_general_correction(ints, t1, t2, *, denominator_tol):
    args = (ints.fock, ints.ovvv, ints.ooov, ints.oovv, t1, t2)
    dtype = jnp.result_type(*args)
    # Promote before the custom rule so its work arrays share one dtype and
    # the cast's transpose returns cotangents in each original input dtype.
    value = _components(denominator_tol, *(a.astype(dtype) for a in args))
    no = t1.shape[0]
    valid = jnp.all(jnp.isfinite(value)) & (value[2] > denominator_tol)
    for block in (ints.fock[:no, :no], ints.fock[no:, no:]):
        threshold = 32 * jnp.finfo(block.dtype).eps * jnp.maximum(1., jnp.linalg.norm(block))
        valid &= jnp.linalg.norm(block - block.T) <= threshold
    components = require_converged_derivative(value[:2], valid)
    return components[0], components[1], value[2], valid


class _TriplesInputs(NamedTuple):
    fock: object
    ovvv: object
    ooov: object
    oovv: object


def _restricted_inputs(ints, t1, t2):
    """Expand only needed spin blocks; never build a full spin-MO ERI."""
    from .spin_amplitudes import SpinAmplitudeSpace
    no, nv = t1.shape
    def spin_block(coulomb, exchange):
        sizes = coulomb.shape
        orbital = tuple(jnp.arange(2 * n) % n for n in sizes)
        spin = tuple((jnp.arange(2 * n) // n).reshape(
            tuple(2 * n if k == axis else 1 for k in range(4)))
            for axis, n in enumerate(sizes))
        direct = (spin[0] == spin[2]) & (spin[1] == spin[3])
        crossed = (spin[0] == spin[3]) & (spin[1] == spin[2])
        return (coulomb[jnp.ix_(*orbital)] * direct
                - exchange[jnp.ix_(*orbital)] * crossed)
    ovvv = spin_block(ints.ovvv.transpose(0, 2, 1, 3), ints.ovvv.transpose(0, 2, 3, 1))
    ooov = spin_block(ints.ovoo.transpose(2, 0, 3, 1), ints.ovoo.transpose(0, 2, 3, 1))
    coulomb = ints.ovov.transpose(0, 2, 1, 3)
    oovv = spin_block(coulomb, coulomb.swapaxes(2, 3))
    orbital = jnp.concatenate((jnp.tile(jnp.arange(no), 2), jnp.tile(jnp.arange(nv) + no, 2)))
    spin = jnp.concatenate((jnp.repeat(jnp.arange(2), no), jnp.repeat(jnp.arange(2), nv)))
    fock = ints.fock[jnp.ix_(orbital, orbital)] * (spin[:, None] == spin[None, :])
    same_spin = t2 - t2.swapaxes(2, 3)
    t1, t2 = SpinAmplitudeSpace((no, no), (nv, nv)).from_blocks(
        (t1, t1), (same_spin, t2, same_spin), ints.fock.dtype)
    return _TriplesInputs(fock, ovvv, ooov, oovv), t1, t2
