# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Real UCCSD(T), adapted from PySCF 2.9.0 gccsd_t and gccsd_t_slow.

Spin-orbital formula: J. Chem. Phys. 98, 8718 (1993), doi:10.1063/1.464480.
Canonical virtual triples are streamed. The opt-in semicanonical reference
path instead materializes bounded six-index moments and uses a tensor solve.
algorithm='streamed' selects four-index panels with response through second order.
See NOTICE.md for software attribution; no PySCF runtime is used.
"""
from itertools import combinations
from math import isfinite

import jax
import jax.numpy as jnp

from .uccsd import prepare_ucc_integrals
from .spin_amplitudes import SpinAmplitudeSpace
from ._spin_equations import residual
from .ground import METHODS
from .types import TriplesResult
from ..solvers import solve_tensor_sum
from ._streamed_triples import _streamed_general_correction


def _general_correction(ints, t1, t2, *, denominator_tol):
    """Basis-invariant semicanonical formula evaluated in the original MO frame.

    W and V are the fully antisymmetrized connected and disconnected moments.
    D X = W; E_connected = W.X/36, E_disconnected = V.X/36. The latter includes
    both T1*V and F_vo*T2, as required for a non-HF determinant (Watts 1993).
    """
    no, _ = t1.shape
    bcei = ints.ovvv.transpose(3, 2, 1, 0)
    majk = ints.ooov.transpose(2, 3, 0, 1)

    def antisymmetrize(value):
        value = value-value.transpose(0, 1, 2, 4, 3, 5)-value.transpose(0, 1, 2, 5, 4, 3)
        return value-value.transpose(1, 0, 2, 3, 4, 5)-value.transpose(2, 1, 0, 3, 4, 5)

    w = antisymmetrize(jnp.einsum("jkae,bcei->ijkabc", t2, bcei)
                       -jnp.einsum("imbc,majk->ijkabc", t2, majk))
    v = antisymmetrize(jnp.einsum("ia,jkbc->ijkabc", t1, ints.oovv)
                       +jnp.einsum("ai,jkbc->ijkabc", ints.fock[no:, :no], t2))
    foo, fvv = ints.fock[:no, :no], ints.fock[no:, no:]
    solved = solve_tensor_sum((foo, foo, foo, -fvv, -fvv, -fvv), w,
                               denominator_tol=denominator_tol)
    return (jnp.sum(w*solved.solution)/36, jnp.sum(v*solved.solution)/36,
            solved.min_abs_denominator, solved.converged)


def evaluate_ucc_triples(h1, eri, result, *, nocc, frozen=None,
                         denominator_tol=1e-10, max_virtual_triples=20000,
                         variant="ccsd(t)", residual_tol=1e-8, canonical_tol=1e-7,
                         orbital_basis="canonical", max_triples_elements=2_000_000,
                         algorithm="auto", max_workspace_elements=2_000_000):
    if variant != "ccsd(t)":
        raise NotImplementedError("Unrestricted triples currently implement only CCSD(T)")
    if any(not isfinite(x) or x <= 0 for x in (denominator_tol, residual_tol, canonical_tol)):
        raise ValueError("Triples tolerances must be finite and positive")
    if not isinstance(max_virtual_triples, int) or max_virtual_triples < 1:
        raise ValueError("max_virtual_triples must be a positive integer")
    if orbital_basis not in {"canonical", "semicanonical"}:
        raise ValueError("orbital_basis must be 'canonical' or 'semicanonical'")
    if not isinstance(max_triples_elements, int) or max_triples_elements < 1:
        raise ValueError("max_triples_elements must be a positive integer")
    if algorithm not in {"auto", "streamed"}:
        raise ValueError("algorithm must be 'auto' or 'streamed'")
    if not isinstance(max_workspace_elements, int) or max_workspace_elements < 1:
        raise ValueError("max_workspace_elements must be a positive integer")
    ints, occupied, virtual = prepare_ucc_integrals(h1, eri, nocc=nocc, frozen=frozen)
    space = SpinAmplitudeSpace(occupied, virtual)
    t1, t2 = space.from_blocks(result.t1, result.t2, ints.fock.dtype)
    symmetry_error = jnp.asarray(0., dtype=t1.dtype)
    for block in (result.t2[0], result.t2[2]):
        block = jnp.asarray(block)
        symmetry_error = jnp.maximum(symmetry_error, jnp.max(jnp.abs(block+block.swapaxes(0, 1)), initial=0.))
        symmetry_error = jnp.maximum(symmetry_error, jnp.max(jnp.abs(block+block.swapaxes(2, 3)), initial=0.))
    norm = jnp.max(jnp.abs(space.pack(*residual(t1, t2, ints))), initial=0.)
    eps = jnp.diag(ints.fock)
    canonical_error = jnp.max(jnp.abs(ints.fock-jnp.diag(eps)), initial=0.)
    state_valid = (result.converged & (result.method_id == METHODS.index("ccsd"))
                   & (norm <= residual_tol) & (symmetry_error <= residual_tol))
    if orbital_basis == "canonical":
        state_valid &= canonical_error <= canonical_tol

    def assemble(connected, singles, minimum):
        valid = state_valid & (minimum > denominator_tol) & jnp.isfinite(connected+singles)
        def checked(value):
            if algorithm == "streamed":
                return value * jnp.where(valid, jnp.ones_like(value), jnp.full_like(value, jnp.nan))
            return jnp.where(valid, value, jnp.nan)
        return TriplesResult(checked(connected+singles), checked(connected), checked(singles),
            minimum, norm, canonical_error, valid, jnp.asarray(1, dtype=jnp.int32))

    no, nv = t1.shape
    zero = jnp.asarray(0., dtype=t1.dtype)
    if no < 3 or nv < 3:
        return assemble(zero, zero, jnp.asarray(jnp.inf, dtype=t1.dtype))
    if algorithm == "streamed":
        if nv * no**3 > max_workspace_elements:
            raise ValueError("Streamed triples panel exceeds max_workspace_elements")
        connected, disconnected, minimum, solved = _streamed_general_correction(
            ints, t1, t2, denominator_tol=denominator_tol)
        state_valid &= solved
        return assemble(connected, disconnected, minimum)
    if orbital_basis == "semicanonical":
        if (no*nv)**3 > max_triples_elements:
            raise ValueError("Semicanonical triples exceed max_triples_elements; reduce the active space")
        connected, disconnected, minimum, solved = _general_correction(
            ints, t1, t2, denominator_tol=denominator_tol)
        state_valid &= solved
        return assemble(connected, disconnected, minimum)
    if nv*(nv-1)*(nv-2)//6 > max_virtual_triples:
        raise ValueError("Virtual triples exceed max_virtual_triples")
    combos = jnp.asarray([(a, b, c) for c, b, a in combinations(range(nv), 3)])
    so = jnp.asarray([0]*occupied[0]+[1]*occupied[1])
    sv = jnp.asarray([0]*virtual[0]+[1]*virtual[1])
    indices = jnp.arange(no)
    distinct = ((indices[:, None, None] != indices[None, :, None])
                & (indices[:, None, None] != indices[None, None, :])
                & (indices[None, :, None] != indices[None, None, :]))
    spin_sum = so[:, None, None]+so[None, :, None]+so[None, None, :]
    eo, ev = eps[:no], eps[no:]
    eijk = eo[:, None, None]+eo[None, :, None]+eo[None, None, :]
    bcei = ints.ovvv.transpose(3, 2, 1, 0)
    majk = ints.ooov.transpose(2, 3, 0, 1)
    bcjk = ints.oovv.transpose(2, 3, 0, 1)
    doubles, singles = t2.transpose(2, 3, 0, 1), t1.T
    fvo = ints.fock[no:, :no]

    def wv(a, b, c):
        w = (jnp.einsum("ejk,ei->ijk", doubles[a], bcei[b, c])
             -jnp.einsum("im,mjk->ijk", doubles[b, c], majk[:, a]))
        v = (jnp.einsum("i,jk->ijk", singles[a], bcjk[b, c])
             +jnp.einsum("i,jk->ijk", fvo[a], doubles[b, c]))
        return w+w.transpose(2, 0, 1)+w.transpose(1, 2, 0), w, v

    def body(index, carry):
        connected, disconnected, minimum = carry
        a, b, c = combos[index]
        w0, v0, s0 = wv(a, b, c)
        w1, v1, s1 = wv(c, a, b)
        w2, v2, s2 = wv(b, a, c)
        d = eijk-ev[a]-ev[b]-ev[c]
        allowed = distinct & (spin_sum == sv[a]+sv[b]+sv[c])
        minimum = jnp.minimum(minimum, jnp.min(jnp.where(allowed, jnp.abs(d), jnp.inf)))
        w = jnp.where(allowed, w0+w1-w2, 0.) / jnp.where(jnp.abs(d) > denominator_tol, d, 1.)
        return (connected+.5*jnp.sum(w*(v0+v1-v2)),
                disconnected+.5*jnp.sum(w*(s0+s1-s2)), minimum)

    return assemble(*jax.lax.fori_loop(0, len(combos), body,
                                      (zero, zero, jnp.asarray(jnp.inf, dtype=t1.dtype))))
