# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# Adapted from PySCF 2.9.0 cc/ccsd_t_slow.py; see NOTICE.md and LICENSE.pyscf.
# Original author: Qiming Sun <osirpt.sun@gmail.com>
# QCISD(T) weight/contractions: PySCF qcisd_t_slow, Qiming Sun and Timothy Berkelbach.
"""CC/QCI triples: canonical streaming, general tensor and panel response."""
from math import isfinite
from itertools import permutations
import numpy as np
import jax
import jax.numpy as jnp
from .integrals import prepare_integrals
from .ground import METHODS
from .amplitudes import AmplitudeSpace
from .rccsd import residual
from .types import TriplesResult


def _r3(w):
    return (
        4 * w
        + w.transpose(1, 2, 0)
        + w.transpose(2, 0, 1)
        - 2 * w.transpose(2, 1, 0)
        - 2 * w.transpose(0, 2, 1)
        - 2 * w.transpose(1, 0, 2)
    )


def _check_options(orbital_basis, algorithm, max_workspace_elements, variant):
    if orbital_basis not in {"canonical", "semicanonical"}:
        raise ValueError("orbital_basis must be 'canonical' or 'semicanonical'")
    if algorithm not in {"auto", "streamed"}:
        raise ValueError("algorithm must be 'auto' or 'streamed'")
    if not isinstance(max_workspace_elements, int) or max_workspace_elements < 1:
        raise ValueError("max_workspace_elements must be a positive integer")
    if algorithm == "streamed" and variant != "ccsd(t)":
        raise NotImplementedError("Streamed full-Fock response currently implements only CCSD(T)")
    if orbital_basis == "semicanonical" and variant != "ccsd(t)":
        raise NotImplementedError("Semicanonical triples currently implement only CCSD(T)")


def _restricted_model(variant, denominator_tol, residual_tol, canonical_tol,
                      max_virtual_triples, orbital_basis, algorithm, max_triples_elements):
    variants = ("ccsd+t(ccsd)", "ccsd(t)", "qcisd(t)")
    if variant not in variants:
        raise ValueError(f"Unknown triples variant {variant!r}; choose {variants}")
    for value in (denominator_tol, residual_tol, canonical_tol):
        if not isfinite(value) or value <= 0:
            raise ValueError("Triples tolerances must be finite and positive")
    if not isinstance(max_virtual_triples, int) or max_virtual_triples < 1:
        raise ValueError("max_virtual_triples must be a positive integer")
    if orbital_basis == "semicanonical" and algorithm == "auto":
        if not isinstance(max_triples_elements, int) or max_triples_elements < 1:
            raise ValueError("max_triples_elements must be a positive integer")
    return "qcisd" if variant == "qcisd(t)" else "ccsd"


def evaluate_triples(
    h1,
    eri,
    result,
    *,
    nocc,
    frozen=None,
    denominator_tol=1e-10,
    max_virtual_triples=20000,
    variant="ccsd(t)",
    residual_tol=1e-8,
    canonical_tol=1e-7,
    orbital_basis="canonical",
    max_triples_elements=2_000_000,
    algorithm="auto",
    max_workspace_elements=2_000_000,
):
    """CCSD(T), restricted Urban +T(CCSD), or canonical restricted QCISD(T).

    Urban et al. (1985), doi:10.1063/1.449067, retains only the connected
    WT2 contribution. Raghavachari et al. (1989),
    doi:10.1016/S0009-2614(89)87395-6, adds the singles-dependent term.
    Definitions were cross-checked against OpenMolcas CCT3; no Fortran was copied.
    QCISD(T) requires QCI amplitudes/residuals and the QCI singles weight.
    result must retain its selected model's amplitude response for derivatives.
    No shifts are applied to the physical triples denominators.
    orbital_basis='semicanonical' selects the general-reference formula,
    including F_vo*T2, through the common tensor-sum inverse. That bounded
    reference path stores (nocc_spin*nvir_spin)**3 elements per triples tensor;
    it is not the canonical streaming algorithm. algorithm='streamed' instead
    opts into bounded panels with full Fock-block JVP/VJP and second response;
    third-order AD is explicitly rejected. max_workspace_elements limits one
    spin-orbital panel, not total memory. See SEMICANONICAL.md.
    """
    _check_options(orbital_basis, algorithm, max_workspace_elements, variant)
    if (orbital_basis == "semicanonical" and algorithm == "auto"
            and not isinstance(nocc, (tuple, list))):
        # Preserve the legacy raw Hamiltonian chart, including independent
        # alpha/beta Fock responses off the spatial ERI permutation manifold.
        same_spin = result.t2-result.t2.swapaxes(2, 3)
        result = result._replace(t1=(result.t1, result.t1),
                                  t2=(same_spin, result.t2, same_spin))
        h1, eri, nocc = (h1, h1), (eri, eri, eri), (nocc, nocc)
    if isinstance(nocc, (tuple, list)):
        from ._spin_triples import evaluate_ucc_triples
        return evaluate_ucc_triples(h1, eri, result, nocc=nocc, frozen=frozen,
            denominator_tol=denominator_tol, max_virtual_triples=max_virtual_triples,
            variant=variant, residual_tol=residual_tol, canonical_tol=canonical_tol,
            orbital_basis=orbital_basis, max_triples_elements=max_triples_elements,
            algorithm=algorithm, max_workspace_elements=max_workspace_elements)
    _restricted_model(variant, denominator_tol, residual_tol, canonical_tol,
                      max_virtual_triples, orbital_basis, algorithm, max_triples_elements)
    ints = prepare_integrals(h1, eri, nocc=nocc, frozen=frozen)
    return evaluate_triples_from_integrals(ints, result,
        denominator_tol=denominator_tol, max_virtual_triples=max_virtual_triples,
        variant=variant, residual_tol=residual_tol, canonical_tol=canonical_tol,
        orbital_basis=orbital_basis, max_triples_elements=max_triples_elements,
        algorithm=algorithm, max_workspace_elements=max_workspace_elements)


def evaluate_triples_from_integrals(
    ints, result, *, denominator_tol=1e-10, max_virtual_triples=20000,
    variant="ccsd(t)", residual_tol=1e-8, canonical_tol=1e-7,
    orbital_basis="canonical", max_triples_elements=2_000_000,
    algorithm="auto", max_workspace_elements=2_000_000,
):
    """Restricted triples from prepared blocks with the original kernel choice.

    Canonical auto uses the spatial streaming formula. Semicanonical auto
    retains the bounded six-index tensor reference, using only the existing
    required spin-block bridge; it does not prepare full UCC integral blocks.
    Explicit streamed panels retain their workspace and second-order AD guard.
    Semicanonical auto assumes a physically consistent restricted snapshot and
    pair-symmetric T2. Its supplied single-Fock chart is independent of the raw
    Hamiltonian API's alpha/beta responses off that physical manifold.
    """
    _check_options(orbital_basis, algorithm, max_workspace_elements, variant)
    variants = ("ccsd+t(ccsd)", "ccsd(t)", "qcisd(t)")
    model = _restricted_model(variant, denominator_tol, residual_tol, canonical_tol,
        max_virtual_triples, orbital_basis, algorithm, max_triples_elements)
    no, nv = ints.nocc, ints.nvir
    space = AmplitudeSpace(no, nv)
    space.pack(result.t1, result.t2)  # Validate before any empty-space shortcut.
    r1, r2 = residual(result.t1, result.t2, ints, model=model)
    cc_residual = space.pack(r1, r2)
    general = orbital_basis == "semicanonical" and algorithm == "auto"
    if general:
        # Match the old restricted-to-UCC reference's residual coordinate norm.
        from .spin_amplitudes import SpinAmplitudeSpace
        spin_space = SpinAmplitudeSpace((no, no), (nv, nv))
        same = r2-r2.swapaxes(2, 3)
        cc_residual = spin_space.pack(*spin_space.from_blocks(
            (r1, r1), (same, r2, same), ints.fock.dtype))
    cc_norm = jnp.max(jnp.abs(cc_residual), initial=0.0)
    canonical_error = jnp.max(
        jnp.abs(ints.fock - jnp.diag(jnp.diag(ints.fock) if general else ints.mo_energy)), initial=0.0
    )
    symmetry_error = jnp.max(
        jnp.abs(result.t2 - result.t2.transpose(1, 0, 3, 2)), initial=0.0
    )
    if general:
        chart_tol = (64*jnp.finfo(result.t2.dtype).eps
                     * jnp.maximum(1., jnp.max(jnp.abs(result.t2), initial=0.)))
        # The restricted residual-to-spin norm is undefined outside this
        # chart; never report it as the legacy UCC residual of a damaged state.
        cc_norm = jnp.where(symmetry_error <= chart_tol, cc_norm, jnp.nan)
        same = result.t2-result.t2.swapaxes(2, 3)
        symmetry_error = jnp.maximum(
            jnp.max(jnp.abs(same+same.swapaxes(0, 1)), initial=0.),
            jnp.max(jnp.abs(same+same.swapaxes(2, 3)), initial=0.))
    state_valid = (
        result.converged
        & (result.method_id == METHODS.index(model))
        & (cc_norm <= residual_tol)
        & (symmetry_error <= residual_tol)
    )
    if orbital_basis == "canonical":
        state_valid &= canonical_error <= canonical_tol
    dtype = jnp.result_type(ints.fock.dtype, result.t1.dtype, result.t2.dtype)

    def assemble(connected, singles, minimum):
        valid = (
            state_valid
            & (minimum > denominator_tol)
            & jnp.isfinite(connected + singles)
        )
        applied_singles = singles if variant != "ccsd+t(ccsd)" else jnp.zeros_like(singles)
        correction = connected + applied_singles
        def checked(value):
            if algorithm == "streamed":
                return value * jnp.where(valid, jnp.ones_like(value), jnp.full_like(value, jnp.nan))
            return jnp.where(valid, value, jnp.nan)
        return TriplesResult(
            checked(correction),
            checked(connected),
            checked(applied_singles),
            minimum,
            cc_norm,
            canonical_error,
            valid,
            jnp.asarray(variants.index(variant), dtype=jnp.int32 if general else None),
        )

    if no < 2 or nv < 2:
        zero = jnp.asarray(0.0, dtype=dtype)
        return assemble(zero, zero, jnp.asarray(jnp.inf, dtype=dtype))
    if algorithm == "streamed" or general:
        from ._streamed_triples import _restricted_inputs, _streamed_general_correction
        if algorithm == "streamed" and (2 * nv) * (2 * no)**3 > max_workspace_elements:
            raise ValueError("Restricted spin triples panel exceeds max_workspace_elements")
        if general and ((2 * no) * (2 * nv))**3 > max_triples_elements:
            raise ValueError("Semicanonical triples exceed max_triples_elements; reduce the active space")
        spin_ints, spin_t1, spin_t2 = _restricted_inputs(ints, result.t1, result.t2)
        if general:
            from ._spin_triples import _general_correction
            kernel = _general_correction
        else:
            kernel = _streamed_general_correction
        connected, singles, minimum, solved = kernel(
            spin_ints, spin_t1, spin_t2, denominator_tol=denominator_tol)
        state_valid &= solved
        return assemble(connected, singles, minimum)
    if nv * (nv + 1) * (nv + 2) // 6 > max_virtual_triples:
        raise ValueError("Virtual triples exceed max_virtual_triples")
    combinations = np.asarray(
        [(a, b, c) for a in range(nv) for b in range(a + 1) for c in range(b + 1)],
        dtype=np.int32,
    )
    combos = jnp.asarray(combinations)
    perms = list(permutations(range(3)))
    indices = jnp.asarray(perms)
    t1 = result.t1.T
    t2 = result.t2.transpose(2, 3, 0, 1)
    vvov = ints.ovvv.transpose(1, 3, 0, 2)
    vooo = ints.ovoo.transpose(1, 0, 2, 3)
    vvoo = ints.ovov.transpose(1, 3, 0, 2)
    eo, ev = ints.mo_energy[:no], ints.mo_energy[no:]
    eijk = eo[:, None, None] + eo[None, :, None] + eo[None, None, :]

    def wv(abc):
        a, b, c = abc
        w = jnp.einsum(
            "if,fkj->ijk", vvov[a, b], t2[c], precision="highest"
        ) - jnp.einsum("ijm,mk->ijk", vooo[a], t2[b, c], precision="highest")
        # Preserve canonical CCSD's convention. QCI uses the full upstream V
        # contraction, with F_vo*T2 normally vanishing for canonical RHF.
        v = jnp.einsum("ij,k->ijk", vvoo[a, b], t1[c])
        if model == "qcisd":
            v += jnp.einsum("ij,k->ijk", t2[a, b], ints.fock[no+c, :no])
        return w, v

    def body(index, carry):
        connected, singles, minimum = carry
        abc = combos[index]
        a, b, c = abc
        d = eijk - ev[a] - ev[b] - ev[c]
        minimum = jnp.minimum(minimum, jnp.min(jnp.abs(d)))
        multiplicity = jnp.where(a == c, 6.0, jnp.where((a == b) | (b == c), 2.0, 1.0))
        denominator = jnp.where(jnp.abs(d) > denominator_tol, d, 1.0) * multiplicity
        ws, vs = jax.vmap(wv)(abc[indices])
        zw = jax.vmap(_r3)(ws) / denominator[None, ...]
        singles_weight = 1.0 if model == "qcisd" else 0.5
        zv = jax.vmap(_r3)(singles_weight * vs) / denominator[None, ...]
        value_w = jnp.asarray(0.0, dtype=dtype)
        value_v = jnp.asarray(0.0, dtype=dtype)
        for iq, q in enumerate(perms):
            for p in perms:
                composed = tuple(q[i] for i in p)
                iw = perms.index(composed)
                inverse = tuple(np.argsort(p))
                value_w = value_w + jnp.sum(ws[iw].transpose(inverse) * zw[iq])
                value_v = value_v + jnp.sum(ws[iw].transpose(inverse) * zv[iq])
        return connected + 2 * value_w, singles + 2 * value_v, minimum

    connected, singles, minimum = jax.lax.fori_loop(
        0,
        len(combinations),
        body,
        (
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(jnp.inf, dtype=dtype),
        ),
    )
    return assemble(connected, singles, minimum)


def triples_correction(
    h1,
    eri,
    result,
    *,
    nocc,
    frozen=None,
    denominator_tol=1e-10,
    max_virtual_triples=20000,
    variant="ccsd(t)",
    residual_tol=1e-8,
    canonical_tol=1e-7,
    orbital_basis="canonical",
    max_triples_elements=2_000_000,
    algorithm="auto",
    max_workspace_elements=2_000_000,
):
    """Scalar correction; use evaluate_triples for decomposition and diagnostics."""
    return evaluate_triples(
        h1,
        eri,
        result,
        nocc=nocc,
        frozen=frozen,
        denominator_tol=denominator_tol,
        max_virtual_triples=max_virtual_triples,
        variant=variant,
        residual_tol=residual_tol,
        canonical_tol=canonical_tol,
        orbital_basis=orbital_basis,
        max_triples_elements=max_triples_elements,
        algorithm=algorithm,
        max_workspace_elements=max_workspace_elements,
    ).energy


def triples_correction_from_integrals(ints, result, **kwargs):
    """Scalar prepared correction; see evaluate_triples_from_integrals."""
    return evaluate_triples_from_integrals(ints, result, **kwargs).energy
