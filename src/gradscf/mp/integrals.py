"""Canonical MP blocks; frozen cores still contribute to the HF Fock and energy."""
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp

from gradscf._memory import DEFAULT_MAX_MEMORY_MB, resolve_memory_mb
from gradscf.ao2mo import validate_integrals, validate_unrestricted_integrals, frozen_indices, unrestricted_frozen_indices, transform_eri_block
from gradscf.ao2mo.scf import prepare_scf_source


class MPIntegrals(NamedTuple):
    energies: tuple
    nocc: tuple
    blocks: object
    factors: object
    reference_energy: object
    canonical_error: object
    finite: object
    focks: tuple
    reference_error: object


def _active_orbitals(nmo, nocc, frozen):
    frozen = frozen_indices(nmo, nocc, frozen)
    return ([i for i in range(nocc) if i not in frozen],
            [a for a in range(nocc, nmo) if a not in frozen])


def active_indices(nmo, nocc, frozen):
    return tuple(jnp.asarray(x, dtype=jnp.int32) for x in _active_orbitals(nmo, nocc, frozen))


def canonical_error(focks):
    return jnp.max(jnp.stack([jnp.max(jnp.abs(f - jnp.diag(jnp.diag(f))), initial=0.)
                              for f in focks]))


def reference_error(focks, nocc, *, frozen=None, orbital_basis="canonical", frozen_gap_tol=1e-10):
    """Check the HF partition and fixed frozen subspaces, without choosing a gauge.

    Noncanonical occupied/virtual rotations must preserve the frozen/active
    boundary. Within each of these subspaces, Fock off-diagonal entries are
    unrestricted. Mixing an active subspace with its frozen complement is
    rejected rather than changing the MP zeroth-order Hamiltonian. Frozen
    boundaries must also preserve complete degenerate Fock eigenspaces; this
    mode does not attach response to independently supplied frozen projectors.
    """
    if orbital_basis == "canonical":
        return canonical_error(focks)
    ns = nocc if isinstance(nocc, tuple) else (nocc,)
    nmo = focks[0].shape[0]
    frozen_sets = (unrestricted_frozen_indices(nmo, ns, frozen)
                   if isinstance(nocc, tuple) else (frozen_indices(nmo, nocc, frozen),))
    errors = []
    for f, no, selected in zip(focks, ns, frozen_sets):
        errors.extend((jnp.max(jnp.abs(f[:no, no:]), initial=0.),
                       jnp.max(jnp.abs(f[no:, :no]), initial=0.)))
        for start, end in ((0, no), (no, nmo)):
            active = jnp.asarray([i for i in range(start, end) if i not in selected], dtype=jnp.int32)
            inactive = jnp.asarray([i for i in range(start, end) if i in selected], dtype=jnp.int32)
            errors.extend((jnp.max(jnp.abs(f[jnp.ix_(active, inactive)]), initial=0.),
                           jnp.max(jnp.abs(f[jnp.ix_(inactive, active)]), initial=0.)))
            if active.size and inactive.size:
                a, b = f[jnp.ix_(active, active)], f[jnp.ix_(inactive, inactive)]
                ea = jnp.linalg.eigvalsh(jax.lax.stop_gradient(.5*(a+a.T)))
                eb = jnp.linalg.eigvalsh(jax.lax.stop_gradient(.5*(b+b.T)))
                gap = jnp.min(jnp.abs(ea[:, None]-eb[None, :]))
                errors.append(jnp.where(gap > frozen_gap_tol, 0., jnp.inf))
    return jnp.max(jnp.stack(errors))


def _active_focks(focks, selections):
    return tuple(f[jnp.ix_(jnp.concatenate((o, v)), jnp.concatenate((o, v)))]
                 for f, (o, v) in zip(focks, selections))


def _reference_data(h1, eri, nocc):
    """Validated full reference frames before projecting any frozen orbitals."""
    if isinstance(nocc, tuple):
        (ha, hb), (gaa, gab, gbb) = validate_unrestricted_integrals(h1, eri)
        n = ha.shape[0]
        unrestricted_frozen_indices(n, nocc)
        na, nb = nocc
        fa = (ha + jnp.einsum("pqii->pq", gaa[:, :, :na, :na])
              - jnp.einsum("piiq->pq", gaa[:, :na, :na, :])
              + jnp.einsum("pqii->pq", gab[:, :, :nb, :nb]))
        fb = (hb + jnp.einsum("pqii->pq", gbb[:, :, :nb, :nb])
              - jnp.einsum("piiq->pq", gbb[:, :nb, :nb, :])
              + jnp.einsum("iipq->pq", gab[:na, :na]))
        energy = .5 * (jnp.trace((ha + fa)[:na, :na])
                       + jnp.trace((hb + fb)[:nb, :nb]))
        return (ha, hb), (gaa, gab, gbb), (fa, fb), energy
    else:
        h, g = validate_integrals(h1, eri)
        if isinstance(nocc, bool) or not isinstance(nocc, int) or not 0 <= nocc <= h.shape[0]:
            raise ValueError("nocc must be an integer between 0 and nmo")
        f = (h + 2 * jnp.einsum("pqii->pq", g[:, :, :nocc, :nocc])
             - jnp.einsum("piiq->pq", g[:, :nocc, :nocc, :]))
        energy = jnp.trace((h + f)[:nocc, :nocc])
        return (h,), (g,), (f,), energy


def prepare_integrals(h1, eri, *, nocc, frozen=None, nuclear_repulsion=0., frozen_gap_tol=1e-10):
    hs, gs, focks, energy = _reference_data(h1, eri, nocc)
    energy = energy + nuclear_repulsion
    if isinstance(nocc, tuple):
        n = hs[0].shape[0]
        fr = unrestricted_frozen_indices(n, nocc, frozen)
        selections = tuple(active_indices(n, no, f) for no, f in zip(nocc, fr))
        (oa, va), (ob, vb) = selections
        gaa, gab, gbb = gs
        blocks = (gaa[jnp.ix_(oa, va, oa, va)], gab[jnp.ix_(oa, va, ob, vb)],
                  gbb[jnp.ix_(ob, vb, ob, vb)])
    else:
        h, g = hs[0], gs[0]
        selections = (active_indices(h.shape[0], nocc, frozen),)
        o, v = selections[0]
        blocks = (g[jnp.ix_(o, v, o, v)],)
    energies = tuple(jnp.diag(f)[jnp.concatenate((o, v))]
                     for f, (o, v) in zip(focks, selections))
    finite = (jnp.isfinite(energy) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x))
               for x in (*focks, *gs)])))
    return MPIntegrals(energies, tuple(len(o) for o, _ in selections), blocks,
                       None, energy, canonical_error(focks), finite,
                       _active_focks(focks, selections),
                       reference_error(focks, nocc, frozen=frozen, orbital_basis="noncanonical",
                                       frozen_gap_tol=frozen_gap_tol))


def from_orbitals(coefficients, focks, nocc, reference_energy, *, frozen=None,
                  frozen_gap_tol=1e-10, max_memory=DEFAULT_MAX_MEMORY_MB,
                  eri=None, eri_pair_matrix=None, df_factors=None, direct_basis=None):
    """Prepare MP2 occupied/virtual data from a real, supplied HF orbital frame.

    ``focks`` contains one restricted or two unrestricted full MO-frame Focks;
    ``reference_energy`` is the total HF energy, including nuclear repulsion.
    ``nocc`` is a static integer or (nalpha, nbeta) tuple. Frozen orbitals are
    removed only from the correlation space, preserving their Fock and energy
    contributions. No canonicalization or SCF source validation occurs here.

    Exactly one AO ERI representation is required. Native direct inputs retain
    only selected ovov blocks; DF keeps the existing Qia factors. ``max_memory``
    budgets each block transform, not all spin blocks, MP amplitudes or AD state.
    """
    max_memory = resolve_memory_mb(max_memory)
    unrestricted = isinstance(nocc, tuple)
    cs = tuple(map(jnp.asarray, coefficients)) if unrestricted else (jnp.asarray(coefficients),)
    if len(cs) != (2 if unrestricted else 1) or any(c.ndim != 2 for c in cs):
        raise ValueError("Supply one restricted or two unrestricted AO/MO coefficient matrices")
    if any(c.shape != cs[0].shape for c in cs):
        raise ValueError("Alpha and beta MO spaces must have equal dimensions")
    focks = tuple(map(jnp.asarray, focks))
    if len(focks) != len(cs) or any(f.shape != (c.shape[1], c.shape[1])
                                   for c, f in zip(cs, focks)):
        raise ValueError("Supply one full MO-frame Fock matrix for each coefficient matrix")
    reference_energy = jnp.asarray(reference_energy)
    if reference_energy.ndim != 0:
        raise ValueError("reference_energy must be scalar")
    if any(jnp.iscomplexobj(x) for x in (*cs, *focks, reference_energy)):
        raise NotImplementedError("MP requires real orbitals, Focks and reference energy")
    nmo = cs[0].shape[1]
    if unrestricted:
        ns = nocc
        fr = unrestricted_frozen_indices(nmo, ns, frozen)
    else:
        if isinstance(nocc, bool) or not isinstance(nocc, Integral) or not 0 <= nocc <= nmo:
            raise ValueError("nocc must be an integer between 0 and nmo")
        ns = (nocc,)
        fr = (frozen_indices(nmo, nocc, frozen),)
    if sum(x is not None for x in (eri, eri_pair_matrix, df_factors, direct_basis)) != 1:
        raise ValueError("Supply exactly one of eri, eri_pair_matrix, df_factors, direct_basis")
    integral_dtype = (jnp.float64 if direct_basis is not None else
                      jnp.asarray(next(x for x in (eri, eri_pair_matrix, df_factors)
                                       if x is not None)).dtype)
    dtype = jnp.result_type(*cs, *focks, integral_dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        raise NotImplementedError("MP requires real integral data")
    # Denominators, accumulators and optional amplitudes must share the dtype
    # of the transformed interaction; native direct integrals are float64.
    focks = tuple(f.astype(dtype) for f in focks)
    selections = tuple(active_indices(nmo, no, f) for no, f in zip(ns, fr))
    ovs = tuple((c[:, o], c[:, v]) for c, (o, v) in zip(cs, selections))
    energies = tuple(jnp.diag(f)[jnp.concatenate((o, v))]
                     for f, (o, v) in zip(focks, selections))
    pairs = ((0, 0), (0, 1), (1, 1)) if unrestricted else ((0, 0),)
    if df_factors is not None:
        factors = jnp.asarray(df_factors)
        if factors.ndim != 3 or factors.shape[1:] != (cs[0].shape[0],) * 2:
            raise ValueError("df_factors must have shape (naux, nao, nao)")
        if jnp.iscomplexobj(factors):
            raise NotImplementedError("MP requires real DF factors")
        ov_factors = tuple(jnp.einsum("Qpq,pi,qa->Qia", factors, o, v, precision="highest")
                           for o, v in ovs)
        blocks = None
        finite_data = (factors, *ov_factors)
    else:
        blocks = tuple(transform_eri_block((*ovs[a], *ovs[b]),
                                           eri=eri, eri_pair_matrix=eri_pair_matrix,
                                           direct_basis=direct_basis, max_memory=max_memory)
                       for a, b in pairs)
        ov_factors = None
        finite_data = (*blocks, *jax.tree.leaves(direct_basis))
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x))
                              for x in (reference_energy, *cs, *focks, *finite_data)]))
    return MPIntegrals(energies, tuple(len(o) for o, _ in selections), blocks,
                       ov_factors, reference_energy, canonical_error(focks), finite,
                       _active_focks(focks, selections), reference_error(
                           focks, nocc, frozen=frozen,
                           orbital_basis="noncanonical", frozen_gap_tol=frozen_gap_tol))


def from_scf(source, *, frozen=None, frozen_gap_tol=1e-10, max_memory=None):
    """Reuse current converged HF Focks and AO data without rebuilding J/K.

    This eager adapter validates the SCF source and occupations. The pure
    ``from_orbitals`` helper prepares selected correlation blocks, including
    exact native-direct inputs without a saved AO ERI tensor or pair matrix.
    UKS already owns a solved reference payload; it is read directly without
    invoking a reference builder. Legacy restricted JAX direct inputs retain
    their small-system packed AO reference path after a construction preflight.
    """
    max_memory = resolve_memory_mb(max_memory, source)
    data = prepare_scf_source(source, max_memory=max_memory)
    return from_orbitals(data.coefficients, data.focks, data.nocc, data.reference_energy,
                         frozen=frozen, frozen_gap_tol=frozen_gap_tol,
                         max_memory=max_memory, **data.ao_kwargs)
