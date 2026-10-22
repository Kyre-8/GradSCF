"""Restricted MP3 from the first-order connected doubles interaction.

For canonical HF, T1^(1)=0 and E^(3) depends only on T2^(2).
The interaction action is the zero-amplitude derivative of the existing CCSD
doubles residual with its one-body Fock part removed. No CC iteration, fitted
scaling, or finite-difference derivative is used. See REFERENCES.md.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp

from ..cc.integrals import prepare_integrals, denominators
from ..cc.rccsd import residual, correlation_energy
from .._memory import (DEFAULT_MAX_MEMORY_MB, jaxpr_temporary_bytes,
                      resolve_memory_mb, tree_storage_bytes,
                      validate_reserved_bytes, workspace_bytes)
from ..ao2mo import frozen_indices, validate_integrals
from .mp2 import evaluate_mp2
from .integrals import (MPIntegrals, canonical_error, reference_error,
                        prepare_integrals as prepare_mp_integrals)


def evaluate_mp3(h1, eri, *, nocc, frozen=None, nuclear_repulsion=0., config,
                 max_memory=None, reserved_bytes=0):
    """Original canonical RHF kernel with optional joint known-array planning.

    ``None`` retains the original unbudgeted numerical path. A supplied MB
    budget covers raw inputs, selected blocks, first/second-order doubles and
    declared outputs of the original residual-JVP graph. Compilation, opaque
    library scratch and arbitrary outer derivative storage are not a cap.
    """
    if isinstance(nocc, tuple):
        raise NotImplementedError("MP3 currently requires a restricted canonical HF reference")
    reserved = validate_reserved_bytes(reserved_bytes)
    maximum = None if max_memory is None else resolve_memory_mb(max_memory)
    def prepare(a, b, nuclear, budget=None, reservation=0):
        first = evaluate_mp2(prepare_mp_integrals(a, b, nocc=nocc, frozen=frozen,
            nuclear_repulsion=nuclear), replace(config, order=2, with_t2=True),
            max_memory=budget, reserved_bytes=reservation)
        return prepare_integrals(a, b, nocc=nocc, frozen=frozen,
                                  nuclear_repulsion=nuclear), first
    source = (h1, eri, nuclear_repulsion)
    if maximum is not None:
        h1, eri = validate_integrals(h1, eri)
        source = (h1, eri, jnp.asarray(nuclear_repulsion))
        workspace_bytes(maximum, resident_bytes=reserved+tree_storage_bytes(source))
        planned, _, scratch = _plan(prepare, source, config, maximum, reserved)
        reservation = reserved+tree_storage_bytes(source, planned)+scratch
        ints, first = prepare(*source, maximum, reservation)
    else:
        ints, first = prepare(*source)
    return _finish(ints, first, config)


def _finish(ints, first, config):
    """The unchanged zero-amplitude residual JVP and MP3 energy contraction."""
    _, d2 = denominators(ints)
    # F_N belongs to H0, so retain only the fluctuation interaction in this action.
    interaction = ints._replace(fock=jnp.zeros_like(ints.fock))
    zero1 = jnp.zeros((ints.nocc, ints.nvir), dtype=ints.fock.dtype)
    zero2 = jnp.zeros_like(first.t2)
    action = lambda t: residual(zero1, t, interaction, model="ccd")[1]
    _, linear = jax.jvp(action, (zero2,), (first.t2,))
    second = linear / jnp.where(jnp.abs(d2) > config.denominator_tol, d2, 1.)
    e3 = correlation_energy(zero1, second, interaction, model="lccd")
    valid = first.valid & jnp.isfinite(e3)
    e3 = jnp.where(valid, e3, jnp.nan)
    ecorr = first.e2 + e3
    return first._replace(total_energy=first.reference_energy + ecorr,
                           correlation_energy=ecorr, e3=e3, valid=valid,
                           corrections=jnp.stack([first.e2, e3]),
                           t2=first.t2 if config.with_t2 else None)


def _plan(prepare, inputs, config, maximum, reserved, *, owners=()):
    """Forecast arrays before preparation; no numerical kernel is executed."""
    ints, first = jax.eval_shape(prepare, *inputs)
    graph = jax.make_jaxpr(lambda data, state: _finish(data, state, config))(ints, first)
    scratch = jaxpr_temporary_bytes(graph)
    scratch += 6*inputs[0].shape[-1]**2*ints.fock.dtype.itemsize
    workspace_bytes(maximum, resident_bytes=reserved+tree_storage_bytes(
        inputs, owners, ints, first)+scratch)
    return ints, first, scratch


def _mp_view(ints, full_fock, nocc, frozen, owners, config):
    """Share the prepared ovov/Fock leaves; retain full-reference guards."""
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in
        jax.tree.leaves((ints, owners)) if hasattr(x, 'dtype')]))
    return MPIntegrals((ints.mo_energy,), (ints.nocc,), (ints.ovov,), None,
        ints.reference_energy, canonical_error((full_fock,)), finite, (ints.fock,),
        reference_error((full_fock,), nocc, frozen=frozen,
                        orbital_basis='noncanonical', frozen_gap_tol=config.denominator_tol))


def run_from_orbitals(coefficients, focks, nocc, reference_energy, *, frozen=None,
                      config, eri=None, eri_pair_matrix=None, df_factors=None,
                      direct_basis=None, max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0):
    """Canonical RHF MP3 from one selected CC preparation and live AO inputs.

    Focks are a length-one tuple containing the complete MO Fock before
    freezing; the HF reference energy includes nuclear repulsion. The seven
    CC blocks and the necessary doubles remain explicit. No full MO/spin ERI,
    extra MP2 transform, SCF or orbital canonicalization is performed.
    The interaction uses C/F/AO precision independently of the reference
    energy; the native AO source requires float64 for both.
    """
    if isinstance(nocc, (tuple, list)):
        raise NotImplementedError('Standalone MP3 requires a restricted reference')
    if config.orbital_basis != 'canonical':
        raise ValueError('Standalone MP3 requires canonical orbitals')
    maximum, reserved = resolve_memory_mb(max_memory), validate_reserved_bytes(reserved_bytes)
    c, fs, energy = jnp.asarray(coefficients), tuple(map(jnp.asarray, focks)), jnp.asarray(reference_energy)
    if (c.ndim != 2 or len(fs) != 1 or fs[0].shape != (c.shape[1],)*2
            or energy.shape != () or isinstance(nocc, bool) or not isinstance(nocc, int)
            or not 0 <= nocc <= c.shape[1]):
        raise ValueError('Supply a complete restricted C/F frame, nocc and scalar HF energy')
    if any(jnp.iscomplexobj(x) or not jnp.issubdtype(x.dtype, jnp.floating)
           for x in (c, *fs, energy)):
        raise ValueError('Canonical MP3 requires real floating-point C/F/energy inputs')
    source = dict(eri=eri, eri_pair_matrix=eri_pair_matrix, df_factors=df_factors,
                  direct_basis=direct_basis)
    if sum(x is not None for x in source.values()) != 1:
        raise ValueError('Supply exactly one AO ERI representation')
    source = {k: (jnp.asarray(v) if k != 'direct_basis' and v is not None else v)
              for k, v in source.items()}
    frozen = frozen_indices(c.shape[1], nocc, frozen)
    no = nocc-sum(i < nocc for i in frozen)
    nv = c.shape[1]-nocc-sum(i >= nocc for i in frozen)
    dtype = (jnp.dtype(jnp.float64) if direct_basis is not None else
             jnp.result_type(c, *fs, *jax.tree.leaves(source)))
    energy_dtype = jnp.dtype(jnp.float64) if direct_basis is not None else energy.dtype
    owners = (c, fs, energy, source)
    planned_fock = (fs[0] if fs[0].dtype == dtype else
                   jax.ShapeDtypeStruct(fs[0].shape, dtype))
    planned_energy = (energy if energy.dtype == energy_dtype else
                     jax.ShapeDtypeStruct(energy.shape, energy_dtype))
    minimum = 2*no*no*nv*nv*dtype.itemsize
    workspace_bytes(maximum, resident_bytes=reserved+tree_storage_bytes(
                        owners, planned_fock, planned_energy)
                    +minimum)
    fock = fs[0] if fs[0].dtype == dtype else fs[0].astype(dtype)
    energy = energy if energy.dtype == energy_dtype else energy.astype(energy_dtype)
    from ..cc.integrals import from_orbitals
    def prepare(c, f, e):
        ints = from_orbitals(c, f, nocc, e, frozen=frozen, max_memory=maximum,
                            reserved_bytes=reserved+minimum, **source)
        first = evaluate_mp2(_mp_view(ints, f, nocc, frozen, owners, config),
                             replace(config, order=2, with_t2=True))
        return ints, first
    _, first, scratch = _plan(prepare, (c, fock, energy), config, maximum,
                              reserved, owners=owners)
    extra = reserved+tree_storage_bytes(owners, exclude=(c, fock, energy, source))
    ints = from_orbitals(c, fock, nocc, energy, frozen=frozen, max_memory=maximum,
        reserved_bytes=extra+tree_storage_bytes(first)+scratch, **source)
    mp_ints = _mp_view(ints, fock, nocc, frozen, owners, config)
    first = evaluate_mp2(mp_ints, replace(config, order=2, with_t2=True), max_memory=maximum,
        reserved_bytes=reserved+scratch+tree_storage_bytes(ints, owners, exclude=(mp_ints,)))
    return _finish(ints, first, config)
