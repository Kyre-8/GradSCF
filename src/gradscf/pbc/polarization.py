"""Discrete Berry polarization for a gapped, integer-occupied full k mesh.

Periodic AO parts use the existing lattice-translation Bloch gauge. Positive
links are <u_k|u_(k+q)>; their phases give +e*a*phase/(2*pi*volume), so an electron
localized at r contributes -e*r/volume. Each explicit spin channel has unit
occupation. Ionic charges are the pseudopotential charges, not atomic numbers.

Polarization is branch-valued. Without reference_phases each string uses its
principal phase; no global transverse branch or topological-sector inference
is made. With reference_phases the result follows the nearest local branch of
each string. It is the caller's responsibility to resolve longer paths with
intermediate points. This module supplies neither a macroscopic electric field
nor dielectric response. See Souza et al., PRL 89, 117602 (2002), Eq. 2.
"""
from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .kpoints import kmesh_shape, reciprocal_vectors, weights_for
from .momentum import momentum_transfer
from ..solvers.diagnostics import require_converged_derivative


@dataclass(frozen=True, eq=False)
class BerryPlan:
    """Host-prepared strings; fractional k representatives remain explicit."""
    kpoints: object
    shape: tuple
    partners: tuple
    shifts: tuple
    strings: tuple


class FrameResult(NamedTuple):
    coefficients: object
    minimum_gram_eigenvalue: object
    projector_error: object
    orthogonality_error: object
    valid: object


class _BerryResult(NamedTuple):
    electronic: object
    phases: tuple
    minimum_link_singular_value: object
    maximum_link_condition: object
    branch_clearance: object
    valid: object
    response_valid: object


class PolarizationResult(NamedTuple):
    total: object
    electronic: object
    ionic: object
    phases: tuple
    minimum_link_singular_value: object
    maximum_link_condition: object
    minimum_gap: object
    orthogonality_error: object
    subspace_residual: object
    spectrum_error: object
    branch_clearance: object
    valid: object
    response_valid: object


def make_plan(fractional_kpoints, *, kweights=None):
    """Prepare three closed strings on a complete equal-weight product mesh.

    The supplied point order and reciprocal representatives are retained.
    Shifted meshes and single-point strings are allowed. Preparation is host
    only; reuse this static plan inside a traced array calculation.
    """
    if np.iscomplexobj(fractional_kpoints):
        raise ValueError('Fractional k points must be real.')
    k = np.asarray(fractional_kpoints, dtype=float)
    if k.ndim != 2 or k.shape[1] != 3 or not len(k) or not np.isfinite(k).all():
        raise ValueError('Use finite fractional k points with shape (nk,3).')
    # Normalize before rounding: mod after rounding a large representative
    # otherwise creates distinct floating residues for the same fractional k.
    wrapped = np.mod(np.round(np.mod(k, 1.), 10), 1.)
    shape = kmesh_shape(np.eye(3), wrapped*(2*np.pi))
    weights = np.full(len(k), 1/len(k)) if kweights is None else np.asarray(kweights)
    if weights.shape != (len(k),) or np.iscomplexobj(weights) or not np.isfinite(weights).all() or not np.allclose(weights, 1/len(k), atol=1e-12, rtol=0):
        raise ValueError('Berry strings require equal normalized k weights.')
    axes = [np.unique(wrapped[:, i], return_inverse=True) for i in range(3)]
    for values, _ in axes:
        if not np.allclose(np.diff(np.r_[values, values[0]+1]), 1/len(values), atol=2e-10, rtol=0):
            raise ValueError('Use a complete uniform k mesh.')
    grid = np.empty(shape, dtype=int)
    grid[tuple(item[1] for item in axes)] = np.arange(len(k))
    transfers = [momentum_transfer(k, np.eye(3)[i]/shape[i]) for i in range(3)]
    strings = tuple(np.moveaxis(grid, i, -1).reshape(-1, shape[i]) for i in range(3))
    return BerryPlan(k.copy(), shape, tuple(t.partners for t in transfers),
                     tuple(t.shifts for t in transfers), strings)


def _checked(valid, message):
    valid = jax.lax.stop_gradient(valid)
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError(message)
    return valid


def _adjoint(a):
    return a.conj().swapaxes(-1, -2)


def ao_links(ao, weights, coordinates, lattice, plan):
    """Cross-k periodic AO overlaps, shape (3,nk,nao,nao).

    ao has the existing (nk,4,ngrid,nao) PeriodicInputs layout. If the stored
    partner is k'=k+q+G, lift its periodic factor with exp(+i*G.r). This applies
    to every link including the closing one, even for a single-point string.
    Grid/lattice/center dependence stays in the ordinary JAX graph.
    """
    values = jnp.asarray(ao)[:, 0]
    weights, coordinates, lattice = map(jnp.asarray, (weights, coordinates, lattice))
    if values.ndim != 3 or values.shape[0] != len(plan.kpoints) or weights.shape != (values.shape[1],) or coordinates.shape != (len(weights), 3) or lattice.shape != (3, 3):
        raise ValueError('Inconsistent AO, grid, lattice or Berry-plan shapes.')
    if any(jnp.iscomplexobj(x) for x in (weights, coordinates, lattice)):
        raise ValueError('Grid weights, coordinates and lattice must be real.')
    valid = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(coordinates))
    valid &= jnp.all(jnp.isfinite(weights) & (weights > 0)) & (jnp.linalg.det(lattice) > 0)
    valid = _checked(valid, 'Finite AOs, positive weights and a right-handed lattice are required.')
    reciprocal = reciprocal_vectors(lattice)
    blocks = []
    for partner, shifts in zip(plan.partners, plan.shifts):
        phase = jnp.exp(1j*(jnp.asarray(shifts)@reciprocal)@coordinates.T)
        lifted = values[partner]*phase[:, :, None]
        blocks.append(jnp.einsum('g,kgp,kgq->kpq', weights, values.conj(), lifted))
    return jnp.stack(blocks)*jnp.where(valid, 1., jnp.nan)


def _projected_frame(projector, overlap, reference):
    """Smooth local frame extension; idempotency is not an input constraint.

    The Gram matrix must remain positive definite. Physical-state callers add
    their projector checks; variational energy derivatives use this extension
    in the surrounding open set and enforce idempotency in their residual.
    """
    y = projector@overlap@reference
    gram = _adjoint(y)@overlap@y
    factor = jnp.linalg.cholesky(gram)
    coefficients = _adjoint(jnp.linalg.solve(factor, _adjoint(y)))
    return coefficients, gram


def occupied_frame(projector, overlap, reference, *, tolerance=1e-8, gram_tol=1e-10):
    """Local S-orthonormal frame of Q S Q=Q using a reference occupied frame.

    Q has unit occupation per spin. Shapes are (...,nao,nao) and
    (...,nao,nocc); overlap may broadcast over leading blocks. Y=Q S Cref,
    L L^H=Y^H S Y and C=Y L^(-H). The chart is rejected when its reference
    projection loses rank. No eigenvectors or hidden regularization are used.
    Concrete invalid data raise; traced invalid data return NaN coefficients.
    """
    q, s, ref = map(jnp.asarray, (projector, overlap, reference))
    if q.ndim < 2 or q.shape[-2] != q.shape[-1] or s.shape[-2:] != q.shape[-2:] or ref.shape[-2] != q.shape[-1] or not 0 < ref.shape[-1] <= q.shape[-1]:
        raise ValueError('Inconsistent projector, overlap or occupied-reference shapes.')
    if not all(isfinite(x) and x > 0 for x in (tolerance, gram_tol)):
        raise ValueError('Frame tolerances must be finite and positive.')
    c, gram = _projected_frame(q, s, ref)
    minimum = jnp.min(jnp.linalg.eigvalsh(jax.lax.stop_gradient(gram)))
    metric_minimum = jnp.min(jnp.linalg.eigvalsh(jax.lax.stop_gradient(s)))
    error = jnp.maximum(jnp.max(jnp.abs(q@s@q-q)), jnp.max(jnp.abs(q-_adjoint(q))))
    rank = jnp.trace(q@s, axis1=-2, axis2=-1)
    valid = jnp.all(jnp.isfinite(gram)) & (minimum > gram_tol) & (error < tolerance)
    valid &= metric_minimum > 0
    valid &= jnp.max(jnp.abs(s-_adjoint(s))) < tolerance
    valid &= jnp.max(jnp.abs(rank-ref.shape[-1])) < tolerance
    valid = _checked(valid, 'Invalid projector/metric or singular occupied-frame chart.')
    orthogonality = jnp.max(jnp.abs(_adjoint(c)@s@c-jnp.eye(ref.shape[-1])))
    valid = _checked(valid & (orthogonality < tolerance), 'Occupied-frame orthogonality failed.')
    c = require_converged_derivative(c*jnp.where(valid, 1., jnp.nan), valid)
    return FrameResult(c, minimum,
                       jax.lax.stop_gradient(error), jax.lax.stop_gradient(orthogonality), valid)


def _berry(occupied, links, lattice, plan, *, reference_phases=None, link_tol=1e-8):
    """Shared phase algebra for the checked SCF facade and local DFPT frames.

    This function does not require canonical orbitals or SCF stationarity. Its
    callers check their occupied-frame domain. It contains the sole determinant,
    folding-link contraction and local branch implementation.
    """
    if reference_phases is not None:
        if len(reference_phases) != 3 or any(jnp.shape(p) != (len(occupied), len(strings)) for p, strings in zip(reference_phases, plan.strings)):
            raise ValueError('Reference phases must match spin and string dimensions.')
        if any(jnp.iscomplexobj(p) for p in reference_phases):
            raise ValueError('Reference phases must be real.')
    phases, minima, conditions, clearances = [], [], [], []
    valid = jnp.array(True)
    for axis, (partner, strings) in enumerate(zip(plan.partners, plan.strings)):
        spin_phases, spin_minima, spin_conditions = [], [], []
        for frame in occupied:
            matrix = _adjoint(frame)@links[axis]@frame[partner]
            singular = jnp.linalg.svd(jax.lax.stop_gradient(matrix), compute_uv=False)
            spin_minima.append(jnp.min(singular))
            spin_conditions.append(jnp.max(singular[:, 0]/singular[:, -1]))
            sign, _ = jnp.linalg.slogdet(matrix)
            phase = jnp.sum(jnp.angle(sign)[strings], axis=-1)
            spin_phases.append(jnp.arctan2(jnp.sin(phase), jnp.cos(phase)))
        phase = jnp.stack(spin_phases)
        delta = phase if reference_phases is None else jnp.angle(jnp.exp(1j*(phase-jnp.asarray(reference_phases[axis]))))
        clearance = jnp.min(jnp.pi-jnp.abs(delta))
        if reference_phases is not None:
            phase = jnp.asarray(reference_phases[axis])+delta
            valid &= clearance > 1e-8
        phases.append(phase)
        minima.append(jnp.stack(spin_minima)); conditions.append(jnp.stack(spin_conditions))
        clearances.append(clearance)
    minima, conditions = jnp.stack(minima), jnp.stack(conditions)
    valid = _checked(valid & jnp.all(minima > link_tol), 'Singular Berry link or ambiguous local branch; refine the mesh/path.')
    clearances = jax.lax.stop_gradient(jnp.stack(clearances))
    response_valid = valid & jnp.all(clearances > 1e-8)
    averages = jnp.stack([jnp.mean(p, axis=-1).sum() for p in phases])
    electronic = averages@lattice/(2*jnp.pi*jnp.linalg.det(lattice))
    checked = lambda value: require_converged_derivative(value*jnp.where(valid, 1., jnp.nan), response_valid)
    return _BerryResult(checked(electronic), tuple(checked(p) for p in phases),
                        minima, conditions, clearances, valid, response_valid)


def polarization(inputs, state, plan, *, nocc, reference_phases=None,
                 tolerance=1e-8, gap_tol=1e-8, link_tol=1e-8):
    """Berry polarization in e/bohr^2 from existing periodic array records.

    nocc is a static occupied count for each explicit spin block. Only neutral
    zero-temperature references with full integer occupation and an indirect
    occupied/empty gap are accepted. state is PeriodicSCFResult-compatible;
    inputs are PeriodicInputs-compatible. No SCF is performed here.
    Diagnostics are nondifferentiated checks of the actual links/reference.
    reference_phases is a tuple of three (nspin,nstrings) phase arrays. When
    absent, the reported principal-string convention is not a globally unwrapped
    polarization branch. Local branch differences require perturbation steps
    smaller than pi on every string; longer paths require intermediate points.
    At a principal-phase cut the forward value remains available, while
    response_valid is false and JVP/VJP return NaN. Supplying a local phase
    anchor defines the continuous branch and removes that artificial cut.
    Occupied and empty frames need not be canonical: their subspace stationarity
    and block spectra are checked without choosing individual orbital gauges.
    """
    if not jax.config.x64_enabled:
        raise ValueError('Berry polarization requires float64.')
    if not all(isfinite(x) and x > 0 for x in (tolerance, gap_tol, link_tol)):
        raise ValueError('Polarization tolerances must be finite and positive.')
    c, energy, occupation = map(jnp.asarray, (state.mo_coeff_spin, state.mo_energy_spin, state.mo_occ_spin))
    s = jnp.asarray(inputs.overlap)
    if c.ndim != 4 or c.shape[1] != len(plan.kpoints) or c.shape[-1] != c.shape[-2] or energy.shape != c.shape[:2]+c.shape[-1:] or occupation.shape != energy.shape or s.shape != c.shape[1:]:
        raise ValueError('Use complete spin/k/AO orbital records and matching overlaps.')
    if len(nocc) != c.shape[0] or any(type(n) not in (int, np.int64, np.int32) or not 0 < n < c.shape[-1] for n in nocc):
        raise ValueError('Each nocc must be a static integer between zero and nao.')
    if jnp.iscomplexobj(energy) or jnp.iscomplexobj(occupation):
        raise ValueError('Energies and occupations must be real.')
    expected = jnp.stack([jnp.broadcast_to((jnp.arange(c.shape[-1])<n).astype(float), energy.shape[1:]) for n in nocc])
    orthogonality = jnp.max(jnp.abs(_adjoint(c)@s@c-jnp.eye(c.shape[-1])))
    density = (c*expected[..., None, :])@_adjoint(c)
    fock_mo = _adjoint(c)@jnp.asarray(state.fock_spin)@c
    hermiticity = jnp.max(jnp.abs(fock_mo-_adjoint(fock_mo)))
    # Validate invariant subspaces, not individual canonical eigenvectors.
    # A full occupied unitary rotation is a gauge even when its energies differ.
    checked_fock, checked_energy = jax.lax.stop_gradient((fock_mo, energy))
    gaps, residuals, spectral_errors = [], [], []
    for spin, n in enumerate(nocc):
        occupied_values = jnp.linalg.eigvalsh(checked_fock[spin, :, :n, :n])
        empty_values = jnp.linalg.eigvalsh(checked_fock[spin, :, n:, n:])
        gaps.append(jnp.min(empty_values)-jnp.max(occupied_values))
        residuals.append(jnp.max(jnp.abs(fock_mo[spin, :, n:, :n])))
        spectral_errors.extend([jnp.max(jnp.abs(occupied_values-jnp.sort(checked_energy[spin, :, :n]))),
                                jnp.max(jnp.abs(empty_values-jnp.sort(checked_energy[spin, :, n:])))])
    gaps = jnp.stack(gaps)
    residual = jnp.max(jnp.stack(residuals))
    spectrum_error = jnp.max(jnp.stack(spectral_errors))
    fractional = jnp.asarray(inputs.kpoints)@jnp.asarray(inputs.lattice).T/(2*jnp.pi)
    valid = jnp.asarray(state.converged) & jnp.all(jnp.isfinite(c)) & jnp.all(jnp.isfinite(energy))
    valid &= jnp.max(jnp.abs(occupation-expected)) < tolerance
    valid &= (orthogonality < tolerance) & (jnp.min(gaps) > gap_tol) & (residual < tolerance)
    valid &= (hermiticity < tolerance) & (spectrum_error < tolerance)
    valid &= jnp.max(jnp.abs(density-state.density_spin)) < tolerance
    valid &= jnp.max(jnp.abs(fractional-jnp.asarray(plan.kpoints))) < tolerance
    valid &= jnp.max(jnp.abs(weights_for(inputs)-1/len(plan.kpoints))) < 1e-12
    charges, positions = jnp.asarray(inputs.ionic_charges), jnp.asarray(inputs.atom_coords)
    if charges.ndim != 1 or positions.shape != (len(charges), 3) or jnp.iscomplexobj(charges) or jnp.iscomplexobj(positions):
        raise ValueError('Supply real ionic charges and Cartesian atom coordinates.')
    valid &= jnp.all(jnp.isfinite(charges)) & jnp.all(jnp.isfinite(positions))
    valid &= jnp.abs(jnp.sum(charges)-sum(nocc)) < tolerance
    valid = _checked(valid, 'Berry reference must be converged, neutral, gapped, S-orthonormal and integer-occupied on the declared mesh.')
    links = ao_links(inputs.ao, inputs.weights, inputs.coords, inputs.lattice, plan)
    frames = tuple(c[spin, :, :, :n] for spin, n in enumerate(nocc))
    berry = _berry(frames, links, inputs.lattice, plan, reference_phases=reference_phases, link_tol=link_tol)
    valid &= berry.valid
    response_valid = valid & berry.response_valid
    volume = jnp.linalg.det(inputs.lattice)
    electronic = berry.electronic
    ionic = charges@positions/volume
    mask = jnp.where(valid, 1., jnp.nan)
    checked = lambda value: require_converged_derivative(value*mask, response_valid)
    return PolarizationResult(checked(electronic+ionic), checked(electronic), ionic*mask,
        tuple(checked(p) for p in berry.phases), berry.minimum_link_singular_value, berry.maximum_link_condition, jnp.min(gaps),
        jax.lax.stop_gradient(orthogonality), jax.lax.stop_gradient(residual), spectrum_error,
        berry.branch_clearance, valid, response_valid)


__all__ = ['BerryPlan', 'FrameResult', 'PolarizationResult', 'make_plan',
           'ao_links', 'occupied_frame', 'polarization']
