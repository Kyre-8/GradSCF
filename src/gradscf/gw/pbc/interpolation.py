"""Explicit finite-grid interpolation and electronic DOS/gap diagnostics.

Interpolation adds no information beyond the source k mesh. Positive periodic
trilinear weights preserve Hermiticity and matrix spectral positivity. This
module performs eager postprocessing, not an SCF solve or an AD calculation.
"""
import numpy as np


def periodic_weights(source_fractional, query_fractional):
    """Nonnegative weights (nquery,nsource) on a complete uniform 3D grid."""
    source, query = np.asarray(source_fractional, float), np.asarray(query_fractional, float)
    if (source.ndim != 2 or source.shape[1] != 3 or not len(source)
            or query.ndim != 2 or query.shape[1] != 3
            or not np.isfinite(source).all() or not np.isfinite(query).all()):
        raise ValueError('Expected finite fractional k points with shape (nk,3)')
    source = np.mod(np.round(source, 10), 1.)
    axes = [np.unique(source[:, d]) for d in range(3)]
    if np.prod([len(a) for a in axes]) != len(source) or len(np.unique(source, axis=0)) != len(source):
        raise ValueError('Source points must form a complete Cartesian-product grid')
    output = np.ones((len(query), len(source)))
    for d, axis in enumerate(axes):
        if not np.allclose(np.diff(np.r_[axis, axis[0]+1]), 1/len(axis), atol=1e-8, rtol=0):
            raise ValueError('Source points must be uniformly spaced')
        position = np.mod(query[:, d]-axis[0], 1.)*len(axis)
        lower = np.floor(position).astype(int) % len(axis)
        fraction = position-np.floor(position)
        indices = np.argmin(abs(source[:, d, None]-axis), axis=1)
        output *= ((indices[None] == lower[:, None])*(1-fraction[:, None])
                   +(indices[None] == ((lower+1) % len(axis))[:, None])*fraction[:, None])
    return output


def interpolate_operator(weights, overlap, source_coeff, source_matrices, query_coeff):
    """Transport orthonormal-MO operators through the common Bloch AO frame.

    source_matrices has shape (nk,...,nmo,nmo). All query coefficients use the
    same AO functions/convention as the source. The source projection is
    S C A C† S, so independent source MO phases cancel exactly. AO interpolation
    itself is a declared finite-mesh approximation and is not gauge invariant
    under arbitrary k-dependent changes of the AO Bloch convention.
    """
    weights, overlap = np.asarray(weights), np.asarray(overlap)
    c, a, cq = map(np.asarray, (source_coeff, source_matrices, query_coeff))
    if (weights.ndim != 2 or weights.shape[1] != len(c) or len(cq) != len(weights)
            or overlap.shape != (len(c), c.shape[1], c.shape[1])
            or a.shape[0] != len(c) or a.shape[-2:] != (c.shape[-1],)*2):
        raise ValueError('Inconsistent source/query operator dimensions')
    if not np.isfinite(weights).all() or np.min(weights) < -1e-14 or not np.allclose(weights.sum(axis=1), 1.):
        raise ValueError('Interpolation weights must be normalized and nonnegative')
    sc = overlap@c
    ao = np.einsum('kai,k...ij,kbj->k...ab', sc, a, sc.conj())
    interpolated = np.einsum('qk,k...ab->q...ab', weights, ao)
    return np.einsum('qai,q...ab,qbj->q...ij', cq.conj(), interpolated, cq)


def electronic_dos(axis, energies, *, weights=None, sigma=.01, spectral_weights=None,
                   spin=2., smearing='gaussian'):
    """Normalized broadened DOS; axis/energy/sigma use the same energy unit.

    Input k weights sum to one. The returned unit is states/cell/energy.
    spectral_weights enables a spectral pole DOS (including satellites),
    whereas omitted weights give one state per supplied band per spin.
    All requested data must be finite; failed QPs cannot be renormalized away.
    sigma is the Gaussian standard deviation or Lorentzian halfwidth. The
    latter matches the explicit retarded resolvent eta used for spectral DOS.
    """
    axis, energies = np.asarray(axis, float), np.asarray(energies, float)
    if (axis.ndim != 1 or energies.ndim != 2 or not len(energies)
            or not np.isfinite(axis).all() or not np.isfinite(energies).all()
            or not np.isfinite(sigma) or sigma <= 0 or spin <= 0):
        raise ValueError('DOS requires finite complete data and positive broadening/spin')
    weights = np.full(len(energies), 1/len(energies)) if weights is None else np.asarray(weights, float)
    if weights.shape != (len(energies),) or np.any(weights < 0) or not np.allclose(weights.sum(), 1.):
        raise ValueError('BZ weights must be nonnegative and sum to one')
    spectral = np.ones_like(energies) if spectral_weights is None else np.asarray(spectral_weights, float)
    if spectral.shape != energies.shape or not np.isfinite(spectral).all() or np.any(spectral < 0):
        raise ValueError('Spectral pole weights must be finite and nonnegative')
    if smearing not in ('gaussian', 'lorentzian'):
        raise ValueError('Smearing must be gaussian or lorentzian')
    output = np.zeros(len(axis))
    for k, energy in enumerate(energies):
        difference = axis[:, None]-energy
        broadened = (np.exp(-.5*(difference/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
                     if smearing == 'gaussian' else sigma/(np.pi*(difference**2+sigma**2)))
        output += spin*weights[k]*(broadened@spectral[k])
    return output


def band_gaps(energies, *, nocc, valid=None):
    """Sampled fundamental and minimum direct gaps, without a hidden nanmin."""
    energies = np.asarray(energies)
    if energies.ndim != 2 or not 0 < nocc < energies.shape[1] or not len(energies):
        raise ValueError('Need occupied and empty bands at every sampled k point')
    if not np.isfinite(energies).all() or (valid is not None and not np.asarray(valid).all()):
        raise ValueError('Bands must all be finite and converged to identify the frontier')
    # QP branches retain KS occupancy labels but can reorder within either
    # group. Checking only nocc-1/nocc would miss the actual band edges.
    valence = np.max(energies[:, :nocc], axis=1)
    conduction = np.min(energies[:, nocc:], axis=1)
    return dict(sampled_gap_hartree=float(conduction.min()-valence.max()),
        minimum_direct_gap_hartree=float(np.min(conduction-valence)),
        vbm_index=int(np.argmax(valence)), cbm_index=int(np.argmin(conduction)))


def align_source(model, overlap, saved_coeff, saved_energy, *matrices, tolerance=1e-8):
    """Transport saved source-MO matrices into an equivalent model MO frame.

    U=C_new^dagger S C_saved must be unitary and intertwine the saved/new
    reference Hamiltonians. Independent phases and rotations within exactly
    degenerate KS spaces are allowed. Each matrix has shape (nk,...,nmo,nmo)
    and is transformed as U A U^dagger, preserving complex, non-Hermitian
    Matsubara data. Calling without matrices validates only the frame.
    """
    current, old, metric = map(np.asarray, (model.coeff, saved_coeff, overlap))
    energy, previous = np.asarray(model.energy), np.asarray(saved_energy)
    if (not np.isfinite(tolerance) or tolerance <= 0 or energy.ndim != 2
            or current.ndim != 3 or old.shape != current.shape
            or current.shape[0] != energy.shape[0] or current.shape[-1] != energy.shape[-1]
            or metric.shape != (len(current), current.shape[1], current.shape[1])
            or previous.shape != energy.shape or np.iscomplexobj(energy) or np.iscomplexobj(previous)
            or not all(np.isfinite(value).all() for value in (current, old, metric, energy, previous))):
        raise ValueError('Use finite matched source frames, real energies and a positive tolerance')
    rotation = current.conj().swapaxes(-1, -2)@metric@old
    if not np.allclose(rotation.conj().swapaxes(-1, -2)@rotation,
                       np.eye(energy.shape[-1]), atol=tolerance, rtol=0):
        raise ValueError('Saved and new reference MO frames are not related by a unitary transformation')
    transported_energy = (rotation*previous[:, None, :])@rotation.conj().swapaxes(-1, -2)
    if not np.allclose(transported_energy, np.array([np.diag(row) for row in energy]),
                       atol=tolerance, rtol=0):
        raise ValueError('Saved and new frames do not describe the same reference Hamiltonian')
    output = []
    for matrix in matrices:
        value = np.asarray(matrix)
        if (value.ndim < 3 or value.shape[0] != len(energy)
                or value.shape[-2:] != (energy.shape[-1],)*2 or not np.isfinite(value).all()):
            raise ValueError('Use finite source matrices with shape (nk,...,nmo,nmo)')
        output.append(np.einsum('kai,k...ij,kbj->k...ab', rotation, value, rotation.conj()))
    return tuple(output)
