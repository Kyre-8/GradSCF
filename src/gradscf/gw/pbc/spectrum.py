"""Eager spectral reduction of the finite periodic GW source model.

Static QP Hamiltonian interpolation is distinct from direct off-mesh GW.
Matsubara scGW instead requires a checked causal full-matrix continuation.
Neither a small imaginary-axis fit error nor a dense drawing proves unique
real-axis spectra or material convergence; test pole/grid stability as well.
"""
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ...solvers.poles import interpolate_causal, evaluate_poles, auxiliary_spectrum
from ...pbc.kpoints import time_reversal_plan
from .interpolation import periodic_weights, interpolate_operator


def _reverse_source(values, partners, sewing):
    """Same-z spinless time reversal, including arbitrary source MO phases."""
    return np.einsum('kai,k...ji,kbj->k...ab', sewing,
                     np.asarray(values)[partners], sewing.conj(), optimize=True)


def _time_reversal_sources(model, result, tolerance):
    """Validate a TR-closed source subspace and the original, unfitted state."""
    coeff = np.asarray(model.coeff)
    fractional = np.asarray(model.kpoints)@np.asarray(model.lattice).T/(2*np.pi)
    partners = time_reversal_plan(fractional).partners
    if (coeff.ndim != 3 or not np.isfinite(coeff).all()
            or partners.shape != (len(coeff),)):
        raise ValueError('Use a finite source MO frame and valid k/-k partner indices')
    if not np.array_equal(partners[partners], np.arange(len(coeff))):
        raise ValueError('The model must have a reciprocal-closed k/-k source pairing')
    sewing, reports = [], []
    for k, partner in enumerate(partners):
        target = coeff[partner].conj()
        rotation, _, rank, _ = np.linalg.lstsq(coeff[k], target, rcond=None)
        span_error = float(np.linalg.norm(coeff[k]@rotation-target)/max(np.linalg.norm(target), 1e-30))
        unitary_error = float(np.max(abs(rotation.conj().T@rotation-np.eye(coeff.shape[-1]))))
        if rank != coeff.shape[-1] or max(span_error, unitary_error) > tolerance:
            raise ValueError('The source MO subspace is not closed under unitary time reversal')
        sewing.append(rotation)
        reports.append(dict(partner=int(partner), subspace_relative_error=span_error,
                            sewing_unitary_error=unitary_error))
    sewing = np.asarray(sewing)
    involution = sewing@sewing[partners].conj()
    if not np.allclose(involution, np.eye(coeff.shape[-1]), atol=tolerance, rtol=0):
        raise ValueError('Source time-reversal sewing must square to the identity')
    for k in range(len(coeff)):
        reports[k]['sewing_involution_error'] = float(np.max(abs(involution[k]-np.eye(coeff.shape[-1]))))
    for name, values in (('fock', result.full_fock), ('moment', result.sigma_moment),
                         ('self_energy', result.sigma_c_iw)):
        values = np.asarray(values)
        reversed_values = _reverse_source(values, partners, sewing)
        if not np.isfinite(values).all() or not np.allclose(values, reversed_values, atol=tolerance, rtol=0):
            raise ArithmeticError(f'The original source {name} breaks time reversal; do not symmetrize it')
        for k in range(len(coeff)):
            difference = values[k]-reversed_values[k]
            reports[k][name+'_maximum_absolute_error'] = float(np.max(abs(difference)))
            reports[k][name+'_relative_error'] = float(np.linalg.norm(difference)/max(np.linalg.norm(values[k]), 1e-30))
    return partners, sewing, reports


def continue_self_energy(result, *, model=None, tolerance=1e-4, rank_tolerance=1e-12,
                          frequency_cutoff=None):
    """Realize each source-k Sigma_c with PSD residues and its 1/z moment.

    Alternate positive Matsubara frequencies are held out of construction.
    Both training and held-out relative errors in the selected frequency
    range must meet tolerance.
    The shared Hermitian least-squares realization determines real poles
    relative to the returned chemical potential, in Ha. It is a controlled
    finite-data approximation, not a unique analytic continuation. All matrix
    elements stay in the reference MO frame. Different source pole sets use
    their exact union with zero residues elsewhere, without another fit.
    frequency_cutoff optionally selects an explicit positive-frequency range
    in Ha. Choose it from an independent Matsubara convergence study; this
    function does not infer a trusted window. Excluded-frequency errors remain
    in the reports, and the complete self-consistent input is never modified.
    Training/holdout and moment checks concern imaginary-axis data. PSD of
    the constructed measure and a matching electron count do not establish
    real-axis stability or principal-peak assignments.

    Supplying model additionally enforces spinless time reversal. It must
    share the result's source MO frame (align saved states first). Original
    F, M1 and Sigma(z) must satisfy the k/-k sewing relation before a convex
    average of paired residues is allowed. This preserves PSD and the first
    moment and rechecks actual train/holdout errors; it does not repair a
    time-reversal-breaking source or prove unique continuation.
    """
    if not bool(result.converged):
        raise ValueError('Only a converged scGW state may be continued')
    symmetry = None if model is None else _time_reversal_sources(model, result, min(tolerance, 1e-8))
    frequency = np.asarray(result.grid.fermion)
    total = frequency > 0
    selected = total.copy()
    if frequency_cutoff is not None:
        if not np.isfinite(frequency_cutoff) or frequency_cutoff <= 0:
            raise ValueError('The frequency cutoff must be finite and positive in Ha')
        selected &= frequency <= frequency_cutoff
    positive = np.flatnonzero(selected)
    if len(positive) < 4:
        raise ValueError('Continuation requires at least four positive Matsubara frequencies')
    train, holdout = positive[::2], positive[1::2]
    fits, reports = [], []
    for k, values in enumerate(np.asarray(result.sigma_c_iw)):
        try:
            fit = interpolate_causal(1j*frequency[train], values[train],
                moment=np.asarray(result.sigma_moment)[k], tolerance=tolerance,
                rank_tolerance=rank_tolerance)
        except (ValueError, ArithmeticError) as error:
            raise ArithmeticError(f'Uncertified scGW realization at k={k}: {error}') from error
        predicted = evaluate_poles(fit, 1j*frequency[holdout])
        error = float(np.linalg.norm(predicted-values[holdout])
                      /max(np.linalg.norm(values[holdout]), 1e-30))
        excluded = total & ~selected
        excluded_error = None
        if excluded.any():
            excluded_error = float(np.linalg.norm(evaluate_poles(fit, 1j*frequency[excluded])-values[excluded])
                                   /max(np.linalg.norm(values[excluded]), 1e-30))
        report = dict(k_index=k, training_relative_error=fit.relative_error,
            holdout_relative_error=error, moment_relative_error=fit.moment_error,
            source_pole_count=len(fit.poles), pole_count=len(fit.poles),
            realization=fit.diagnostics,
            fit_frequency_count=len(positive), total_positive_frequency_count=int(total.sum()),
            frequency_cutoff_hartree=frequency_cutoff,
            fitted_frequency_min_hartree=float(np.min(frequency[positive])),
            fitted_frequency_max_hartree=float(np.max(frequency[positive])),
            excluded_frequency_relative_error=excluded_error)
        if not fit.converged or error > tolerance or not np.isfinite(error):
            raise ArithmeticError(f'Uncertified scGW continuation fit at k={k}: {report}')
        fits.append(fit)
        reports.append(report)
    if fits and not all(np.array_equal(fit.poles, fits[0].poles) for fit in fits):
        common = np.unique(np.concatenate([fit.poles for fit in fits]))
        for k, fit in enumerate(fits):
            residues = np.zeros((len(common), *fit.residues.shape[1:]), dtype=fit.residues.dtype)
            np.add.at(residues, np.searchsorted(common, fit.poles), fit.residues)
            fits[k] = replace(fit, poles=common, residues=residues)
            reports[k]['pole_count'] = len(common)
    if symmetry is not None:
        partners, sewing, source_reports = symmetry
        original = np.asarray([fit.residues for fit in fits])
        # All partners use the original residues, including non-TRIM pairs.
        residues = .5*(original+_reverse_source(original, partners, sewing))
        reversed_residues = _reverse_source(residues, partners, sewing)
        for k, fit in enumerate(fits):
            updated = replace(fit, residues=residues[k])
            values = np.asarray(result.sigma_c_iw)[k]
            error = float(np.linalg.norm(evaluate_poles(updated, 1j*frequency[train])-values[train])
                          /max(np.linalg.norm(values[train]), 1e-30))
            hold_error = float(np.linalg.norm(evaluate_poles(updated, 1j*frequency[holdout])-values[holdout])
                               /max(np.linalg.norm(values[holdout]), 1e-30))
            moment = np.asarray(result.sigma_moment)[k]
            moment_error = float(np.linalg.norm(residues[k].sum(axis=0)-moment)/max(np.linalg.norm(moment), 1e-30))
            minimum = float(np.linalg.eigvalsh(residues[k]).min()) if residues[k].size else 0.
            residue_scale = max(1., float(abs(residues[k]).max())) if residues[k].size else 1.
            if (not np.isfinite(error+hold_error+moment_error+minimum)
                    or max(error, hold_error) > tolerance or moment_error > 1e-9
                    or minimum < -1e-12*residue_scale):
                raise ArithmeticError('Time-reversal residue average failed the original continuation certificates')
            diagnostic = dict(fit.diagnostics or {})
            diagnostic.update(time_reversal=dict(source_reports[k],
                construction='Convex average R_k and U_k R_-k^T U_k^dagger on the common real pole set',
                original_training_relative_error=fit.relative_error,
                original_holdout_relative_error=reports[k]['holdout_relative_error'],
                residue_relative_error=float(np.linalg.norm(residues[k]-reversed_residues[k])
                    /max(np.linalg.norm(residues[k]), 1e-30)),
                minimum_residue_eigenvalue=minimum),
                unconstrained_realization_scope='HX/Gram diagnostics precede the time-reversal residue average',
                interpolation_relative_error=error, moment_relative_error=moment_error)
            fits[k] = replace(updated, relative_error=error, moment_error=moment_error, diagnostics=diagnostic)
            reports[k].update(training_relative_error=error, holdout_relative_error=hold_error,
                              moment_relative_error=moment_error, realization=diagnostic)
            if (total & ~selected).any():
                excluded = total & ~selected
                reports[k]['excluded_frequency_relative_error'] = float(np.linalg.norm(
                    evaluate_poles(fits[k], 1j*frequency[excluded])-values[excluded])
                    /max(np.linalg.norm(values[excluded]), 1e-30))
    return tuple(fits), reports


def continued_particle_number(result, fits):
    """Spin-summed electron count from all continued Dyson poles and weights.

    The Hermitian auxiliary Hamiltonian integrates the full spectral measure
    with the original beta/mu, including satellite occupation. Counting all
    states or integrating a finite plotting window is not this electron count.
    Matching the count is an integral check, not a principal-peak or real-axis
    stability certificate.
    """
    beta, mu = float(result.grid.beta), float(result.chemical_potential)
    fock = np.asarray(result.full_fock)
    if (not result.converged or len(fits) != len(fock)
            or not len(fits) or not all(fit.converged for fit in fits)):
        raise ValueError('A converged scGW state and certified source fits are required')
    if not np.isfinite(beta) or beta <= 0 or not np.isfinite(mu):
        raise ValueError('Use the finite positive source beta and chemical potential')
    numbers = []
    for matrix, fit in zip(fock, fits):
        energy, weight = auxiliary_spectrum(matrix, fit, mu=mu)
        occupation = np.exp(-np.logaddexp(0., beta*(energy-mu)))
        numbers.append(weight@occupation)
    return float(2*np.mean(numbers))


def _query(mf, model, kpoints, chunk_size, query_bands=None):
    points = np.asarray(kpoints)
    reference, coeff = (mf.get_bands(points, chunk_size=chunk_size)
                        if query_bands is None else query_bands)
    reference, coeff = np.asarray(reference), np.asarray(coeff)
    if (reference.ndim != 2 or len(reference) != len(points)
            or coeff.shape != (len(points), model.coeff.shape[1], reference.shape[-1])
            or not np.isfinite(reference).all() or not np.isfinite(coeff).all()):
        raise ValueError('Query bands must match the requested points and reference AO basis')
    reciprocal = 2*np.pi*np.linalg.inv(np.asarray(model.lattice)).T
    source = np.asarray(model.kpoints)@np.linalg.inv(reciprocal)
    target = points@np.linalg.inv(reciprocal)
    weights = periodic_weights(source, target)
    return np.asarray(reference), np.asarray(coeff), weights


def interpolate_quasiparticles(mf, model, result, kpoints, *, chunk_size=1,
                                query_bands=None):
    """Interpolate a static QP Hamiltonian through the common Bloch AO frame.

    Diagonal G0W0/evGW uses diag(E_QP-E_KS). Mode-A qsGW uses its complete
    rotated Hermitian Hamiltonian. The source nodes are reproduced exactly.
    This is explicitly an interpolation, not a direct GW evaluation at each
    drawing/DOS point. Nonorthogonal frequency-dependent QP vectors cannot
    define a static Hamiltonian; full-matrix G0W0 is therefore rejected here.
    query_bands=(energy,coeff) can reuse one get_bands evaluation for several
    methods; it must come from the same reference and requested point order.
    """
    energy = np.asarray(result.mo_energy)
    computed = np.asarray(getattr(result, 'qp_computed_mask', None))
    if (not bool(result.converged) or energy.shape != np.shape(model.energy)
            or not np.isfinite(energy).all()
            or not np.asarray(result.converged_mask).all()
            or computed.shape != energy.shape or not computed.all()):
        raise ValueError('Interpolate only complete, converged source QP spectra')
    method = result.method
    overlap = np.asarray(mf.inputs.overlap)
    if 'qsgw' in method:
        u = np.asarray(model.coeff).conj().swapaxes(-1, -2)@overlap@np.asarray(result.mo_coeff)
        hamiltonian = (u*energy[:, None, :])@u.conj().swapaxes(-1, -2)
    elif 'diagonal' in method:
        if np.any(np.max(energy[:, :model.nocc], axis=1)
                  >= np.min(energy[:, model.nocc:], axis=1)):
            raise ValueError('Occupied/virtual QP branches cross; sorted static interpolation would change their occupancy labels')
        hamiltonian = np.array([np.diag(row) for row in energy])
    else:
        raise NotImplementedError('Static QP interpolation requires diagonal G0W0/evGW or Hermitian qsGW')
    correction = hamiltonian-np.array([np.diag(row) for row in np.asarray(model.energy)])
    reference, coeff, weights = _query(mf, model, kpoints, chunk_size, query_bands)
    interpolated = interpolate_operator(weights, overlap, model.coeff, correction, coeff)
    matrix = interpolated+np.array([np.diag(row) for row in reference])
    return np.linalg.eigvalsh((matrix+matrix.conj().swapaxes(-1, -2))/2), reference


def interpolate_scgw(mf, model, result, fits, kpoints, frequencies, *, eta=.01,
                      chunk_size=1, query_bands=None, trace_only=False):
    """Full spectral matrices from continued scGW, in Ha^-1.

    PSD pole residues are interpolated with nonnegative weights. The static
    part is the converged interacting F, not a separate effective qsGW
    potential. G is evaluated directly on the retarded axis. QP roots are
    not inferred here: the caller can inspect peak locations or solve the
    same continued Dyson equation separately with explicit branch criteria.
    query_bands reuses a same-reference (energy,coeff) query in point order.
    trace_only retains one scalar per frequency/k for DOS/band maps, rather
    than all nmo^2 entries. The full matrix remains the default output.
    """
    if not result.converged or len(fits) != len(model.energy) or not all(fit.converged for fit in fits):
        raise ValueError('A converged scGW state and certified source fits are required')
    if not all(np.array_equal(fit.poles, fits[0].poles) for fit in fits):
        raise ValueError('All source fits must use the same real pole grid')
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError('Retarded broadening eta must be positive')
    reference, coeff, weights = _query(mf, model, kpoints, chunk_size, query_bands)
    overlap = np.asarray(mf.inputs.overlap)
    diagonal = np.array([np.diag(row) for row in np.asarray(model.energy)])
    correction = interpolate_operator(weights, overlap, model.coeff,
        np.asarray(result.full_fock)-diagonal, coeff)
    fock = correction+np.array([np.diag(row) for row in reference])
    residues = interpolate_operator(weights, overlap, model.coeff,
        np.asarray([fit.residues for fit in fits]), coeff)
    axis = np.asarray(frequencies)
    if axis.ndim != 1 or not np.isfinite(axis).all():
        raise ValueError('Use a finite real frequency vector in Hartree')
    mu = float(result.chemical_potential)
    kernel = 1/(axis[:, None]-mu+1j*eta-fits[0].poles)
    spectral, traces = [], []
    for k in range(len(fock)):
        sigma = np.einsum('wl,lij->wij', kernel, residues[k], optimize=True)
        inverse = (axis+1j*eta)[:, None, None]*np.eye(fock.shape[-1])-fock[k]-sigma
        green = np.linalg.inv(inverse)
        matrix = -(green-green.conj().swapaxes(-1, -2))/(2j*np.pi)
        traces.append(np.trace(matrix, axis1=-2, axis2=-1).real)
        if not trace_only:
            spectral.append(matrix)
    return dict(spectral_matrix=None if trace_only else np.asarray(spectral),
        spectral_trace=np.asarray(traces), fock=fock,
        residues=residues, poles=np.asarray(fits[0].poles),
        reference_energy=reference, coefficients=coeff, weights=weights)


@partial(jax.jit, static_argnames=('root_config',))
def _continued_query(reference, fock, residues, poles, mu, eta, nocc, *, root_config):
    """Reuse the full shared matrix QP selector across numerical query data."""
    from .query import _solve_subspace_qp
    def correlation(omega):
        return jnp.einsum('l,lij->ij', 1/(omega-mu+1j*eta-poles), residues)
    diagonal = jnp.diag(reference)
    return _solve_subspace_qp(reference, fock-diagonal, jnp.zeros_like(diagonal),
        correlation, config=root_config, occupied=jnp.arange(len(reference)) < nocc)


def continued_quasiparticles(data, *, mu, nocc, eta=.001, root_config=None):
    """Solve the continued matrix Dyson equation, with explicit QP masks.

    ``data`` is the interpolated full-Fock/residue representation above.
    The shared matrix QP solver selects finite-eta real-part Dyson roots and
    reports local inverse-slope Z, not physical spectral weights. branch_valid
    certifies an untied nearest-target choice among tested positive-slope
    roots, not complete root enumeration or a principal spectral peak. This
    is not a static-Fock eigenvalue surrogate or a real-axis stability check.
    """
    from ...solvers.nonlinear import ScalarRootConfig
    reference = np.asarray(data['reference_energy'])
    fock, residues = np.asarray(data['fock']), np.asarray(data['residues'])
    poles = jnp.asarray(data['poles'])
    if (reference.ndim != 2 or not 0 < nocc < reference.shape[-1]
            or fock.shape != reference.shape+(reference.shape[-1],)
            or residues.shape != (len(reference), len(poles), reference.shape[-1], reference.shape[-1])
            or not np.isfinite(eta) or eta < 0):
        raise ValueError('Inconsistent continued source/query matrices or broadening')
    config = root_config or ScalarRootConfig(method='hybrid', ftol=1e-8, xtol=1e-8, maxiter=100)
    chemical_potential, broadening, occupation = map(jnp.asarray, (mu, eta, nocc))
    outputs = []
    for k, energy in enumerate(reference):
        outputs.append(_continued_query(jnp.asarray(energy), jnp.asarray(fock[k]),
            jnp.asarray(residues[k]), poles, chemical_potential, broadening, occupation,
            root_config=config))
    return dict(energy=np.asarray([out['roots'] for out in outputs]),
        converged_mask=np.asarray([out['converged'] for out in outputs]),
        residual=np.asarray([out['residual'] for out in outputs]),
        weight=np.asarray([out['weight'] for out in outputs]),
        candidate_count=np.asarray([out['candidate_count'] for out in outputs]),
        branch_valid=np.asarray([out['branch_valid'] for out in outputs]))


__all__ = ['continue_self_energy', 'continued_particle_number',
           'interpolate_quasiparticles', 'interpolate_scgw',
           'continued_quasiparticles']
