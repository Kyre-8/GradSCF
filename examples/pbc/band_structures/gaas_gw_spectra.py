"""Inspect four saved GaAs GW states through the public spectral routines.

Run analyze(directory) after all four source calculations have converged.
The moment-anchored Hermitian realization has PSD residues and separate
training/holdout certificates. Check numerical rank, Matsubara range and
frequency convergence independently. Imaginary-axis, PSD and electron-count
checks do not establish real-axis stability or unique continuation. The scGW
panel shows the spectral function. Additional selected finite-eta ReSigma
roots and gaps are diagnostics, not certified principal peaks; their local Z
values are not physical spectral weights. No static-Fock
or failed-QP fallback is used. All four DOS curves use Lorentzian eta=.1 eV.
The three static QP curves count one state per band/spin; scGW includes satellites.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf.gw.pbc import build_periodic_model
from gradscf.gw.pbc.interpolation import align_source, electronic_dos, band_gaps
from gradscf.gw.pbc.spectrum import (continue_self_energy, interpolate_quasiparticles,
    interpolate_scgw, continued_quasiparticles, continued_particle_number)
from gradscf_tools.gaas_gw_plot import plot_comparison
from examples.pbc.band_structures.gaas_self_consistent_gw import MAX_MEMORY_BYTES, reference
from examples.pbc.band_structures.systems import structure


EV_PER_HARTREE = 27.211386245988
METHODS = ('g0w0', 'evgw', 'qsgw', 'scgw')


def _close(actual, expected, label, tolerance=1e-8):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape or not np.isfinite(actual).all()
            or not np.allclose(actual, expected, atol=tolerance, rtol=0)):
        raise ValueError(f'Inconsistent saved/reconstructed {label}')


def analyze(directory, *, path_points=121, dos_kmesh=(4, 4, 4),
            axis_ev=None, fit_tolerance=1e-4, spectral_eta_ev=.1,
            particle_tolerance=1e-3, rank_tolerance=1e-12,
            frequency_cutoff=None):
    """Save comparison arrays and figures after the stated numerical checks.

    Plotting and uniform BZ samples interpolate source operators; they do not
    increase screening/source resolution. axis_ev uses the common PBE-VBM
    zero and defaults to [-50,200] eV. All 14 finite-basis bands are retained;
    the drawing shows [-15,15] eV around the gap, without truncating DOS inputs.
    Rank/range/eta stability requires separate runs with explicit controls;
    training, holdout, first-moment and electron-count checks all apply.
    These checks do not certify real-axis stability. scGW lines and reported
    gaps use selected finite-eta ReSigma Dyson roots, not principal-peak or
    spectral-onset assignments. Local inverse-slope Z is a root diagnostic,
    not a physical spectral weight.
    A unique root is not required for the retarded spectral map or DOS.
    Uncertified roots remain NaN, their raw candidates and masks are saved,
    and a gap is unavailable unless its own sample has complete roots.
    A frequency_cutoff (Ha) must come from a separate frequency-convergence
    check. The full input is retained and excluded-frequency errors reported.
    """
    directory = Path(directory)
    stored, reports = {}, {}
    for method in METHODS:
        reports[method] = json.loads((directory/f'{method}.json').read_text())
        with np.load(directory/f'{method}.npz', allow_pickle=False) as saved:
            stored[method] = {name: saved[name].copy() for name in saved.files}
        if not reports[method]['converged']:
            raise ValueError(f'{method} source calculation is not converged')
    common = reports['g0w0']
    controls = ('source_kmesh', 'fft_mesh', 'basis', 'pseudo', 'xc', 'soc',
                'fc', 'exxdiv', 'screening_cutoff_rydberg')
    for method in METHODS:
        if (any(reports[method][name] != common[name] for name in controls)
                or reports[method]['source_model']['compression_tolerance'] != common['source_model']['compression_tolerance']):
            raise ValueError(f'{method} uses different physical source controls')
    if (any(reports[method]['eta_hartree'] != common['eta_hartree'] for method in METHODS[:3])
            or reports['scgw']['eta_hartree'] is not None):
        raise ValueError('Match the CD broadenings; Matsubara scGW must have no imaginary-axis eta')
    if common['basis'] != 'gth-szv' or common['pseudo'] != 'gth-pbe' or common['soc'] or common['fc']:
        raise ValueError('This example requires the declared scalar GaAs/SZV fc=False model')
    mf = reference(source_kmesh=tuple(common['source_kmesh']), mesh=tuple(common['fft_mesh']),
                   exxdiv=common['exxdiv'])
    model = build_periodic_model(mf, tolerance=common['source_model']['compression_tolerance'],
        max_memory_bytes=MAX_MEMORY_BYTES,
        fc=False, screening_cutoff_rydberg=common['screening_cutoff_rydberg'])
    atoms, path_name, _ = structure('gaas')
    overlap, nocc = np.asarray(mf.inputs.overlap), model.nocc
    geometry = dict(lattice_bohr=model.lattice, coordinates_bohr=mf.cell.coords,
                    atomic_numbers=atoms.get_atomic_numbers(), valence_charges=mf.cell.charges)
    restored, verification = {}, {}
    for method, data in stored.items():
        for name, expected in geometry.items():
            _close(data[name], expected, name)
        _close(data['source_kpoints'], model.kpoints, 'source k points')
        _close(data['overlap'], overlap, 'AO overlap')
        _close(data['reference_energy'], model.energy, 'PBE source energies')
        _close(data['reference_density'], mf.result.density_spin, 'PBE source density')
        if int(data['nocc']) != nocc:
            raise ValueError(f'{method} has a different occupied-band count')
        verification[method] = dict(source_energy_max_error_hartree=float(np.max(
            abs(data['reference_energy']-np.asarray(model.energy)))),
            source_density_max_error=float(np.max(abs(data['reference_density']-np.asarray(mf.result.density_spin)))))
        state = SimpleNamespace(converged=True, method=reports[method]['method'])
        if method == 'scgw':
            state.sigma_c_iw, state.sigma_moment, state.full_fock = align_source(model, overlap,
                data['reference_coeff'], data['reference_energy'],
                data['sigma_c_iw'], data['sigma_moment'], data['full_fock'])
            state.grid = SimpleNamespace(fermion=data['fermion'],
                beta=reports[method]['beta_hartree_inverse'])
            state.chemical_potential = float(data['chemical_potential'])
        else:
            if not data['converged_mask'].all() or not np.isfinite(data['qp_energy']).all():
                raise ArithmeticError(f'{method} has incomplete source QP coverage')
            if 'qp_branch_valid' in data and not data['qp_branch_valid'].all():
                raise ArithmeticError(f'{method} has uncertified source branches')
            if 'qp_weight' in data and (not np.isfinite(data['qp_weight']).all() or np.min(data['qp_weight']) <= 0):
                raise ArithmeticError(f'{method} has invalid source QP slopes')
            state.mo_energy, state.mo_coeff = data['qp_energy'], data['qp_coeff']
            state.converged_mask = data['converged_mask']
            state.qp_computed_mask = data['qp_computed_mask']
            if method == 'qsgw':
                align_source(model, overlap, data['reference_coeff'], data['reference_energy'])
            else:
                diagonal = np.array([np.diag(row) for row in data['qp_energy']])
                aligned, = align_source(model, overlap, data['reference_coeff'], data['reference_energy'], diagonal)
                values = np.diagonal(aligned, axis1=-2, axis2=-1).real
                _close(aligned, np.array([np.diag(row) for row in values]), 'diagonal QP gauge')
                state.mo_energy = values
        restored[method] = state
    path = atoms.cell.bandpath(path_name, npoints=path_points, pbc=atoms.pbc)
    distance, ticks, labels = path.get_linear_kpoint_axis()
    # Keep the physical path continuous; interpolation wraps its own weights.
    path_k = path.kpts@np.asarray(mf.cell.reciprocal_vectors())
    dos_k = np.asarray(mf.cell.make_kpts(dos_kmesh))
    points = np.concatenate((path_k, dos_k, np.zeros((1, 3))))
    npath, ndos = len(path_k), len(dos_k)
    bz = slice(npath, npath+ndos)
    query_bands = tuple(np.asarray(value) for value in mf.get_bands(points, chunk_size=1))
    pbe = query_bands[0]
    zero = float(np.max(pbe[:npath+ndos, :nocc]))
    axis_ev = np.linspace(-50., 200., 5001) if axis_ev is None else np.asarray(axis_ev, float)
    if axis_ev.ndim != 1 or len(axis_ev) < 2 or np.any(np.diff(axis_ev) <= 0) or not np.isfinite(axis_ev).all():
        raise ValueError('Use a finite increasing energy axis in eV')
    fits, fit_reports = continue_self_energy(restored['scgw'], model=model,
        tolerance=fit_tolerance, rank_tolerance=rank_tolerance,
        frequency_cutoff=frequency_cutoff)
    electrons = continued_particle_number(restored['scgw'], fits)
    if not np.isfinite(particle_tolerance) or particle_tolerance <= 0:
        raise ValueError('particle_tolerance must be positive in electrons/cell')
    if abs(electrons-2*nocc) > particle_tolerance:
        raise ArithmeticError(f'Continued scGW electron count {electrons:.9f} differs from {2*nocc}')
    energy, arrays = {'pbe': pbe}, {}
    for method in METHODS[:3]:
        energy[method], _ = interpolate_quasiparticles(mf, model, restored[method], points,
                                                       query_bands=query_bands)
    spectral = interpolate_scgw(mf, model, restored['scgw'], fits, points,
        zero+axis_ev/EV_PER_HARTREE, eta=spectral_eta_ev/EV_PER_HARTREE,
        query_bands=query_bands, trace_only=True)
    qp = continued_quasiparticles(spectral, mu=restored['scgw'].chemical_potential, nocc=nocc,
                                  eta=common['eta_hartree'])
    qp_valid = qp['converged_mask'] & qp['branch_valid'] & np.isfinite(qp['energy'])
    # A retarded spectrum need not have a unique quasiparticle for every
    # reference orbital. Preserve its diagnostics and leave such roots absent.
    raw_qp_energy = qp['energy'].copy()
    qp['energy'] = energy['scgw'] = np.where(qp_valid, raw_qp_energy, np.nan)
    qp['certified_mask'] = qp_valid
    for method, values in energy.items():
        if method != 'scgw' and not np.isfinite(values).all():
            raise ArithmeticError(f'{method} has nonfinite query bands')
    finite_energy = np.concatenate([value[np.isfinite(value)] for value in energy.values()])
    limits = [float(5*np.floor((finite_energy.min()-zero)*EV_PER_HARTREE/5)),
              float(5*np.ceil((finite_energy.max()-zero)*EV_PER_HARTREE/5))]
    if axis_ev[0] > limits[0] or axis_ev[-1] < limits[1]:
        raise ValueError('Widen axis_ev to include every retained finite-basis QP band')
    gaps = {}
    for method, values in energy.items():
        sampled = band_gaps(values[bz], nocc=nocc) if np.isfinite(values[bz]).all() else None
        gamma = band_gaps(values[[-1]], nocc=nocc) if np.isfinite(values[-1]).all() else None
        gaps[method] = dict(gamma_direct_ev=None if gamma is None else gamma['sampled_gap_hartree']*EV_PER_HARTREE,
            sampled_gap_ev=None if sampled is None else sampled['sampled_gap_hartree']*EV_PER_HARTREE,
            minimum_direct_ev=None if sampled is None else sampled['minimum_direct_gap_hartree']*EV_PER_HARTREE)
        if method != 'pbe':
            arrays[method+'_bands'] = (values[:npath]-zero)*EV_PER_HARTREE
            if method != 'scgw':
                arrays[method+'_dos'] = electronic_dos(axis_ev, (values[bz]-zero)*EV_PER_HARTREE,
                    sigma=spectral_eta_ev, smearing='lorentzian')
    trace = 2*spectral['spectral_trace']/EV_PER_HARTREE
    if not np.isfinite(trace).all() or np.min(trace) < -1e-9:
        raise ArithmeticError('The retarded scGW spectral trace is nonfinite or negative')
    arrays.update(scgw_dos=trace[bz].mean(axis=0), scgw_spectral=trace[:npath],
        axis_ev=axis_ev, path_distance=distance, path_ticks=ticks, path_labels=np.asarray(labels),
        pbe_bands=(pbe[:npath]-zero)*EV_PER_HARTREE, query_kpoints=points, dos_kpoints=dos_k,
        scgw_poles=fits[0].poles, scgw_source_residues=np.asarray([fit.residues for fit in fits]),
        scgw_qp_raw_energy=raw_qp_energy,
        **{'scgw_qp_'+name: value for name, value in qp.items()})
    for method in METHODS:
        for metric, change in (('sampled_gap_ev', 'sampled_gap_change_ev'),
                               ('gamma_direct_ev', 'gamma_gap_change_ev'),
                               ('minimum_direct_ev', 'minimum_direct_gap_change_ev')):
            value = gaps[method][metric]
            gaps[method][change] = None if value is None else value-gaps['pbe'][metric]
        for name in ('qp_weight', 'qp_branch_valid', 'qp_candidate_count'):
            if name in stored[method]:
                arrays[method+'_source_'+name] = stored[method][name]
    report = {name: common[name] for name in controls}
    report.update(nocc=nocc, display_bands=list(range(pbe.shape[1])), gaps=gaps,
        eta_hartree=common['eta_hartree'], scgw_imaginary_axis_eta_hartree=None,
        scgw_qp_root_eta_hartree=common['eta_hartree'],
        scgw_band_definition='Selected finite-eta ReSigma matrix Dyson roots; scGW gaps use these roots, not spectral-onset gaps or certified principal peaks',
        scgw_qp_weight_definition='Formal local finite-eta inverse residual slope Z; not physical spectral weights or normalized peak intensities',
        scgw_qp_branch_valid_definition='Untied nearest-target selection among tested positive-slope roots; not complete root enumeration or a principal-peak certificate',
        scgw_qp_coverage=dict(requested=int(qp_valid.size), certified=int(qp_valid.sum()),
            incomplete_query_indices=np.flatnonzero(~qp_valid.all(axis=1)).tolist(),
            gamma_complete=bool(qp_valid[-1].all()), dos_complete=bool(qp_valid[bz].all()),
            gap_policy='Report each gap only with complete certified roots for its own sample; missing roots stay NaN and unavailable gaps/changes are null. The retarded spectral map and DOS do not require a unique QP root.'),
        retained_band_count=pbe.shape[1], spin_degeneracy=2, dos_units='states/cell/eV',
        method_discretization={method: dict(nw=reports[method]['nw'],
            axis='Matsubara' if method == 'scgw' else 'Legendre CD') for method in METHODS},
        scgw_time_quadrature=reports['scgw']['time_quadrature'],
        scgw_time_oversampling=reports['scgw']['time_oversampling'],
        scgw_ntau=reports['scgw']['ntau'],
        beta_hartree_inverse=reports['scgw']['beta_hartree_inverse'], dos_kmesh=list(dos_kmesh),
        dos_broadening_ev=spectral_eta_ev, spectral_eta_ev=spectral_eta_ev, energy_zero_hartree=zero,
        energy_limits_ev=[-15., 15.], continuation_fit=fit_reports,
        continuation_tolerance=fit_tolerance, continuation_rank_tolerance=rank_tolerance,
        continuation_frequency_cutoff_hartree=frequency_cutoff,
        continuation_poles_hartree=np.asarray(fits[0].poles).tolist(),
        continuation_checks_scope='Selected imaginary-axis train/holdout and first moment, PSD of the constructed measure, and continued electron count; these do not certify real-axis stability',
        continuation_stability='A single run does not establish real-axis stability; separate numerical-rank, Matsubara-range, eta and forward-grid comparisons are required',
        scgw_interpretation_note='scGW lines and gap bars: selected finite-η ReΣ Dyson roots, not certified principal peaks. Local Z is not physical spectral weight. Imaginary-axis/N/PSD checks do not establish real-axis stability.',
        source_verification=verification, theoretical_states_per_cell=2*pbe.shape[1],
        continued_electrons_per_cell=electrons, target_electrons_per_cell=2*nocc,
        continuation_particle_tolerance=particle_tolerance,
        dos_window_integral={method: float(np.trapezoid(arrays[method+'_dos'], axis_ev)) for method in METHODS},
        dos_definitions=dict(g0w0='Lorentzian QP DOS; unit weight per band/spin',
            evgw='Lorentzian QP DOS; unit weight per band/spin', qsgw='Lorentzian QP DOS; unit weight per band/spin',
            scgw='Spin-summed retarded interacting spectral trace; includes satellite weight'),
        finite_model_note='Body-only fc=False; static QP curves and continued interacting scGW have different DOS definitions.')
    np.savez_compressed(directory/'comparison.npz', **arrays)
    (directory/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    return plot_comparison(directory)


if __name__ == '__main__':
    analyze(Path('artifacts/gaas_gw_comparison'),
            axis_ev=np.linspace(-250., 250., 10001), frequency_cutoff=2.01)
