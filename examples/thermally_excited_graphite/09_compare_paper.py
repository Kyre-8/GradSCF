"""Compare a synthesized unresolved Gamma peak with reported EELS peaks.

Run after 01; this only postprocesses saved modes. Run on the same CPU float64
environment as the calculations. Edit CASES to add a separately archived run;
never replace the pilot files. No CLI, empirical energy shift, new q point,
electronic calculation, or experimental curve reconstruction is used here.
"""
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import os

import jax
import numpy as np
from scipy.constants import physical_constants
from scipy.linalg import eigh
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares, minimize_scalar
from scipy.signal import find_peaks
from scipy.special import voigt_profile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gradscf.pbc.vibrational import mode_scattering_weights, one_phonon_spectrum
import gradscf.pbc.vibrational as vibrational
from settings import OUTPUT, HARTREE_TO_MEV

jax.config.update('jax_enable_x64', True)
START = time.perf_counter()
REFERENCE_PATH = Path(__file__).with_name('experiment_reference.json')
REFERENCE = json.loads(REFERENCE_PATH.read_text())
# Keep the earlier PBE artifact directory intact when adding r2SCAN cases.
DESTINATION = OUTPUT/('r2scan_comparison' if any((OUTPUT/'gamma_precision').glob('r2scan*'))
                      else 'paper_comparison')
DESTINATION.mkdir(parents=True, exist_ok=True)
CASES = [('pilot', OUTPUT, 'unconverged basis/k/FFT; fixed geometry')]
OMITTED_CASES = []
# Optional independent series: each directory must contain a 300 K reference.
for folder in sorted((OUTPUT/'gamma_precision').glob('*')):
    if folder.is_dir() and (folder/'gamma_300K.npz').exists():
        CASES.append((folder.name, folder, 'numerical refinement; convergence not presumed'))
    elif folder.is_dir():
        OMITTED_CASES.append(dict(case=folder.name, reason='Missing gamma_300K.npz baseline',
                                  available_files=[p.name for p in sorted(folder.glob('gamma_*K.npz'))]))

FWHM = REFERENCE['instrument']['model_fwhm_meV']
# Fit/sampling windows follow all actual modes, with generous Gaussian tails.
# The analytic one-line check at 180 meV also stays inside this common window.
mode_limits = [180.]
for _, folder, _ in CASES:
    for path in folder.glob('gamma_*K.npz'):
        with np.load(path) as archive:
            mode_limits.extend((archive['energies']*HARTREE_TO_MEV).tolist())
FIT_BOUNDS = (max(1., min(mode_limits)-25.), max(mode_limits)+25.)
GRID = np.arange(max(.01, np.floor(min(mode_limits)-50.)),
                 np.ceil(max(mode_limits)+50.)+.005, .01)
EXPERIMENT = {p['comparison_temperature_K']: p for p in REFERENCE['gamma_static_peak']}
EXPERIMENT_SHIFT = EXPERIMENT[1377.]['energy_meV']-EXPERIMENT[300.]['energy_meV']
# Formal propagation ONLY under independent endpoint fit errors. No model error
# estimate or confidence-level interpretation is supplied by this calculation.
EXPERIMENT_SHIFT_ERROR = np.hypot(EXPERIMENT[300.]['energy_reported_uncertainty_meV'],
                                  EXPERIMENT[1377.]['energy_reported_uncertainty_meV'])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def peak_properties(energy, intensity):
    """Maximum and half-height crossings of one resolved *envelope*."""
    spline = CubicSpline(energy, intensity)
    index = int(np.argmax(intensity))
    assert 0 < index < len(energy)-1, 'Peak lies outside the sampled interval.'
    peak = minimize_scalar(lambda x: -float(spline(x)),
                           bounds=(energy[index-1], energy[index+1]), method='bounded',
                           options={'xatol': 1e-10})
    assert peak.success
    half = -peak.fun/2
    crossings = np.where(np.diff(np.sign(intensity-half)) != 0)[0]
    assert len(crossings) == 2, 'Envelope is not one peak with two half-height crossings.'
    left, right = [brentq(lambda x: float(spline(x))-half, energy[i], energy[i+1],
                          xtol=1e-11) for i in crossings]
    return float(peak.x), float(right-left)


def peak_diagnostics(energy, intensity):
    """Count resolved maxima without assuming an unresolved single peak.

    The 1e-8 relative height/prominence cutoff only rejects numerical tails;
    it is not an experimental detectability threshold. Heights are not areas.
    """
    assert np.all(np.isfinite(intensity)) and np.max(intensity) > 0
    normalized = intensity/np.max(intensity)
    indices, _ = find_peaks(normalized, height=1e-8, prominence=1e-8)
    assert len(indices), 'No interior maximum: inspect the energy interval.'
    spline = CubicSpline(energy, normalized)
    peaks = []
    for index in indices:
        result = minimize_scalar(lambda x: -float(spline(x)),
            bounds=(energy[index-1], energy[index+1]), method='bounded',
            options={'xatol': 1e-10})
        assert result.success
        peaks.append(dict(energy_meV=float(result.x), relative_height=float(-result.fun)))
    crossings = np.where(np.diff(np.sign(normalized-.5)) != 0)[0]
    return dict(local_peaks=peaks, local_peak_count=len(peaks),
                half_height_crossing_count=len(crossings), numerical_peak_threshold=1e-8,
                single_peak_eligible=len(peaks) == 1 and len(crossings) == 2)


def rigid_translation_diagnostics(data):
    """Read-only Rayleigh quotients in the saved projected displacement space.

    Cartesian translations have Euclidean norm one; the actual projected
    electron-mass metric supplies their inertial denominator. No ASR is used.
    Eigenvector overlaps refer to the mass metric and must be summed across
    degenerate pairs before interpreting a mode-family assignment.
    """
    basis = np.asarray(data['directions']).reshape(len(data['directions']), -1)
    ratio = physical_constants['atomic mass constant'][0]/physical_constants['electron mass'][0]
    mass = np.repeat(np.asarray(data['masses'])*ratio, 3)
    metric = (basis*mass)@basis.T
    phi = np.asarray(data['force_constants'])
    phi = (phi+phi.conj().T)/2
    values, vectors = eigh(phi, metric)
    energies = np.sign(values)*np.sqrt(np.abs(values))*HARTREE_TO_MEV
    assert np.allclose(energies, np.asarray(data['all_inplane_energies'])*HARTREE_TO_MEV,
                       rtol=0., atol=1e-6), 'Saved Phi and signed frequencies differ.'
    result = dict(signed_energies_meV=energies.tolist(), projected_mass_metric_electron_masses=metric.tolist(),
                  asr_applied=False, full_space_operator_available=False, translations=[])
    for axis, name in enumerate(('x', 'y')):
        translation = np.zeros((len(data['masses']), 3))
        translation[:, axis] = 1/np.sqrt(len(data['masses']))
        coefficients = np.linalg.lstsq(basis.T, translation.ravel(), rcond=None)[0]
        assert np.linalg.norm(basis.T@coefficients-translation.ravel()) < 1e-12
        curvature = float(np.real(coefficients@phi@coefficients))
        inertia = float(coefficients@metric@coefficients)
        quotient = curvature/inertia
        overlaps = np.abs(vectors.conj().T@metric@coefficients)**2/inertia
        assert np.isclose(np.sum(overlaps), 1., atol=1e-12)
        result['translations'].append(dict(axis=name, cartesian_euclidean_norm=1.,
            curvature_Hartree_per_Bohr2=curvature, inertia_electron_masses=inertia,
            rayleigh_omega_squared_Hartree2=quotient,
            signed_energy_meV=float(np.sign(quotient)*np.sqrt(abs(quotient))*HARTREE_TO_MEV),
            mass_metric_eigenmode_overlaps=overlaps.tolist(),
            projected_mass_weighted_residual_Hartree2=float(np.sqrt(np.sum(overlaps*(values-quotient)**2)))))
    return result


def fitted_voigt(energy, intensity):
    """Uniform least squares, free area/center/sigma/gamma, zero background.

    The spectrum is noiseless: fitted parameter covariance is deliberately not
    interpreted as experimental or numerical uncertainty. The freely fitted
    Lorentzian parameter is a shape descriptor, never a computed lifetime.
    """
    selected = (energy >= FIT_BOUNDS[0]) & (energy <= FIT_BOUNDS[1])
    x, y = energy[selected], intensity[selected]/np.max(intensity)

    def profile(parameters, coordinates):
        area, center, sigma, gamma = parameters
        return area*voigt_profile(coordinates-center, sigma, gamma)

    result = least_squares(lambda p: profile(p, x)-y,
        [np.trapezoid(y, x), float(x[np.argmax(y)]), FWHM/np.sqrt(8*np.log(2)), .01],
        bounds=([0., FIT_BOUNDS[0], .01, 0.], [1000., FIT_BOUNDS[1], 40., 40.]),
        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1000)
    assert result.success, result.message
    _, center, sigma, gamma = result.x
    assert FIT_BOUNDS[0]+FWHM < center < FIT_BOUNDS[1]-FWHM, 'Voigt center too near fit boundary.'
    assert max(y[0], y[-1]) < 1e-4, 'Fit interval truncates significant spectral intensity.'
    half = voigt_profile(0., sigma, gamma)/2
    width = 2*brentq(lambda x: voigt_profile(x, sigma, gamma)-half, 0., 200.)
    return dict(voigt_center_meV=float(center), voigt_fwhm_meV=float(width),
                voigt_sigma_meV=float(sigma), voigt_gamma_meV=float(gamma),
                voigt_normalized_rms=float(np.sqrt(np.mean(result.fun**2))),
                voigt_center_error_meV=None), profile(result.x, energy)


rows, spectra, raw_spectra, provenance, excluded_cases = [], {}, {}, [], []
translation_records = []
translation_checks = []
for case, folder, status in CASES:
    files = sorted(folder.glob('gamma_*K.npz'))
    with np.load(folder/'gamma_300K.npz') as archive:
        cold = dict(archive)
    case_settings = json.loads(str(cold['settings']))
    check_path = folder/'translation_check.json'
    if check_path.exists():
        exported_name = f'{case}_translation_check.json'
        shutil.copyfile(check_path, DESTINATION/exported_name)
        translation_checks.append(dict(case=case, source_file=str(check_path), source_sha256=digest(check_path),
                                       exported_file=exported_name, data=json.loads(check_path.read_text())))
    reciprocal = 2*np.pi*np.linalg.inv(cold['lattice']).T
    momentum = reciprocal[0]  # Same full Q = first reciprocal vector as 04.
    form_factors = np.ones(len(cold['masses']))
    provenance.append(dict(case=case, status=status, settings=case_settings,
        momentum_bohr_inverse=momentum.tolist(), form_factors=form_factors.tolist(),
        momentum_acceptance_integrated=False))
    for path in files:
        with np.load(path) as archive:
            data = dict(archive)
        temperature = float(data['electron_temperature'])
        assert json.loads(str(data['settings'])) == case_settings
        assert np.array_equal(data['lattice'], cold['lattice'])
        assert np.array_equal(data['coordinates'], cold['coordinates'])
        assert float(data['relative_subspace_residual']) < 1e-3
        translation_records.append(dict(case=case, temperature_K=temperature, settings=case_settings,
            source_file=str(path), source_sha256=digest(path), **rigid_translation_diagnostics(data)))
        for mechanism, modes_data in [('electronic_plus_bose', data), ('bose_only', cold)]:
            energies = np.asarray(modes_data['energies'])
            assert np.all(energies > 0)
            weights = mode_scattering_weights(momentum, modes_data['coordinates'],
                modes_data['modes'], modes_data['masses'], form_factors=form_factors)
            strengths = np.asarray(weights)/(2*energies)
            spectrum = one_phonon_spectrum(energies, strengths, GRID/HARTREE_TO_MEV,
                temperature=temperature, fwhm=FWHM/HARTREE_TO_MEV)
            intensity = np.asarray(spectrum.loss)/HARTREE_TO_MEV
            diagnostics = peak_diagnostics(GRID, intensity)
            fitted = dict.fromkeys(('voigt_center_meV', 'voigt_fwhm_meV', 'voigt_sigma_meV',
                                   'voigt_gamma_meV', 'voigt_normalized_rms', 'voigt_center_error_meV'))
            maximum = width = center_difference = width_difference = profile = None
            if diagnostics['single_peak_eligible']:
                maximum, width = peak_properties(GRID, intensity)
                fitted, profile = fitted_voigt(GRID, intensity)
                # This checks postprocessing sampling, not SCF accuracy.
                coarse_center, coarse_width = peak_properties(GRID[::2], intensity[::2])
                center_difference, width_difference = coarse_center-maximum, coarse_width-width
                assert abs(center_difference) < 1e-5 and abs(width_difference) < 1e-5
            # Finite-T mode areas come from the shared spectrum library, not a
            # second Bose implementation or an assignment of peaks to symmetry.
            mode_areas = []
            for mode in range(len(energies)):
                contribution = one_phonon_spectrum(energies[mode:mode+1], strengths[mode:mode+1],
                    GRID/HARTREE_TO_MEV, temperature=temperature, fwhm=FWHM/HARTREE_TO_MEV)
                mode_areas.append(float(np.trapezoid(np.asarray(contribution.loss)/HARTREE_TO_MEV, GRID)))
            mode_areas = np.asarray(mode_areas)
            assert np.isclose(np.sum(mode_areas), np.trapezoid(intensity, GRID), rtol=1e-12)
            experiment = EXPERIMENT.get(temperature)
            row = dict(case=case, numerical_status=status, mechanism=mechanism,
                temperature_K=temperature, maximum_meV=maximum, envelope_fwhm_meV=width,
                **fitted, eigenvalues_meV=(energies*HARTREE_TO_MEV).tolist(),
                signed_inplane_energies_meV=(np.asarray(modes_data['all_inplane_energies'])*HARTREE_TO_MEV).tolist(),
                negative_inplane_mode_count=int(np.count_nonzero(modes_data['all_inplane_energies'] < 0)),
                relative_zero_temperature_strengths=(strengths/np.sum(strengths)).tolist(),
                relative_loss_mode_areas=(mode_areas/np.sum(mode_areas)).tolist(),
                peak_diagnostics=diagnostics,
                single_peak_comparison_eligible=diagnostics['single_peak_eligible'],
                experiment_energy_meV=None if experiment is None else experiment['energy_meV'],
                experiment_reported_error_meV=None if experiment is None else experiment['energy_reported_uncertainty_meV'],
                fitted_center_bias_meV=(None if experiment is None or fitted['voigt_center_meV'] is None
                                       else fitted['voigt_center_meV']-experiment['energy_meV']),
                coarse_grid_center_difference_meV=center_difference,
                coarse_grid_width_difference_meV=width_difference,
                source_file=str(path), source_sha256=digest(path),
                mode_source_file=str(path if mechanism == 'electronic_plus_bose' else folder/'gamma_300K.npz'),
                mode_source_sha256=digest(path if mechanism == 'electronic_plus_bose' else folder/'gamma_300K.npz'))
            rows.append(row)
            spectra[(case, mechanism, temperature)] = (intensity/np.max(intensity), profile)
            raw_spectra[(case, mechanism, temperature)] = intensity
    case_rows = [row for row in rows if row['case'] == case]
    eligible = all(row['single_peak_comparison_eligible'] for row in case_rows)
    if not eligible:
        reason = 'Multiple resolved maxima or non-single half-height envelope; single-Voigt experimental comparison is not applicable.'
        excluded_cases.append(dict(case=case, reason=reason,
            failing_spectra=[dict(temperature_K=r['temperature_K'], mechanism=r['mechanism'],
                                 diagnostics=r['peak_diagnostics']) for r in case_rows
                             if not r['single_peak_comparison_eligible']]))
        # A mixed series is excluded as a case, preventing a partial single-peak
        # fit from being compared with another temperature's multiple peaks.
        for row in case_rows:
            row['single_peak_comparison_eligible'] = False
            row['fitted_center_bias_meV'] = None
            for key in fitted:
                row[key] = None
    for mechanism in ('electronic_plus_bose', 'bose_only'):
        series = [row for row in rows if row['case'] == case and row['mechanism'] == mechanism]
        baseline = next(row for row in series if row['temperature_K'] == 300.)
        for row in series:
            if not eligible:
                for key in ('voigt_shift_from_300K_meV', 'maximum_shift_from_300K_meV',
                            'width_change_from_300K_meV', 'shift_mismatch_meV'):
                    row[key] = None
                continue
            row['voigt_shift_from_300K_meV'] = row['voigt_center_meV']-baseline['voigt_center_meV']
            row['maximum_shift_from_300K_meV'] = row['maximum_meV']-baseline['maximum_meV']
            row['width_change_from_300K_meV'] = row['envelope_fwhm_meV']-baseline['envelope_fwhm_meV']
            row['shift_mismatch_meV'] = (row['voigt_shift_from_300K_meV']-EXPERIMENT_SHIFT
                                         if row['temperature_K'] == 1377. else None)
    present = sorted({r['temperature_K'] for r in rows if r['case'] == case})
    provenance[-1]['present_temperatures_K'] = present
    provenance[-1]['missing_endpoint_temperatures_K'] = [t for t in (300., 1377.) if t not in present]
    provenance[-1]['single_peak_comparison_eligible'] = eligible
    provenance[-1]['thermal_comparison_available'] = eligible and all(t in present for t in (300., 1377.))

# A one-line analytic limit checks the actual measurement helpers and the
# library spectrum together; no new model implementation is introduced.
single = one_phonon_spectrum(np.array([180.])/HARTREE_TO_MEV, np.ones(1),
    GRID/HARTREE_TO_MEV, temperature=300., fwhm=FWHM/HARTREE_TO_MEV)
single_center, single_width = peak_properties(GRID, np.asarray(single.loss))
single_fit, _ = fitted_voigt(GRID, np.asarray(single.loss))
assert abs(single_center-180.) < 1e-5 and abs(single_width-FWHM) < 1e-5
assert abs(single_fit['voigt_center_meV']-180.) < 1e-5
assert abs(single_fit['voigt_fwhm_meV']-FWHM) < 1e-4
assert peak_diagnostics(GRID, np.asarray(single.loss))['single_peak_eligible']
# Regression: even a second peak below the global half-height excludes a
# single-peak comparison. These synthetic checks are not experimental curves.
multipeak_checks = []
for strengths in (np.ones(2), np.array([1., .05])):
    check = one_phonon_spectrum(np.array([165., 205.])/HARTREE_TO_MEV, strengths,
        GRID/HARTREE_TO_MEV, temperature=300., fwhm=FWHM/HARTREE_TO_MEV)
    diagnostic = peak_diagnostics(GRID, np.asarray(check.loss))
    assert diagnostic['local_peak_count'] == 2 and not diagnostic['single_peak_eligible']
    multipeak_checks.append(diagnostic)

limitations = [
    'Fixed representative lattice; no thermal expansion or geometry relaxation.',
    'Only positive optical modes are synthesized; signed low in-plane modes are retained and no bulk stability is claimed.',
    'No bulk cubic/quartic phonon self-energy or anharmonic lifetime.',
    'No finite experimental momentum acceptance, screening, detector geometry or multiple scattering.',
    'Unit form factors and Gaussian 11 meV instrumental FWHM are declared model assumptions.',
    'Apparent widths contain instrument and unresolved mode mixing only; fitted Lorentzian is not a lifetime.',
    'Experimental raw spectra/width endpoints are unavailable here; no synthetic experimental curves or error bars.',
    'Model fit has zero background and uniform residual weights; it does not reproduce the full experimental fitting pipeline.',
    'Model errors are unquantified; reported formal experimental errors exclude model and calibration systematics.',
    'Cases with resolved multiple peaks are retained as raw spectra and excluded from single-Voigt experimental bias/shift comparisons.',
]
summary = dict(source_reference=REFERENCE, cases=provenance, omitted_cases=OMITTED_CASES,
    excluded_from_single_peak_comparison=excluded_cases, records=rows,
    experimental_shift_meV=EXPERIMENT_SHIFT,
    experimental_formal_shift_error_meV=float(EXPERIMENT_SHIFT_ERROR),
    experimental_error_assumption='Independent endpoint fit errors; covariance and confidence coverage unspecified. No propagation of temperature errors into energy errors.',
    fit=dict(energy_bounds_meV=FIT_BOUNDS, energy_spacing_meV=float(GRID[1]-GRID[0]),
             model='single Voigt, free area/center/Gaussian sigma/Lorentzian gamma; zero background',
             loss='uniform least squares on peak-normalized noiseless model loss spectrum'),
    limitations=limitations,
    validation=dict(single_gaussian_center_meV=single_center, single_gaussian_fwhm_meV=single_width,
                    single_gaussian_voigt_fit=single_fit, multipeak_checks=multipeak_checks,
                    grid_refinement_check='Eligible single peaks: 0.01 vs 0.02 meV, tolerance 1e-5 meV'),
    runtime=dict(host=platform.node(), backend=jax.default_backend(), x64=jax.config.x64_enabled,
                 jax_version=jax.__version__, numpy_version=np.__version__,
                 omp_threads=os.environ.get('OMP_NUM_THREADS'),
                 openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS')),
    source_hashes={str(REFERENCE_PATH): digest(REFERENCE_PATH), str(Path(__file__)): digest(__file__),
                   str(vibrational.__file__): digest(vibrational.__file__)})
with (DESTINATION/'comparison.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)

for case, _, status in CASES:
    actual = sorted([r for r in rows if r['case'] == case and r['mechanism'] == 'electronic_plus_bose'],
                    key=lambda r: r['temperature_K'])
    bose = sorted([r for r in rows if r['case'] == case and r['mechanism'] == 'bose_only'],
                  key=lambda r: r['temperature_K'])
    temperatures = [r['temperature_K'] for r in actual]
    xc = next(item['settings']['xc'] for item in provenance if item['case'] == case)
    if not all(r['single_peak_comparison_eligible'] for r in actual+bose):
        fig, (shape_axis, weight_axis) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        columns = {'energy_meV': GRID}
        peak_rows = []
        for mechanism, series in (('electronic_plus_bose', actual), ('bose_only', bose)):
            for row in series:
                temperature = row['temperature_K']
                key = (case, mechanism, temperature)
                intensity = raw_spectra[key]
                label = f'{temperature:.0f} K; {mechanism.replace("_", " ")}'
                line, = shape_axis.plot(GRID, intensity, '-' if mechanism == 'electronic_plus_bose' else '--',
                                       label=label)
                columns[f'{mechanism}_{temperature:.0f}K_loss_per_meV'] = intensity
                columns[f'{mechanism}_{temperature:.0f}K_normalized'] = intensity/np.max(intensity)
                for number, peak in enumerate(row['peak_diagnostics']['local_peaks']):
                    peak_rows.append(dict(case=case, mechanism=mechanism, temperature_K=temperature,
                                          peak_index=number, **peak))
                    shape_axis.scatter(peak['energy_meV'], peak['relative_height']*np.max(intensity),
                                       marker='x', color=line.get_color())
                if mechanism == 'electronic_plus_bose':
                    # Degenerate individual weights depend on the eigenvector
                    # basis. JSON retains them; the plot does not assign modes.
                    weight_axis.scatter(row['eigenvalues_meV'], row['relative_loss_mode_areas'],
                                        label=f'{temperature:.0f} K', color=line.get_color())
        visible = [energy for row in actual+bose for energy in row['eigenvalues_meV']]
        shape_axis.set(xlim=(min(visible)-15., max(visible)+15.), xlabel='Energy loss (meV)',
                       ylabel='Relative model intensity / meV', title='Raw synthesized loss spectra; x = local maxima')
        weight_axis.set(xlabel='Mode energy (meV)', ylabel='Fraction of integrated loss',
                        title='Mode weights; degenerate-basis dependent')
        shape_axis.legend(fontsize=8)
        weight_axis.legend(fontsize=8)
        fig.suptitle(f'{xc}, {case}: resolved multiple model peaks\n'
                     f'One Q, unit form factors, Gaussian FWHM {FWHM:g} meV; no lifetime inference\n'
                     'Excluded from single-Voigt experimental comparison', fontsize=12)
        fig.savefig(DESTINATION/f'{case}_multipeak.png', dpi=180)
        fig.savefig(DESTINATION/f'{case}_multipeak.pdf')
        plt.close(fig)
        np.savetxt(DESTINATION/f'{case}_raw_spectra.csv', np.column_stack(list(columns.values())),
                   delimiter=',', header=','.join(columns), comments='')
        with (DESTINATION/f'{case}_local_peaks.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(peak_rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(peak_rows)
        continue
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    a, b, c, d, e, f = axes.ravel()
    for i, row in enumerate(actual):
        a.scatter([row['temperature_K']]*len(row['eigenvalues_meV']), row['eigenvalues_meV'],
                  c='0.65', s=18, marker='|', label='Raw eigenvalues' if i == 0 else None)
    a.plot(temperatures, [r['voigt_center_meV'] for r in actual], 'o-', label='Model Voigt center')
    a.plot(temperatures, [r['maximum_meV'] for r in actual], 'x--', label='Model envelope maximum')
    a.errorbar([300., 1377.], [EXPERIMENT[t]['energy_meV'] for t in (300., 1377.)],
               yerr=[EXPERIMENT[t]['energy_reported_uncertainty_meV'] for t in (300., 1377.)],
               xerr=[EXPERIMENT[t]['temperature_uncertainty_K'] or 0. for t in (300., 1377.)],
               fmt='ks', capsize=3, label='Reported unresolved peak')
    a.set(title='Eigenvalues and apparent peak', xlabel='Temperature (K)', ylabel='Energy (meV)')
    a.legend(fontsize=8)
    if 300. in temperatures and 1377. in temperatures:
        b.plot(temperatures, [r['voigt_shift_from_300K_meV'] for r in actual], 'o-', label='Electronic + Bose')
        b.plot(temperatures, [r['voigt_shift_from_300K_meV'] for r in bose], 's--', label='Bose only')
    else:
        b.text(.05, .8, 'Model endpoint series incomplete;\nno thermal-shift comparison.', transform=b.transAxes)
    b.errorbar([1377.], [EXPERIMENT_SHIFT], yerr=[EXPERIMENT_SHIFT_ERROR],
               xerr=[EXPERIMENT[1377.]['temperature_uncertainty_K']],
               fmt='ks', capsize=4, label='Experiment; independent errors')
    b.axhline(0., color='0.7', lw=.7)
    b.set(title='Shift relative to 300 K', xlabel='Temperature (K)', ylabel='Peak shift (meV)')
    b.legend(fontsize=8)
    for series, label, marker in ((actual, 'Electronic + Bose envelope', 'o-'),
                                  (bose, 'Bose-only envelope', 's--')):
        c.plot(temperatures, [r['envelope_fwhm_meV'] for r in series], marker, label=label)
    c.plot(temperatures, [r['voigt_fwhm_meV'] for r in actual], 'x:', label='Electronic + Bose Voigt')
    c.axhline(FWHM, color='0.6', label='Assumed instrument FWHM')
    c.set(title='Apparent width; no intrinsic lifetime', xlabel='Temperature (K)', ylabel='FWHM (meV)')
    c.legend(fontsize=8)
    for row in actual:
        temperature = row['temperature_K']
        shape, fit_shape = spectra[(case, 'electronic_plus_bose', temperature)]
        d.plot(GRID, shape, label=f'{temperature:.0f} K')
        e.plot(GRID-row['voigt_center_meV'], shape, label=f'{temperature:.0f} K model')
        e.plot(GRID-row['voigt_center_meV'], fit_shape, ':', color='0.6', alpha=.7)
    visible_energies = [energy for row in actual for energy in row['eigenvalues_meV']]
    visible_energies.extend(point['energy_meV'] for point in EXPERIMENT.values())
    d.set(xlim=(min(visible_energies)-15., max(visible_energies)+15.),
          title='Normalized model loss shapes', xlabel='Energy transfer (meV)', ylabel='Peak-normalized intensity')
    e.set(xlim=(-20., 20.), title='Centered shapes; dotted = Voigt fits', xlabel='Energy minus fitted center (meV)', ylabel='Peak-normalized intensity')
    d.legend(fontsize=8)
    e.legend(fontsize=8)
    f.axis('off')
    f.text(0., 1., 'Comparison boundaries\n\n'
        'Experimental static shift: -8.5 meV\n'
        'Formal error: 0.361 meV, independent endpoints.\n'
        'Room temperature mapped to 300 K.\n\n'
        'Reported ~7 meV broadening concerns Fig. 4f\n'
        'time-resolved data, not the static endpoints.\n'
        'No exact width endpoints or errors supplied.\n\n'
        'Model: one Q, unit form factors, Gaussian 11 meV.\n'
        'No momentum-window integration.\n'
        'No thermal expansion or bulk anharmonicity.\n'
        'No model uncertainty or lifetime prediction.\n\n'
        'Source: arXiv:2610.18561v1, II / III.1 / IV.', va='top', fontsize=10)
    fig.suptitle(f'AB graphite {xc}, {case}: {status}\nUnresolved-peak comparison; no empirical frequency correction', fontsize=13)
    fig.savefig(DESTINATION/f'{case}_comparison.png', dpi=180)
    fig.savefig(DESTINATION/f'{case}_comparison.pdf')
    plt.close(fig)
    columns = {'energy_meV': GRID}
    for mechanism in ('electronic_plus_bose', 'bose_only'):
        for temperature in temperatures:
            shape, profile = spectra[(case, mechanism, temperature)]
            columns[f'{mechanism}_{temperature:.0f}K_normalized'] = shape
            columns[f'{mechanism}_{temperature:.0f}K_voigt'] = profile
    np.savetxt(DESTINATION/f'{case}_normalized_shapes.csv', np.column_stack(list(columns.values())),
               delimiter=',', header=','.join(columns), comments='')

fig, (absolute, shifts) = plt.subplots(1, 2, figsize=(max(12, 2.5*len(CASES)), 5), constrained_layout=True)
labels = []
for index, (case, _, _) in enumerate(CASES):
    case_info = next(item for item in provenance if item['case'] == case)
    settings = case_info['settings']
    fft_label = 'x'.join(map(str, settings['fft_mesh']))
    k_label = 'x'.join(map(str, settings['k_mesh']))
    labels.append(f"{case}\n{settings['xc']}, FFT {fft_label}\nk {k_label}")
    actual = [r for r in rows if r['case'] == case and r['mechanism'] == 'electronic_plus_bose']
    cold_row = next(r for r in actual if r['temperature_K'] == 300.)
    absolute.scatter([index]*len(cold_row['eigenvalues_meV']), cold_row['eigenvalues_meV'],
                     marker='_', color='0.5', s=100, label='Raw modes' if index == 0 else None)
    if cold_row['single_peak_comparison_eligible']:
        absolute.scatter(index, cold_row['voigt_center_meV'], color='tab:blue',
                         label='Model Voigt center' if index == 0 else None)
    hot = next((r for r in actual if r['temperature_K'] == 1377.), None)
    if not case_info['single_peak_comparison_eligible']:
        shifts.text(index, 0., 'multiple peaks\nexcluded', ha='center', fontsize=8)
    elif hot is not None:
        shifts.scatter(index, hot['voigt_shift_from_300K_meV'], color='tab:blue')
    else:
        shifts.text(index, 0., 'missing\n1377 K', ha='center', fontsize=8)
cold_energy = EXPERIMENT[300.]['energy_meV']
cold_error = EXPERIMENT[300.]['energy_reported_uncertainty_meV']
absolute.axhspan(cold_energy-cold_error, cold_energy+cold_error, alpha=.25, color='black',
                 label=f'Reported room-T peak +/- {cold_error:g} meV')
absolute.axhline(cold_energy, color='black', lw=.8)
shifts.axhspan(EXPERIMENT_SHIFT-EXPERIMENT_SHIFT_ERROR, EXPERIMENT_SHIFT+EXPERIMENT_SHIFT_ERROR,
               alpha=.25, color='black', label='Experiment; independent endpoint errors')
shifts.axhline(EXPERIMENT_SHIFT, color='black', lw=.8)
for panel in (absolute, shifts):
    panel.set_xticks(range(len(labels)), labels, fontsize=8)
    panel.set_xlim(-.5, len(labels)-.5)
    panel.legend(fontsize=8)
absolute.set(title='300 K: separate numerical settings', ylabel='Energy (meV)')
shifts.set(title='300 to 1377 K: complete endpoints only', ylabel='Fitted peak shift (meV)')
fig.suptitle('XC and numerical settings comparison; no convergence or full-physics claim')
fig.savefig(DESTINATION/'settings_comparison.png', dpi=180)
fig.savefig(DESTINATION/'settings_comparison.pdf')
plt.close(fig)

fig, axes = plt.subplots(1, 2 if translation_checks else 1,
                         figsize=(15 if translation_checks else 8, 5), constrained_layout=True,
                         squeeze=False)
translation_axis = axes[0, 0]
for index, (case, _, _) in enumerate(CASES):
    for record in [r for r in translation_records if r['case'] == case]:
        temperature = record['temperature_K']
        for number, translation in enumerate(record['translations']):
            translation_axis.scatter(index+(number-.5)*.08, translation['signed_energy_meV'],
                marker={300.: 'o', 800.: 's', 1377.: '^'}.get(temperature, 'x'),
                color={300.: 'tab:blue', 800.: 'tab:green', 1377.: 'tab:orange'}.get(temperature, 'black'),
                label=f'{temperature:.0f} K, x/y translations' if index == 0 and number == 0 else None)
translation_axis.axhline(0., color='black', lw=.7)
translation_axis.set_xticks(range(len(CASES)), [case for case, _, _ in CASES], rotation=25, ha='right', fontsize=8)
translation_axis.set(ylabel='Signed Rayleigh energy (meV)',
                     title='Rigid translation: continuum target is zero; no ASR')
translation_axis.legend(fontsize=8)
if translation_checks:
    finite_difference_axis = axes[0, 1]
    for check in translation_checks:
        values = check['data']
        displacements = np.asarray(values['amplitudes_bohr'])
        energies = np.asarray(values['free_energies_Eh'])
        zero = np.flatnonzero(displacements == 0.)
        assert len(zero) == 1
        finite_difference_axis.scatter(displacements, (energies-energies[zero[0]])*1e9,
                                       label=f"{check['case']}: reconverged SCF")
        axis = np.linspace(displacements.min(), displacements.max(), 101)
        finite_difference_axis.plot(axis, .5*values['dfpt_curvature_Eh_Bohr2']*axis**2*1e9,
                                    label=f"{check['case']}: DFPT quadratic reference")
    finite_difference_axis.set(xlabel='Euclidean-unit rigid displacement amplitude (Bohr)',
                               ylabel='Free-energy change (1e-9 Hartree/cell)',
                               title='Independent displacement check; no fitted correction')
    finite_difference_axis.legend(fontsize=7)
fig.suptitle('Translation-invariance diagnostic from stored data; nonzero curvature is not a physical acoustic gap')
fig.savefig(DESTINATION/'rigid_translation_diagnostics.png', dpi=180)
fig.savefig(DESTINATION/'rigid_translation_diagnostics.pdf')
plt.close(fig)

summary['runtime']['elapsed_seconds'] = time.perf_counter()-START
(DESTINATION/'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
(DESTINATION/'rigid_translation_diagnostics.json').write_text(json.dumps(dict(
    scope='Read-only projected-Phi diagnostic; nonzero rigid-translation curvature violates translational invariance. Grid eggbox is a hypothesis requiring convergence evidence.',
    source_script_sha256=digest(__file__), records=translation_records,
    independent_translation_checks=translation_checks), indent=2, allow_nan=False)+'\n')
for row in rows:
    print(row['case'], row['mechanism'], row['temperature_K'],
          'center/width/shift/bias (meV):', row['voigt_center_meV'], row['envelope_fwhm_meV'],
          row['voigt_shift_from_300K_meV'], row['fitted_center_bias_meV'])
print('Saved', DESTINATION, 'elapsed (s):', summary['runtime']['elapsed_seconds'], flush=True)
