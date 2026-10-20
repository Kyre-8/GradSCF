"""Prepare a same-orbital GaAs BerkeleyGW contour-deformation pilot.

Run ``prepare('artifacts/gaas_bgw_pilot')`` in Python, then run BGW epsilon
and sigma in that folder. This example prepares input; it does not launch
external processes. Full 14-band input requires the separately documented
complete-finite-space guard patch. The 1 Ry default is an interface pilot,
not a converged material gap. Increase cutoffs/frequency resolution only
after the pilot succeeds, and record each calculation in a fresh directory.
After actual BGW output exists, ``compare_gaas`` compares its components and
nonlinear table roots with the native frozen-source model. A separately
certified ``copy_eps0_without_head_feedback`` diagnostic can remove the
small-q dielectric head Schur feedback in a new file before rerunning Sigma;
that finite-body diagnostic is distinct from standard semiconductor output.
"""
from pathlib import Path
import hashlib
import json
import os
import platform
from time import perf_counter

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf.pbc import dft, gto
from gradscf_tools.berkeleygw import export_scf


def prepare(directory, *, source_kmesh=(2, 1, 1), cutoff_rydberg=1.,
            q0_fractional=(0., 0., .001), screening_bands=14, head='body'):
    """Export actual native orbitals, a shifted WFNq, and epsilon/sigma input.

    Set screening_bands=13 only for an unpatched-BGW format pilot: it omits
    the final empty state and is explicitly not the full finite model.
    The same 14-orbital SCF reference is exported in either case.
    ``head='body'`` disables the BGW Gamma Coulomb-cell average using its debug
    keywords. Native Ewald exchange has an additional known occupied-state
    constant, recorded separately; it is not silently added to BGW results.
    The small-q dielectric matrix still includes head feedback before Sigma
    removes its head. This differs from inverting a head-free native response.
    """
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError('Use a fresh BGW directory to preserve prior outputs')
    if screening_bands not in (13, 14):
        raise ValueError('Choose the explicit 13-band format pilot or full 14-band finite model')
    if head not in ('body', 'semiconductor'):
        raise ValueError("head must be 'body' or 'semiconductor'")
    a = 5.6533
    lattice = np.array([[0., .5, .5], [.5, 0., .5], [.5, .5, 0.]])*a
    cell = gto.M(atom=[('Ga', (0., 0., 0.)), ('As', (a/4, a/4, a/4))],
        a=lattice, unit='Angstrom', basis='gth-szv', pseudo='gth-pbe',
        mesh=(21, 21, 21), cart=True, precision=1e-10, dimension=3)
    mf = dft.KRKS(cell, kpts=cell.make_kpts(source_kmesh), xc='pbe',
        exxdiv='ewald', temperature=None, conv_tol=1e-11,
        conv_tol_density=1e-10, conv_tol_grad=1e-9, max_cycle=150, damping=.15)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('The native reference did not converge')
    q0 = np.asarray(q0_fractional, dtype=float)
    reciprocal = np.asarray(cell.reciprocal_vectors())
    shifted = np.asarray(mf.kpts)+q0@reciprocal
    # BGW checks the WFN and WFNq header cutoffs for exact consistency.
    # Declare one bound covering both sets, without changing coefficients.
    all_points = np.concatenate([np.asarray(mf.kpts), shifted])
    gk = np.asarray(mf.inputs.gvectors)[None]+all_points[:, None]
    common_cutoff = float(np.ceil(np.max(np.sum(gk*gk, axis=-1))))
    report = export_scf(mf, directory, wavefunction_cutoff_rydberg=common_cutoff)
    (directory/'WFN_inner').symlink_to('WFN')
    shifted_report = export_scf(mf, directory, kpoints=shifted,
        kgrid=source_kmesh, shift=q0*np.asarray(source_kmesh),
        wfn_name='WFNq', vxc_name=None, wavefunction_cutoff_rydberg=common_cutoff)
    # The valence WFNq is shifted by +q0, following BGW's EPM example.
    qpoints = np.mod(np.asarray(mf.kpts)@np.asarray(cell.lattice).T/(2*np.pi), 1.)
    qlines = [' '.join(f'{x:.12f}' for x in q0)+' 1.0 1']
    qlines.extend(' '.join(f'{x:.12f}' for x in point)+' 1.0 0'
        for point in qpoints if np.max(abs(point)) > 1e-12)
    epsilon = f'''# Same Gaussian Bloch orbitals; all physical bands retained in WFN.
epsilon_cutoff {cutoff_rydberg:.10f}
number_bands {screening_bands}
degeneracy_check_override
frequency_dependence 2
frequency_dependence_method 2
number_imaginary_freqs 16
plasma_freq 27.211386245988
broadening 0.027211386245988
delta_frequency 0.25
frequency_low_cutoff 60.0
frequency_high_cutoff 60.0
no_min_fftgrid
verbosity 2
write_vcoul
begin qpoints
{chr(10).join(qlines)}
end
'''
    sigma = f'''# Diagonal CD comparison; not nonlinear matrix Dyson or full scGW.
number_bands {screening_bands}
band_index_min 7
band_index_max 12
screened_coulomb_cutoff {cutoff_rydberg:.10f}
bare_coulomb_cutoff {cutoff_rydberg:.10f}
degeneracy_check_override
no_symmetries_q_grid
frequency_dependence 2
frequency_dependence_method 2
frequency_grid_shift 2
delta_frequency_eval 0.1
max_frequency_eval 3.0
cd_integration_method 2
screening_semiconductor
verbosity 2
write_vcoul
begin kpoints
0.0 0.0 0.0 1.0
end
'''
    if head == 'body':
        sigma += '\n# Debug finite body protocol: exact Gamma Coulomb head is zero.\ncell_average_cutoff 0.0\nskip_averagew\n'
    (directory/'epsilon.inp').write_text(epsilon)
    (directory/'sigma.inp').write_text(sigma)
    report.update(shifted_export=shifted_report,
        screening_bands=screening_bands, full_finite_model=screening_bands == 14,
        requires_bgw_complete_finite_space_patch=screening_bands == 14,
        q0_fractional=q0.tolist(), cutoff_rydberg=cutoff_rydberg,
        head=head,
        bare_exchange_boundary='BGW zero Gamma head; native occupied Ewald constant reported separately' if head == 'body' else 'BGW default Gamma-cell average',
        screening_boundary='BGW small-q epsilon inverted with head/wings; Sigma-only head removal does not undo body Schur feedback',
        source_kmesh=list(source_kmesh), scf_energy_hartree=float(mf.e_tot),
        scope='Same-orbital diagonal BGW CD interface pilot; cutoffs, q0 and frequency grids not converged')
    (directory/'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(directory/'native_reference.npz',
        energies_hartree=np.asarray(mf.mo_energy),
        density_spin=np.asarray(mf.result.density_spin),
        source_kpoints=np.asarray(mf.kpts))
    return mf, report


def compare_gaas(bgw_directory, directory, *, screening_cutoff_rydberg=8.,
                 nw_values=(16, 64), bands=tuple(range(6, 12)), solve_roots=True,
                 bgw_hartree_to_ev=27.21138506, eta_g=.001, screening_eta=None):
    """Compare actual BGW and native diagonal G0W0 components at Gamma.

    The PBE source and all 14 internal bands stay fixed. Native
    exxdiv=None adds no Madelung term. Only W vertices get the stated sphere
    cutoff; native bare exchange retains its full FFT model. BGW's remaining
    spatial and quadrature differences are retained, not fitted away.
    Tabulated BGW roots use the shared positive-slope nearest-on-shell
    selector on piecewise-linear Re Sigma. Values outside the actual table
    are NaN; the residual certifies only that interpolated table equation.
    The explicit BGW conversion defaults to twice RYD=13.60569253 eV in
    the inspected 4.0 build, rather than silently treating legacy code units
    as modern CODATA units. This parameter never shifts or fits a spectrum.
    ``screening_eta=None`` preserves the legacy time-ordered G-pole response.
    Explicit ``screening_eta`` uses independent signed retarded/advanced W
    broadening; ``eta_g=0`` gives the real-G imaginary integral used by BGW.
    This aligns frequency definitions, not the distinct product/q0 closures.
    """
    import jax.numpy as jnp
    from gradscf.gw.pbc import build_periodic_model
    from gradscf.gw.pbc.source import build_correlation, correlation_diagonal, hartree_exchange
    from gradscf.gw.qp import solve_quasiparticle_roots
    from gradscf.solvers.nonlinear import ScalarRootConfig
    from gradscf_tools.berkeleygw import read_sigma, read_eqp, HARTREE_TO_EV
    from examples.pbc.band_structures.gaas_self_consistent_gw import reference

    bgw_directory, directory = Path(bgw_directory), Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    config = ScalarRootConfig(method='hybrid', ftol=1e-9, xtol=1e-9, maxiter=120)
    spectra = {row['band']:row for row in read_sigma(bgw_directory, hartree_to_ev=bgw_hartree_to_ev)
        if row['spin'] == 1 and np.max(abs(row['kpoint_fractional'])) < 1e-10}
    references = {row['band']:row['reference_hartree'] for row in read_eqp(bgw_directory/'eqp0.dat', hartree_to_ev=bgw_hartree_to_ev)
        if row['spin'] == 1 and np.max(abs(row['kpoint_fractional'])) < 1e-10}
    if not all(band in spectra and band in references for band in bands):
        raise ValueError('The BGW case must contain every requested Gamma state')
    sources = ['src/gradscf/gw/pbc/source.py', 'src/gradscf/gw/qp.py',
        'src/gradscf/gw/freq.py', 'src/gradscf/gw/pbc/query.py',
        'src/gradscf_tools/berkeleygw.py',
        'examples/pbc/band_structures/gaas_self_consistent_gw.py',
        'examples/pbc/band_structures/berkeleygw_gaas.py']
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = 'threadpoolctl unavailable; environment only'
    schur_path = bgw_directory/'schur_copy.json'
    schur_receipt = json.loads(schur_path.read_text()) if schur_path.exists() else None
    report = dict(complete=False, stage='reference', rows=[],
        bgw_directory=str(bgw_directory), screening_cutoff_rydberg=screening_cutoff_rydberg,
        bgw_hartree_to_ev=bgw_hartree_to_ev, reporting_hartree_to_ev=HARTREE_TO_EV,
        native_nw_values=list(nw_values), eta_hartree=eta_g,
        screening_eta_hartree=screening_eta, exxdiv=None,
        response_broadening_protocol='legacy time-ordered G pole' if screening_eta is None else 'independent signed retarded/advanced response shift',
        source_kmesh=[2, 1, 1], original_fft_mesh=[21, 21, 21],
        dtype='float64/complex128', backend=jax.default_backend(), hostname=platform.node(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        thread_environment={name:os.environ.get(name) for name in ('MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
        threadpools=pools, source_sha256={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sources},
        bgw_sha256={name:hashlib.sha256((bgw_directory/name).read_bytes()).hexdigest()
            for name in ('spectrum.dat', 'eqp0.dat', 'eqp1.dat', 'x.dat', 'vxc.dat')},
        branch_rule='SRC shared positive-slope nearest on-shell among tested seeds; counts are not global enumeration',
        bgw_root_scope='Residual of piecewise-linear evaluated table only; no extrapolation',
        bgw_inverse_dielectric_transform=schur_receipt,
        protocol_differences=['native full-FFT bare exchange versus BGW bare sphere',
            ('native Gamma body response versus BGW finite-small-q body response; head Schur feedback removed'
             if schur_receipt else 'native head-free response versus BGW small-q inverse body Schur feedback'),
            'native scaled Legendre x0=0.5 Ha versus BGW piecewise quadratic CD grid',
            ('native time-ordered response at complex G poles versus BGW retarded-chi table without an additional G-pole shift'
             if screening_eta is None else 'native full OV/VO signed R/A response versus BGW folded CV response; spatial frame/closure differences remain')])
    def save():
        (directory/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    def real(value):
        value = float(np.asarray(value))
        return value if np.isfinite(value) else None
    def root_record(value):
        return dict(energy_hartree=real(value.roots[0]), residual_hartree=real(value.residual[0]),
            weight=real(value.weight[0]), candidate_count=int(value.candidate_count[0]),
            branch_valid=bool(value.branch_valid[0]), converged=bool(value.converged[0]),
            on_shell_hartree=real(value.on_shell[0]))
    save()
    mf = reference(source_kmesh=(2, 1, 1), mesh=(21, 21, 21), exxdiv=None)
    report['scf_seconds'] = perf_counter()-started
    model = build_periodic_model(mf, screening_cutoff_rydberg=screening_cutoff_rydberg)
    report['model_seconds'] = perf_counter()-started-report['scf_seconds']
    report['model_metadata'] = model.metadata
    k = int(np.argmin(np.linalg.norm(np.asarray(model.kpoints), axis=1)))
    _, exchange = hartree_exchange(model, model.reference_density)
    xdiag = np.diag(np.asarray(exchange[k])).real
    vdiag = np.diag(np.asarray(model.reference_potential[k])).real
    energy = np.asarray(model.energy[k])
    error = max(abs(energy[band]-references[band]) for band in bands)
    report.update(reference_energy_max_error_hartree=float(error),
        native_reference_energy_hartree=energy.tolist(),
        bgw_reference_energy_hartree={str(band):float(value) for band, value in references.items()},
        scf_energy_hartree=float(mf.e_tot), model_madelung_hartree=model.madelung,
        actual_exxdiv=mf.exxdiv)
    save()
    if error > 1e-8 or model.madelung != 0.:
        raise ValueError('BGW/native references differ or native Madelung is nonzero')
    report['reference_energy_max_error_hartree'] = float(error)
    report['scf_energy_hartree'] = float(mf.e_tot)
    report['model_madelung_hartree'] = model.madelung
    bgw_equations, bgw_roots, bgw_physical_roots = {}, {}, {}
    for band in bands:
        row = spectra[band]
        frequencies = jnp.asarray(row['frequency_hartree'])
        total = jnp.asarray(row['sigma_total_hartree'].real)
        def equation(omega, frequency=frequencies, sigma=total,
                     eref=references[band], vxc=float(row['vxc_hartree'].real)):
            return omega-eref-jnp.interp(omega, frequency, sigma, left=jnp.nan, right=jnp.nan)+vxc
        bgw_equations[band] = equation
        if solve_roots:
            bgw_roots[band] = solve_quasiparticle_roots(equation,
                jnp.asarray([references[band]]), config=config)
            # vxc.dat was written in modern physical eV, but this BGW build
            # divided it by its older internal RYD. Show the convention
            # conversion separately without changing the measured Sigma.
            actual_vxc = float(row['vxc_hartree'].real)
            physical_vxc = actual_vxc*bgw_hartree_to_ev/HARTREE_TO_EV
            physical_equation = lambda omega, equation=equation, delta=physical_vxc-actual_vxc: equation(omega)+delta
            bgw_physical_roots[band] = solve_quasiparticle_roots(physical_equation,
                jnp.asarray([references[band]]), config=config)
    save()
    for nw in nw_values:
        begin = perf_counter()
        report.update(stage='screening', current_nw=int(nw))
        save()
        print(f'Native W cutoff={screening_cutoff_rydberg:g} Ry nw={nw}', flush=True)
        broadening = {} if screening_eta is None else dict(screening_eta=screening_eta)
        cache = build_correlation(model, model.energy, nw=nw, eta=eta_g, **broadening)
        for band in bands:
            report.update(stage='sigma', current_band=int(band))
            bgw = spectra[band]
            sigma = correlation_diagonal(cache, k, jnp.asarray(energy[band]), (band,))
            bgw_c = np.interp(references[band], bgw['frequency_hartree'], bgw['sigma_correlation_hartree'].real)
            item = dict(nw=int(nw), band=int(band), energy_reference_hartree=float(energy[band]),
                native_exchange_hartree=float(xdiag[band]), bgw_exchange_hartree=float(bgw['exchange_hartree'].real),
                native_vref_hartree=float(vdiag[band]), bgw_vxc_hartree=float(bgw['vxc_hartree'].real),
                bgw_vxc_physical_hartree=float(bgw['vxc_hartree'].real*bgw_hartree_to_ev/HARTREE_TO_EV),
                bgw_vxc_input_unit_delta_ev=float(bgw['vxc_hartree'].real*(1-bgw_hartree_to_ev/HARTREE_TO_EV)*HARTREE_TO_EV),
                native_correlation_on_shell_hartree=real(sigma.real), bgw_correlation_on_shell_hartree=float(bgw_c),
                delta_exchange_ev=float((xdiag[band]-bgw['exchange_hartree'].real)*HARTREE_TO_EV),
                delta_vref_ev=float((vdiag[band]-bgw['vxc_hartree'].real)*HARTREE_TO_EV),
                delta_correlation_on_shell_ev=float((sigma.real-bgw_c)*HARTREE_TO_EV),
                bgw_reported_qp_flag=bgw['reported_qp_flag'])
            if solve_roots:
                def native_equation(omega, band=band):
                    return omega-energy[band]-xdiag[band]+vdiag[band]-correlation_diagonal(cache, k, omega[0], (band,)).real
                solved = solve_quasiparticle_roots(native_equation,
                    jnp.asarray([energy[band]]), config=config)
                other = bgw_roots[band]
                item.update(native_root=root_record(solved), bgw_table_root=root_record(other),
                    bgw_table_root_physical_vxc=root_record(bgw_physical_roots[band]),
                    bgw_table_residual_at_native_root_hartree=real(bgw_equations[band](solved.roots)[0]),
                    native_residual_at_bgw_root_hartree=real(native_equation(other.roots)[0])
                        if bool(other.branch_valid[0]) else None)
            report['rows'].append(item)
            save()
            print(f'band={band} delta X={item["delta_exchange_ev"]:.6g} eV delta Cor={item["delta_correlation_on_shell_ev"]:.6g} eV', flush=True)
            # Bound compiled closure retention across independent scalar states.
            jax.clear_caches()
        report.setdefault('nw_seconds', {})[str(nw)] = perf_counter()-begin
        del cache
        jax.clear_caches()
    report.update(complete=True, stage='complete', elapsed_seconds=perf_counter()-started)
    save()
    return report


def sample_comparison_curves(comparison_directory, directory, *, step_ev=.1,
                             root_step_ev=.01):
    """Save actual six-state SRC correlation curves for an executed comparison.

    This glue evaluates the existing correlation kernel, without changing
    its Hamiltonian, broadening, quadrature or QP branch selection. The
    local root interval is sampled more finely than the overview interval.
    All input comparison rows must use the same native frequency count.
    """
    import jax.numpy as jnp
    from gradscf.gw.pbc import build_periodic_model
    from gradscf.gw.pbc.source import build_correlation, correlation_diagonal
    from gradscf_tools.berkeleygw import HARTREE_TO_EV, read_sigma
    from examples.pbc.band_structures.gaas_self_consistent_gw import reference

    comparison_directory, directory = Path(comparison_directory), Path(directory)
    report = json.loads((comparison_directory/'comparison.json').read_text())
    if not report['complete'] or len({row['nw'] for row in report['rows']}) != 1:
        raise ValueError('Use a completed comparison with one native frequency count')
    if min(step_ev, root_step_ev) <= 0:
        raise ValueError('Curve spacings must be positive')
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    mf = reference(source_kmesh=(2, 1, 1), mesh=(21, 21, 21), exxdiv=None)
    model = build_periodic_model(mf, screening_cutoff_rydberg=report['screening_cutoff_rydberg'])
    k = int(np.argmin(np.linalg.norm(np.asarray(model.kpoints), axis=1)))
    broadening = {} if report.get('screening_eta_hartree') is None else dict(screening_eta=report['screening_eta_hartree'])
    cache = build_correlation(model, model.energy, nw=report['native_nw_values'][0],
        eta=report['eta_hartree'], **broadening)
    spectra = {row['band']:row for row in read_sigma(report['bgw_directory'],
        hartree_to_ev=report['bgw_hartree_to_ev']) if np.max(abs(row['kpoint_fractional'])) < 1e-10}
    arrays = {}
    sampled_root_residuals = []
    for row in report['rows']:
        band, reference_energy = row['band'], row['energy_reference_hartree']
        if abs(float(model.energy[k, band])-reference_energy) > 1e-10:
            raise ValueError('The curve reference differs from the executed comparison')
        native_root = row['native_root']['energy_hartree']
        bgw_root = row['bgw_table_root']['energy_hartree']
        roots_ev = (np.array([native_root, bgw_root])-reference_energy)*HARTREE_TO_EV
        lower, upper = min(-.3, roots_ev.min()-.4), max(.3, roots_ev.max()+.4)
        overview = np.arange(lower, upper+step_ev/2, step_ev)
        near_root = np.arange(roots_ev.min()-.25, roots_ev.max()+.25, root_step_ev)
        relative = np.unique(np.r_[overview, near_root, roots_ev, 0.])
        frequency = reference_energy+relative/HARTREE_TO_EV
        values = []
        for omega in frequency:
            value = correlation_diagonal(cache, k, jnp.asarray(omega), (band,))
            values.append(complex(np.asarray(value)))
        arrays[f'band{band}_frequency_hartree'] = frequency
        arrays[f'band{band}_native_correlation_hartree'] = np.asarray(values)
        arrays[f'band{band}_bgw_frequency_hartree'] = spectra[band]['frequency_hartree']
        arrays[f'band{band}_bgw_correlation_hartree'] = spectra[band]['sigma_correlation_hartree']
        root_index = int(np.argmin(abs(frequency-native_root)))
        residual = (native_root-reference_energy-row['native_exchange_hartree']
            +row['native_vref_hartree']-values[root_index].real)
        if abs(frequency[root_index]-native_root) > 1e-13 or abs(residual) > 1e-7:
            raise ValueError('The sampled native curve no longer matches the executed root')
        sampled_root_residuals.append(float(abs(residual)))
        np.savez_compressed(directory/'correlation_curves.npz', **arrays)
        print(f'Actual native Sigma_c curve: band {band}, {len(frequency)} frequencies', flush=True)
        jax.clear_caches()
    report.update(bgw_preparation=json.loads((Path(report['bgw_directory'])/'preparation.json').read_text()),
        curve_sampling=dict(step_ev=step_ev, root_step_ev=root_step_ev,
        elapsed_seconds=perf_counter()-started, native_curve_source='SRC correlation_diagonal',
        max_sampled_root_residual_hartree=max(sampled_root_residuals),
        bgw_curve_source='Actual spectrum.dat; linear connection is a display/table interpolation',
        band_labels='zero based, with one-based labels shown explicitly in the figure',
        source_sha256={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in
            ['src/gradscf/gw/pbc/source.py', 'src/gradscf_tools/berkeleygw.py',
             'examples/pbc/band_structures/berkeleygw_gaas.py']}))
    (directory/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def plot_comparison_curves(directory):
    """Render this six-state certified Schur-body comparison, with no GW work."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from gradscf_tools.berkeleygw import HARTREE_TO_EV

    directory = Path(directory)
    report = json.loads((directory/'comparison.json').read_text())
    controls = report['bgw_preparation']
    if ([row['band'] for row in report['rows']] != list(range(6, 12))
            or report['bgw_inverse_dielectric_transform'] is None):
        raise ValueError('This figure requires Gamma bands 6--11 and the certified head-Schur body adapter')
    with np.load(directory/'correlation_curves.npz', allow_pickle=False) as stored:
        arrays = {key:stored[key].copy() for key in stored.files}
    style = {'font.family':'DejaVu Sans', 'font.size':9.5, 'axes.titlesize':11,
             'pdf.fonttype':42, 'savefig.dpi':210}
    compact = []
    if report.get('screening_eta_hartree') is None:
        broadening_note = f"Finite broadening differs: native G-pole η={report['eta_hartree']:g} Ha (positive OV/VO shifts 2η/0); BGW retarded χ δ≈0.001 Ha (both +δ), no extra G shift."
    else:
        broadening_note = f"Explicit frequency protocol: native ηG={report['eta_hartree']:g} Ha, δW={report['screening_eta_hartree']:.10g} Ha (signed R/A); BGW retarded χ δ≈0.001 Ha, no extra G shift."
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.5))
        for axis, row in zip(axes.flat, report['rows']):
            band, eref = row['band'], row['energy_reference_hartree']
            frequency = arrays[f'band{band}_frequency_hartree']
            x = (frequency-eref)*HARTREE_TO_EV
            native = arrays[f'band{band}_native_correlation_hartree'].real*HARTREE_TO_EV
            bgw_frequency = arrays[f'band{band}_bgw_frequency_hartree']
            bgw_x = (bgw_frequency-eref)*HARTREE_TO_EV
            bgw = arrays[f'band{band}_bgw_correlation_hartree'].real*HARTREE_TO_EV
            selected = (bgw_x >= x.min()) & (bgw_x <= x.max())
            axis.plot(x, native, color='#2166AC', lw=1.45)
            axis.plot(bgw_x[selected], bgw[selected], '--', color='#B35806', lw=1.3,
                      marker='.', ms=2.7)
            for key, color, marker in [('native_root', '#2166AC', 'o'), ('bgw_table_root', '#B35806', 'D')]:
                root = row[key]['energy_hartree']
                root_x = (root-eref)*HARTREE_TO_EV
                root_y = (root-eref-row['native_exchange_hartree']+row['native_vref_hartree'])*HARTREE_TO_EV if key == 'native_root' else (root-eref-row['bgw_exchange_hartree']+row['bgw_vxc_hartree'])*HARTREE_TO_EV
                axis.axvline(root_x, color=color, lw=.8, alpha=.55)
                axis.scatter(root_x, root_y, s=34, color=color, marker=marker, zorder=5)
            native_root, bgw_root = row['native_root'], row['bgw_table_root']
            difference = (bgw_root['energy_hartree']-native_root['energy_hartree'])*HARTREE_TO_EV*1000
            axis.axvline(0., color='#888888', lw=.7, ls=':')
            axis.set(title=f'Γ band {band} (1-based {band+1}) · {row["bgw_reported_qp_flag"]}',
                xlabel='E − εPBE for this state (eV)', ylabel='Re Σc(E) (eV)', xlim=(x.min(), x.max()))
            axis.text(.03, .97, f'QP BGW−native: {difference:+.2f} meV\nShared candidates: {native_root["candidate_count"]} / {bgw_root["candidate_count"]}',
                transform=axis.transAxes, va='top', fontsize=8.2,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
            axis.spines[['top', 'right']].set_visible(False)
            axis.grid(alpha=.18)
            compact.append(dict(band_zero_based=band, band_one_based=band+1,
                native_sigma_c_on_shell_ev=row['native_correlation_on_shell_hartree']*HARTREE_TO_EV,
                bgw_sigma_c_on_shell_ev=row['bgw_correlation_on_shell_hartree']*HARTREE_TO_EV,
                native_minus_bgw_sigma_c_on_shell_mev=row['delta_correlation_on_shell_ev']*1000,
                native_minus_bgw_exchange_mev=row['delta_exchange_ev']*1000,
                native_qp_ev=native_root['energy_hartree']*HARTREE_TO_EV,
                bgw_qp_ev=bgw_root['energy_hartree']*HARTREE_TO_EV, bgw_minus_native_qp_mev=difference,
                native_F_at_bgw_qp_mev=row['native_residual_at_bgw_root_hartree']*HARTREE_TO_EV*1000,
                bgw_F_at_native_qp_mev=row['bgw_table_residual_at_native_root_hartree']*HARTREE_TO_EV*1000,
                native_candidates=native_root['candidate_count'], bgw_candidates=bgw_root['candidate_count'],
                native_own_residual_ev=native_root['residual_hartree']*HARTREE_TO_EV,
                bgw_table_own_residual_ev=bgw_root['residual_hartree']*HARTREE_TO_EV,
                bgw_original_flag=row['bgw_reported_qp_flag']))
        fig.suptitle('GaAs · actual same-orbital G₀W₀ self-energies and selected QP roots', fontsize=15, y=.98)
        fig.text(.5, .936, f"14 physical WFN / GW states · source 2×1×1 · W {report['screening_cutoff_rydberg']:g} Ry · BGW bare {controls['bare_coulomb_cutoff_rydberg']:g} Ry; native bare full FFT",
                 ha='center', fontsize=9.5)
        fig.legend(handles=[Line2D([], [], color='#2166AC', lw=1.5, marker='o', label=f"Native ν{report['native_nw_values'][0]} / selected root"),
            Line2D([], [], color='#B35806', ls='--', lw=1.5, marker='D', label=f"BGW ν{controls['nw']} / selected table root")],
            ncol=2, loc='upper center', bbox_to_anchor=(.5, .921), frameon=False, fontsize=9)
        note = ('Finite-body Schur adapter: head=1, wings=0, body=E−CA⁻¹B; finite q₀=(0,0,0.001). Not standard bulk semiconductor output.\n'
                f"BGW: Ep={controls['bgw_plasma_frequency_hartree']:g} Ha, real screening Δ={controls['screening_real_frequency_delta_ev']:g} eV to {controls['screening_real_frequency_cutoff_ev']:g} eV, Σ table Δ={controls['delta_frequency_eval_ev']:g} eV. Native: scaled Legendre ν{report['native_nw_values'][0]}.\n"
                +broadening_note+'\n'
                'BGW legacy conversion: 1 Ha=27.21138506 eV; displayed physical energies: 27.211386245988 eV/Ha. Known Vxc input-unit effect retained.\n'
                'MULT remains: root markers follow the same positive-slope nearest-on-shell selector; counts are tested seeds, not exhaustive roots. No full-scGW equivalence or bulk-gap claim.')
        fig.text(.065, .025, note, fontsize=8.1, linespacing=1.5, color='#444444', va='bottom')
        fig.subplots_adjust(left=.065, right=.985, bottom=.22, top=.85, hspace=.42, wspace=.3)
        outputs = {suffix:directory/f'gaas_native_bgw_sigma.{suffix}' for suffix in ['pdf', 'png']}
        for path in outputs.values():
            fig.savefig(path, facecolor='white', bbox_inches='tight')
        plt.close(fig)
    with (directory/'selected_gamma_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(compact[0]))
        writer.writeheader()
        writer.writerows(compact)
    return outputs


__all__ = ['prepare', 'compare_gaas', 'sample_comparison_curves', 'plot_comparison_curves']
