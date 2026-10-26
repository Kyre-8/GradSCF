"""Plot source-checked raw LiH Gamma matrices and independent gradient FD."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path('reproducibility/ad_showcase/P03_lih_gamma_phonons')
LABELS = ['Li x', 'Li y', 'Li z', 'H x', 'H y', 'H z']


def main(name='pbe_szv_k2_fft31', *, grid_report='grid_convergence.json'):
    native_path, reference_path = OUTPUT/(name+'.json'), OUTPUT/('pyscf_'+name+'.json')
    native, reference = json.loads(native_path.read_text()), json.loads(reference_path.read_text())
    rotation_path = OUTPUT/'rotation_covariance.json'
    rotation = json.loads(rotation_path.read_text())
    native_hash = hashlib.sha256(native_path.read_bytes()).hexdigest()
    for report in [native, reference, rotation]:
        assert report['complete'] and report['passed']
    assert reference['native_sha256'] == rotation['native_sha256'] == native_hash
    assert not native['asr_projected']
    phi = np.asarray(native['force_constants_real_hartree_per_bohr2'])
    matrices = np.asarray(reference['raw_hessians_hartree_per_bohr2'])
    mass = np.repeat(np.asarray(native['masses_amu'])*native['atomic_mass_to_electron_mass'], 3)
    matrix = .5*(matrices[-1]+matrices[-1].T)/np.sqrt(mass[:, None]*mass[None, :])
    eigenvalues = np.linalg.eigvalsh(matrix)
    finite_frequencies = np.sign(eigenvalues)*np.sqrt(abs(eigenvalues))*native['hartree_to_wavenumber']
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, values, title, unit in [
        (axes[0, 0], phi, 'Raw GradSCF Γ force constants', 'Eh/bohr²'),
        (axes[0, 1], 1e6*(matrices[-1]-phi), 'PySCF gradient FD − AD (h = 1e−4 bohr)', 'µEh/bohr²')]:
        extent = max(float(np.max(abs(values))), 1e-14)
        picture = ax.imshow(values, cmap='coolwarm', vmin=-extent, vmax=extent)
        ax.set_xticks(range(6), LABELS, rotation=45); ax.set_yticks(range(6), LABELS)
        ax.set_title(title); fig.colorbar(picture, ax=ax, label=unit, shrink=.8)
    values = 1e6*np.asarray(native['raw_translation_action_hartree_per_bohr2'])
    extent = max(float(np.max(abs(values))), 1e-14)
    picture = axes[1, 0].imshow(values, cmap='coolwarm', vmin=-extent, vmax=extent, aspect='auto')
    axes[1, 0].set_xticks(range(3), ['Translate x', 'Translate y', 'Translate z'])
    axes[1, 0].set_yticks(range(6), LABELS)
    axes[1, 0].set_title('Raw translation action ΦT; no ASR projection')
    fig.colorbar(picture, ax=axes[1, 0], label='µEh/bohr²', shrink=.8)
    modes = np.arange(1, 7)
    axes[1, 1].plot(modes, native['frequencies_signed_cm1'], 'x', markersize=8, label='GradSCF AD')
    axes[1, 1].plot(modes, finite_frequencies, 'o', markerfacecolor='none', label='PySCF gradient FD')
    axes[1, 1].axhline(0., color='.5', linewidth=.8)
    axes[1, 1].set_xticks(modes); axes[1, 1].set_xlabel('Sorted raw mode')
    axes[1, 1].set_ylabel('Signed frequency (cm⁻¹)'); axes[1, 1].grid(alpha=.2)
    axes[1, 1].set_title('Negative values remain visible'); axes[1, 1].legend(frameon=False)
    fig.suptitle('⁷Li¹H: PBE / GTH-Pade / SZV, k2³ / FFT31³\n'
                 'a = 7.7800322767 bohr; finite-model Γ curvatures, no LO–TO correction', fontsize=12)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUTPUT/f'lih_gamma_raw.{suffix}')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for column, label in enumerate(LABELS):
        rows = sorted((row for row in reference['columns'] if row['column'] == column), key=lambda row: row['step_bohr'])
        ax.loglog([row['step_bohr'] for row in rows],
            [max(row['max_native_error_hartree_per_bohr2'], 1e-15) for row in rows], 'o-', label=label)
    ax.axhline(reference['gates']['hessian_hartree_per_bohr2'], color='black', linestyle='--', label='Declared gate')
    ax.set_xlabel('Central-difference displacement (bohr)')
    ax.set_ylabel('Largest component error per column (Eh/bohr²)')
    ax.set_title('Independent, reconverged PySCF gradients; raw matrix comparison')
    ax.legend(ncol=3, frameon=False, fontsize=9); ax.grid(alpha=.2, which='both')
    for suffix in ['png', 'pdf']:
        fig.savefig(OUTPUT/f'lih_gamma_fd_steps.{suffix}')
    plt.close(fig)
    with (OUTPUT/'gamma_force_constants.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['row', 'column', 'native_Eh_bohr2', 'fd_h3e-4_Eh_bohr2', 'fd_h1e-4_Eh_bohr2'])
        for i, left in enumerate(LABELS):
            for j, right in enumerate(LABELS):
                writer.writerow([left, right, phi[i, j], matrices[0, i, j], matrices[1, i, j]])
    report = dict(passed=True, native_sha256=native_hash,
        independent_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        rotation_sha256=hashlib.sha256(rotation_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Signed raw Γ frequencies retain finite-grid acoustic errors. No ASR, full-dispersion, LO-TO or converged-material claim.')
    grid_path = OUTPUT/grid_report
    if grid_path.exists():
        grids = json.loads(grid_path.read_text())
        plot_grids(grids)
        report['grid_convergence_passed'] = grids['passed']
        report['grid_validation_file'] = grid_report
        report['passed'] &= grids['passed']
        report['grid_validation_sha256'] = hashlib.sha256(grid_path.read_bytes()).hexdigest()
    (OUTPUT/'plot_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Raw Gamma matrix, frequency and FD-step figures complete', flush=True)


def plot_grids(report):
    """Display failed grid gates as failures; never project the input modes."""
    assert report['complete']
    for case, fingerprints in report['input_sha256'].items():
        for extension, expected in fingerprints.items():
            assert hashlib.sha256((OUTPUT/(case+'.'+extension)).read_bytes()).hexdigest() == expected
    rows = report['rows']
    phi = []
    for row in rows:
        case = row.get('case', f'pbe_szv_k2_fft{row["mesh"]}')
        with np.load(OUTPUT/(case+'.npz')) as archive:
            phi.append(archive['force_constants'])
    force_changes = [float(np.max(abs(right-left))) for left, right in zip(phi[:-1], phi[1:])]
    optical = [np.asarray(row['optical_frequencies_signed_cm1']) for row in rows]
    optical_changes = [float(np.max(abs(right-left))) for left, right in zip(optical[:-1], optical[1:])]
    steps = [f'{left["mesh"]}³→{right["mesh"]}³' for left, right in zip(rows[:-1], rows[1:])]
    indices = list(range(len(steps)))
    grids = [row['mesh'] for row in rows]
    panels = [
        (force_changes, indices, steps, 'max |ΔΦ| (Eh/bohr²)', 'last_force_constant_increment_hartree_per_bohr2'),
        (optical_changes, indices, steps, 'max optical |Δν| (cm⁻¹)', 'last_optical_frequency_increment_cm1'),
        ([row['raw_asr_frobenius_hartree_per_bohr2'] for row in rows], grids, grids,
         'Raw ‖ΦT‖ (Eh/bohr²)', 'final_raw_asr_frobenius_hartree_per_bohr2'),
        ([float(np.max(abs(np.asarray(row['acoustic_frequencies_signed_cm1'])))) for row in rows], grids, grids,
         'max acoustic |ν| (cm⁻¹)', 'final_max_absolute_acoustic_frequency_cm1')]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for ax, (values, x, labels, title, key) in zip(axes.ravel(), panels):
        ax.semilogy(x, np.maximum(values, 1e-16), 'o-', color='#246b8e')
        ax.axhline(report['gates'][key], color='#c65b22', linestyle='--', label='Predeclared gate')
        ax.set_xticks(x, labels); ax.set_ylabel(title); ax.grid(alpha=.2)
        ax.set_title('PASS' if report['checks'][key] else 'FAIL',
                     color='#246b8e' if report['checks'][key] else '#a32929')
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_xlabel('FFT mesh per direction'); axes[1, 1].set_xlabel('FFT mesh per direction')
    signed = ', '.join(f'{value:+.4f}' for value in rows[-1]['acoustic_frequencies_signed_cm1'])
    fig.suptitle('Raw LiH Γ FFT checks; no acoustic projection\n'
                 'Fixed a = 7.7800322767 bohr, PBE / GTH-Pade / SZV, k2³\n'
                 f'FFT{rows[-1]["mesh"]} signed acoustic frequencies: {signed} cm⁻¹', fontsize=11)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUTPUT/(report.get('figure_stem', 'lih_gamma_grid_convergence')+'.'+suffix))
    plt.close(fig)


if __name__ == '__main__':
    main()

# Measured on jy_gpu_1, CPU float64, 2026-10-16.
# main(grid_report='grid_extension_to71.json')
# Raw Gamma matrix, frequency and FD-step figures complete
