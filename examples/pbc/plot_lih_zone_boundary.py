"""Plot saved raw q-half force constants and the independent H x column.

The full matrix has native response checks; only the H x column is checked by
independent gradient differences. Signed numerical modes keep negative roots.
No ASR, interpolation, LO-TO correction or material stability claim is made.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants, h, c


OUTPUT = Path('reproducibility/ad_showcase/P03_lih_zone_boundary')
LABELS = ['Li x', 'Li y', 'Li z', 'H x', 'H y', 'H z']


def main():
    paths = {key:OUTPUT/name for key,name in dict(
        full='pbe_szv_k2_fft31_qhalf_full.json',
        probe='pbe_szv_k2_fft31_qhalf_probe.json',
        reference='pyscf_pbe_szv_k2_fft31_qhalf_probe_fd.json',
        folding='pyscf_pbe_szv_k2_fft31_qhalf_probe.json').items()}
    full, probe, reference, folding = (json.loads(paths[key].read_text())
                                      for key in ['full', 'probe', 'reference', 'folding'])
    for record in [full, probe, reference, folding]:
        assert record['complete'] and record['passed']
    assert not full['probe'] and not reference['probe']
    fingerprint = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert reference['native_sha256'] == folding['native_sha256'] == fingerprint(paths['probe'])
    assert reference['parent_probe_sha256'] == fingerprint(paths['folding'])
    assert not full['asr_projected'] and not probe['asr_projected']
    for key in ['settings', 'source_python_sha256', 'masses_amu', 'q_evaluated',
                'lattice_bohr', 'coordinates_bohr', 'fractional_kpoints', 'linear_solver']:
        assert full[key] == probe[key], key
    phi = np.asarray(full['force_constant_actions_real_hartree_per_bohr2'])
    phi = phi+1j*np.asarray(full['force_constant_actions_imag_hartree_per_bohr2'])
    np.testing.assert_array_equal(full['directions'], np.eye(6).reshape(6, 2, 3))
    np.testing.assert_array_equal(probe['directions'], np.eye(6)[3:4].reshape(1, 2, 3))
    column = np.asarray(probe['force_constant_actions_real_hartree_per_bohr2'])[:, 0]
    np.testing.assert_allclose(phi[:, 3].real, column, atol=1e-10, rtol=0)
    masses = np.repeat(np.asarray(full['masses_amu']), 3)
    masses *= physical_constants['atomic mass constant'][0]/physical_constants['electron mass'][0]
    # Eager numerical postprocessing after the raw Hermiticity gate. The
    # archived Phi is never symmetrized, projected or replaced by this matrix.
    dynamical = .5*(phi+phi.conj().T)/np.sqrt(masses[:, None]*masses[None, :])
    eigenvalues = np.linalg.eigvalsh(dynamical)
    conversion = physical_constants['Hartree energy'][0]/(h*c*100)
    frequencies = np.sign(eigenvalues)*np.sqrt(abs(eigenvalues))*conversion
    rows = sorted(reference['columns'], key=lambda row:row['step_bohr'])
    assert len(rows) == 2 and {row['column'] for row in rows} == {0}
    finite = np.asarray(rows[0]['finite_difference_hartree_per_bohr2'])
    plt.rcParams.update({'font.size':10, 'pdf.fonttype':42, 'savefig.dpi':220})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    limit = float(np.max(abs(phi.real)))
    picture = axes[0, 0].imshow(phi.real, cmap='coolwarm', vmin=-limit, vmax=limit)
    axes[0, 0].set_xticks(range(6), LABELS, rotation=45)
    axes[0, 0].set_yticks(range(6), LABELS)
    axes[0, 0].set_title('Raw native force constants; no ASR')
    fig.colorbar(picture, ax=axes[0, 0], label='Eh/bohr²', shrink=.8)
    axes[0, 1].bar(np.arange(1, 7), frequencies,
                    color=np.where(frequencies < 0, '#b44738', '#246b8e'))
    axes[0, 1].axhline(0, color='black', linewidth=.8)
    axes[0, 1].set_xlabel('Sorted native mode')
    axes[0, 1].set_ylabel('Signed frequency (cm⁻¹)')
    axes[0, 1].set_title('Negative curvatures are retained')
    axes[1, 0].plot(range(6), column, 'x', markersize=8, label='Native response')
    axes[1, 0].plot(range(6), finite, 'o', markerfacecolor='none', label='Independent gradient FD')
    axes[1, 0].set_xticks(range(6), LABELS, rotation=45)
    axes[1, 0].set_ylabel('Φ(q) · (H x) (Eh/bohr²)')
    axes[1, 0].set_title('Only this column has independent FD')
    axes[1, 0].legend(frameon=False, fontsize=9)
    axes[1, 1].loglog([row['step_bohr'] for row in rows],
        [row['max_native_error_hartree_per_bohr2'] for row in rows], 'o-', color='#246b8e')
    axes[1, 1].axhline(reference['gates']['hessian_hartree_per_bohr2'], linestyle='--',
                       color='#b44738', label='Predeclared gate')
    axes[1, 1].set_xlabel('Central-difference displacement (bohr)')
    axes[1, 1].set_ylabel('Maximum column error (Eh/bohr²)')
    axes[1, 1].set_title('Two independently reconverged steps')
    axes[1, 1].legend(frameon=False)
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.grid(alpha=.2)
    fig.suptitle('LiH q = (0, 0, ½): PBE / GTH-Pade / SZV, k2³ / FFT31³\n'
                 'Fixed a = 7.7800322767 bohr; finite-model curvatures', fontsize=12)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUTPUT/('lih_zone_boundary.'+suffix))
    plt.close(fig)
    with (OUTPUT/'zone_boundary_hx.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['component', 'native_Eh_bohr2', 'fd_h1e-4_Eh_bohr2', 'fd_h3e-4_Eh_bohr2'])
        for index, label in enumerate(LABELS):
            writer.writerow([label, column[index]]+[row['finite_difference_hartree_per_bohr2'][index] for row in rows])
    report = dict(passed=True, input_sha256={key:fingerprint(path) for key,path in paths.items()},
        script_sha256=fingerprint(Path(__file__)), frequencies_signed_cm1=frequencies.tolist(),
        negative_native_modes=int(np.count_nonzero(eigenvalues < 0)),
        independent_columns=['H x'], asr_projected=False,
        scope='Raw finite-model q-half matrix and selected-column independent check; no complete dispersion or material stability claim.')
    (OUTPUT/'plot_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Signed raw q-half frequencies (cm^-1):', frequencies)
    print('Saved raw matrix and H x comparison figures', flush=True)


def rayleigh():
    """Independently test the fixed H transverse curvature, keeping its sign."""
    path = OUTPUT/'pyscf_pbe_szv_k2_fft31_qhalf_full_rayleigh_fd.json'
    reference = json.loads(path.read_text())
    native_path = OUTPUT/'pbe_szv_k2_fft31_qhalf_full.json'
    native = json.loads(native_path.read_text())
    assert reference['complete'] and reference['passed'] and not reference['probe']
    assert reference['native_sha256'] == hashlib.sha256(native_path.read_bytes()).hexdigest()
    direction = np.array([[0., 0., 0.], [1., -1., 0.]])/np.sqrt(2.)
    np.testing.assert_array_equal(reference['requested_direction'], direction)
    phi = np.asarray(native['force_constant_actions_real_hartree_per_bohr2'])
    expected = float(direction.ravel()@phi@direction.ravel())
    np.testing.assert_allclose(reference['native_rayleigh_hartree_per_bohr2'], expected, atol=1e-14, rtol=0)
    rows = sorted(reference['columns'], key=lambda row:row['step_bohr'])
    values = [expected]+[row['rayleigh_hartree_per_bohr2'] for row in rows]
    assert len(rows) == 2 and all(value < 0 for value in values)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    labels = ['GradSCF']+[f'PySCF h={row["step_bohr"]:.0e}' for row in rows]
    axes[0].barh(labels, values, color=['#246b8e', '#c67d2c', '#c67d2c'])
    axes[0].axvline(0., color='black', linewidth=.8)
    axes[0].set_xlabel('Signed Rayleigh curvature (Eh/bohr²)')
    axes[0].set_title('The independently checked direction is negative')
    steps = [row['step_bohr'] for row in rows]
    axes[1].loglog(steps, [abs(row['rayleigh_hartree_per_bohr2']-expected) for row in rows],
                   'o-', color='#246b8e', label='Scalar curvature')
    axes[1].loglog(steps, [row['max_native_error_hartree_per_bohr2'] for row in rows],
                   's-', color='#c67d2c', label='Full action, maximum component')
    axes[1].axhline(reference['gates']['rayleigh_hartree_per_bohr2'], color='#b44738',
                    linestyle='--', label='Predeclared gate')
    axes[1].set_xlabel('Central-difference displacement (bohr)')
    axes[1].set_ylabel('Absolute error (Eh/bohr²)')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Fixed LiH model: q = (0, 0, ½), uLi = 0, uH = (1, −1, 0)/√2\n'
                 'Independent negative-direction check; no claim about converged bulk LiH', fontsize=11)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUTPUT/('lih_negative_direction.'+suffix))
    plt.close(fig)
    report = dict(passed=True, native_rayleigh_hartree_per_bohr2=expected,
        independent_rayleigh_hartree_per_bohr2=values[1:],
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='One fixed negative Cartesian direction of the specified finite model; no basis/k/FFT or material stability conclusion.')
    (OUTPUT/'rayleigh_plot_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Native / independent transverse curvatures (Eh/bohr^2):', values, flush=True)


if __name__ == '__main__':
    main()
    rayleigh()
