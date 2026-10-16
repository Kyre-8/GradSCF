"""Same-source direct/GMRES comparison before increasing the LiH FFT mesh.

Fresh processes use the same FFT31 geometry, SCF settings and response
tolerances. Only the shared linear-solver method differs. No solver is defined
in this example; the native Phonons driver checks the actual linear residual.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


OUTPUT = Path('reproducibility/ad_showcase/P03_lih_gamma_phonons')
GATES = dict(force_constants_hartree_per_bohr2=1e-8,
             signed_frequencies_cm1=1e-4, raw_asr_hartree_per_bohr2=1e-8,
             baseline_force_constants_hartree_per_bohr2=1e-10)


def run():
    reports, fingerprints = [], {}
    for method in ['direct', 'gmres']:
        name = f'pbe_szv_k2_fft31_{method}'
        path = OUTPUT/(name+'.json')
        report = json.loads(path.read_text())
        assert report['complete'] and report['passed'] and not report['asr_projected']
        assert report['linear_solver']['method'] == method
        assert np.isfinite(report['linear_residual_norm'])
        reports.append(report)
        fingerprints[method] = hashlib.sha256(path.read_bytes()).hexdigest()
    direct, iterative = reports
    for key in ['settings', 'basis_data', 'pseudopotential_data', 'masses_amu',
                'source_python_sha256', 'script_sha256', 'helper_sha256', 'workspace_limits']:
        assert direct[key] == iterative[key], key
    for key in ['coordinates_bohr', 'lattice_bohr', 'fractional_kpoints']:
        np.testing.assert_allclose(direct[key], iterative[key], atol=1e-13, rtol=0)
    assert {key: value for key, value in direct['linear_solver'].items() if key != 'method'} == {
        key: value for key, value in iterative['linear_solver'].items() if key != 'method'}
    matrices = [np.asarray(row['force_constants_real_hartree_per_bohr2'])
                +1j*np.asarray(row['force_constants_imag_hartree_per_bohr2']) for row in reports]
    baseline_path = OUTPUT/'pbe_szv_k2_fft31.json'
    baseline = json.loads(baseline_path.read_text())
    assert baseline['complete'] and baseline['passed'] and baseline['settings'] == direct['settings']
    assert baseline['source_python_sha256'] == direct['source_python_sha256']
    baseline_matrix = (np.asarray(baseline['force_constants_real_hartree_per_bohr2'])
                       +1j*np.asarray(baseline['force_constants_imag_hartree_per_bohr2']))
    errors = dict(force_constants_hartree_per_bohr2=float(np.max(abs(matrices[1]-matrices[0]))),
        signed_frequencies_cm1=float(np.max(abs(np.asarray(iterative['frequencies_signed_cm1'])
                                                -np.asarray(direct['frequencies_signed_cm1'])))),
        raw_asr_hartree_per_bohr2=abs(iterative['raw_asr_residual_hartree_per_bohr2']
                                     -direct['raw_asr_residual_hartree_per_bohr2']),
        baseline_force_constants_hartree_per_bohr2=float(np.max(abs(matrices[0]-baseline_matrix))))
    report = dict(complete=True, passed=all(errors[key] < value for key, value in GATES.items()),
        errors=errors, gates=GATES, input_sha256=fingerprints,
        baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        source_python_sha256=direct['source_python_sha256'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        methods={method: {key: row[key] for key in ['linear_solver', 'linear_residual_norm',
                    'elapsed_seconds', 'phonon_seconds_includes_compilation', 'peak_rss_kib']}
                 for method, row in zip(['direct', 'gmres'], reports)},
        scope='Same finite FFT31 model; fresh-process total times and high-water RSS, not exclusive-core or general scaling results.')
    (OUTPUT/'solver_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Direct/GMRES Gamma agreement:', report['passed'], errors, flush=True)
    assert report['passed'], errors
    return report


if __name__ == '__main__':
    run()

# Measured on jy_gpu_1, CPU float64, 2026-10-16.
# run()
# Direct/GMRES Gamma agreement: True {'force_constants_hartree_per_bohr2': 4.73232564246473e-15, 'signed_frequencies_cm1': 9.234213393938262e-11, 'raw_asr_hartree_per_bohr2': 1.7316877098938477e-15, 'baseline_force_constants_hartree_per_bohr2': 0.0}
