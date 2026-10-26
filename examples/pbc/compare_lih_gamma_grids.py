"""Predeclared raw Gamma FFT convergence gates at fixed LiH geometry.

These thresholds were fixed before the FFT41/51 calculations. Do not apply
ASR, remove negative curvatures or change the lattice minimum between meshes.
The independent gradient-FD reference is a separate same-FFT31 check.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


OUTPUT = Path('reproducibility/ad_showcase/P03_lih_gamma_phonons')
GATES = dict(last_force_constant_increment_hartree_per_bohr2=2e-6,
    last_optical_frequency_increment_cm1=1., final_raw_asr_frobenius_hartree_per_bohr2=1e-6,
    final_max_absolute_acoustic_frequency_cm1=1.)


def run(meshes=(31, 41, 51), *, method=None, output_name='grid_convergence.json'):
    """Apply the unchanged criteria to the last pair and finest raw matrix.

    A later method-specific extension gets its own output_name; the original
    failed FFT31/41/51 report remains an independent record.
    """
    if len(meshes) < 2 or list(meshes) != sorted(set(meshes)):
        raise ValueError('Supply at least two distinct increasing FFT meshes.')
    if method not in (None, 'direct', 'gmres'):
        raise ValueError('Select a recorded shared solver method or the original cases.')
    rows, matrices, hashes = [], [], {}
    baseline = None
    for mesh in meshes:
        name = f'pbe_szv_k2_fft{mesh}'+('' if method is None else '_'+method)
        path, state_path = OUTPUT/(name+'.json'), OUTPUT/(name+'.npz')
        record = json.loads(path.read_text())
        assert record['complete'] and record['passed'] and not record['asr_projected']
        if baseline is None:
            baseline = record
        for field in ['coordinates_bohr', 'lattice_bohr', 'fractional_kpoints', 'masses_amu']:
            np.testing.assert_allclose(record[field], baseline[field], atol=1e-13, rtol=0)
        for field in ['basis_data', 'pseudopotential_data', 'source_python_sha256']:
            assert record[field] == baseline[field]
        assert {key: value for key, value in record['settings'].items() if key != 'mesh'} == {
            key: value for key, value in baseline['settings'].items() if key != 'mesh'}
        if method is not None:
            assert record['linear_solver'] == baseline['linear_solver']
            assert record['linear_solver']['method'] == method
            assert record['script_sha256'] == baseline['script_sha256']
            assert record['helper_sha256'] == baseline['helper_sha256']
        assert record['settings']['mesh'] == [mesh]*3
        with np.load(state_path) as archive:
            phi, modes = archive['force_constants'], archive['modes']
        mass = np.repeat(np.asarray(record['masses_amu'])*record['atomic_mass_to_electron_mass'], 3)
        translations = np.tile(np.eye(3), (2, 1))
        mass_translations = np.sqrt(mass)[:, None]*translations
        mass_translations /= np.linalg.norm(mass_translations, axis=0)
        weights = np.sum(abs(mass_translations.T@modes)**2, axis=0)
        acoustic = np.argsort(weights)[-3:]
        optical = np.array([index for index in range(6) if index not in acoustic])
        weight_gap = float(np.min(weights[acoustic])-np.max(weights[optical]))
        # A translation-overlap gap, not frequency order, identifies the group.
        # Ambiguous mixtures cannot be reported as resolved acoustic modes.
        assert weight_gap > .5, 'Acoustic/optical groups are not resolved by translation overlap.'
        frequencies = np.asarray(record['frequencies_signed_cm1'])
        raw_asr = float(np.linalg.norm(phi@translations))
        np.testing.assert_allclose(raw_asr, record['raw_asr_residual_hartree_per_bohr2'], atol=1e-12, rtol=0)
        rows.append(dict(mesh=mesh, case=name, acoustic_modes=acoustic.tolist(), optical_modes=optical.tolist(),
            translation_weights=weights.tolist(), translation_weight_gap=weight_gap,
            acoustic_frequencies_signed_cm1=np.sort(frequencies[acoustic]).tolist(),
            optical_frequencies_signed_cm1=np.sort(frequencies[optical]).tolist(),
            raw_asr_frobenius_hartree_per_bohr2=raw_asr,
            peak_rss_kib=record['peak_rss_kib'], elapsed_seconds=record['elapsed_seconds']))
        matrices.append(phi)
        hashes[name] = dict(json=hashlib.sha256(path.read_bytes()).hexdigest(),
                            npz=hashlib.sha256(state_path.read_bytes()).hexdigest())
    metrics = dict(last_force_constant_increment_hartree_per_bohr2=float(np.max(abs(matrices[-1]-matrices[-2]))),
        last_optical_frequency_increment_cm1=float(np.max(abs(np.asarray(rows[-1]['optical_frequencies_signed_cm1'])
                                                            -np.asarray(rows[-2]['optical_frequencies_signed_cm1'])))),
        final_raw_asr_frobenius_hartree_per_bohr2=rows[-1]['raw_asr_frobenius_hartree_per_bohr2'],
        final_max_absolute_acoustic_frequency_cm1=float(np.max(abs(np.asarray(rows[-1]['acoustic_frequencies_signed_cm1'])))))
    checks = {key: metrics[key] < limit for key, limit in GATES.items()}
    report = dict(complete=True, passed=all(checks.values()), rows=rows, metrics=metrics,
        gates=GATES, checks=checks, input_sha256=hashes, meshes=list(meshes), method=method,
        figure_stem='lih_gamma_'+Path(output_name).stem,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Fixed a=7.780032276700287 bohr, PBE/GTH-Pade/SZV, k2^3. Raw Gamma FFT convergence only; no ASR, k/basis convergence, finite-q or LO-TO claim.')
    (OUTPUT/output_name).write_text(json.dumps(report, indent=2)+'\n')
    print('Raw Gamma FFT gates:', json.dumps(checks), metrics, flush=True)
    assert report['passed'], metrics
    return report


if __name__ == '__main__':
    run()

# Measured on jy_gpu_1, CPU float64, 2026-10-16.
# run((51, 61, 71), method='gmres', output_name='grid_extension_to71.json')
# Raw Gamma FFT gates: {"last_force_constant_increment_hartree_per_bohr2": true, "last_optical_frequency_increment_cm1": true, "final_raw_asr_frobenius_hartree_per_bohr2": true, "final_max_absolute_acoustic_frequency_cm1": true} {'last_force_constant_increment_hartree_per_bohr2': 4.02835123411438e-07, 'last_optical_frequency_increment_cm1': 0.0001752600553004413, 'final_raw_asr_frobenius_hartree_per_bohr2': 3.596611608262572e-08, 'final_max_absolute_acoustic_frequency_cm1': 0.23125338797386674}
