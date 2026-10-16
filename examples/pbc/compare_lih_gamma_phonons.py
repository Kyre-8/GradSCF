"""Independent PySCF gradient differences for the raw LiH Gamma Hessian.

Every displaced reference reconverges SCF at the saved basis, GTH potential,
FFT mesh and ordered fractional k points. No GradSCF physics is imported.
The raw FD matrices retain their asymmetry and translation defects.
"""
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf.pbc import dft, gto
from pyscf.pbc.dft.gen_grid import UniformGrids


OUTPUT = Path('reproducibility/ad_showcase/P03_lih_gamma_phonons')
STEPS_BOHR = (3e-4, 1e-4)


def calculate(native, coordinates, initial_density=None):
    started = perf_counter()
    settings = native['settings']
    assert settings['xc'] == 'pbe'
    cell = gto.M(atom=list(zip(['Li', 'H'], coordinates)), a=native['lattice_bohr'],
        unit='Bohr', basis=native['basis_data'],
        pseudo=dict(zip(['Li', 'H'], native['pseudopotential_data'])),
        mesh=settings['mesh'], precision=settings['precision'], cart=True, spin=0, verbose=0)
    mean_field = dft.KRKS(cell, kpts=cell.get_abs_kpts(native['fractional_kpoints']), exxdiv='ewald')
    mean_field.xc = 'pbe'
    mean_field.grids = UniformGrids(cell)
    mean_field.grids.mesh = np.asarray(settings['mesh'])
    mean_field.with_df.mesh = np.asarray(settings['mesh'])
    mean_field.conv_tol, mean_field.conv_tol_grad, mean_field.max_cycle = 1e-13, 1e-10, 150
    energy = mean_field.kernel(dm0=initial_density)
    density, overlap = np.asarray(mean_field.make_rdm1()), np.asarray(mean_field.get_ovlp())
    energies = np.asarray(mean_field.mo_energy)
    gap = float(energies[:, 2:].min()-energies[:, :2].max())
    residual = float(np.linalg.norm(mean_field.get_grad(mean_field.mo_coeff, mean_field.mo_occ, mean_field.get_fock())))
    number = float(np.einsum('kij,kji->', density, overlap).real/len(overlap))
    assert mean_field.converged and residual < 2e-8 and gap > 1e-3
    assert cell.nelectron == native['explicit_nelectron'] == 4
    np.testing.assert_allclose(number, 4., atol=1e-9, rtol=0)
    eigenvalues = np.linalg.eigvalsh(overlap)
    assert eigenvalues.min() > 1e-8*max(1., eigenvalues.max())
    np.testing.assert_array_equal(cell.atom_charges(), native['pseudopotential_valence_charges'])
    gradient = np.asarray(mean_field.nuc_grad_method().kernel())
    assert np.isfinite(gradient).all()
    return dict(energy_hartree=float(energy), gradient_hartree_per_bohr=gradient.tolist(),
        converged=True, orbital_gradient_norm=residual, sampled_indirect_gap_hartree=gap,
        electron_number=number, elapsed_seconds=perf_counter()-started), density


def run(name='pbe_szv_k2_fft31'):
    started = perf_counter()
    native_path = OUTPUT/(name+'.json')
    native = json.loads(native_path.read_text())
    assert native['complete'] and native['passed'] and not native['asr_projected']
    native_hash = hashlib.sha256(native_path.read_bytes()).hexdigest()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path = OUTPUT/('pyscf_'+name+'.json')
    state_path = OUTPUT/('pyscf_'+name+'_state.npz')
    coordinates = np.asarray(native['coordinates_bohr'])
    phi = np.asarray(native['force_constants_real_hartree_per_bohr2'])
    np.testing.assert_allclose(native['force_constants_imag_hartree_per_bohr2'], 0., atol=1e-10, rtol=0)
    if path.exists():
        report = json.loads(path.read_text())
        assert report['native_sha256'] == native_hash and report['script_sha256'] == script_hash
        density = np.load(state_path)['density']
    else:
        reference, density = calculate(native, coordinates)
        report = dict(complete=False, native_sha256=native_hash, script_sha256=script_hash,
            reference=reference, steps_bohr=STEPS_BOHR, columns=[])
        np.savez_compressed(state_path, density=density)
        path.write_text(json.dumps(report, indent=2)+'\n')
    for step in STEPS_BOHR:
        for column, direction in enumerate(np.eye(6).reshape(6, 2, 3)):
            if any(row['step_bohr'] == step and row['column'] == column for row in report['columns']):
                continue
            plus, plus_density = calculate(native, coordinates+step*direction, density)
            minus, _ = calculate(native, coordinates-step*direction, density)
            finite = (np.asarray(plus['gradient_hartree_per_bohr'])
                      -np.asarray(minus['gradient_hartree_per_bohr']))/(2*step)
            report['columns'].append(dict(step_bohr=step, column=column,
                finite_difference_hartree_per_bohr2=finite.ravel().tolist(),
                max_native_error_hartree_per_bohr2=float(np.max(abs(finite.ravel()-phi[:, column]))),
                plus=plus, minus=minus))
            if step == 1e-4 and column == 3:
                cold, cold_density = calculate(native, coordinates+step*direction)
                report['displaced_cold_warm_check'] = dict(
                    energy_hartree=abs(cold['energy_hartree']-plus['energy_hartree']),
                    relative_density=float(np.linalg.norm(cold_density-plus_density)/(1+np.linalg.norm(plus_density))))
            path.write_text(json.dumps(report, indent=2)+'\n')
            print('Gradient FD:', step, column,
                  report['columns'][-1]['max_native_error_hartree_per_bohr2'], flush=True)
    matrices = []
    translations = np.tile(np.eye(3), (2, 1))
    for step in STEPS_BOHR:
        columns = sorted((row for row in report['columns'] if row['step_bohr'] == step), key=lambda row: row['column'])
        matrix = np.stack([row['finite_difference_hartree_per_bohr2'] for row in columns], axis=1)
        matrices.append(matrix)
    reference = report['reference']
    errors = dict(energy_hartree=abs(reference['energy_hartree']-native['energy_hartree']),
        gradient_hartree_per_bohr=float(np.max(abs(np.asarray(reference['gradient_hartree_per_bohr'])
                                                   +np.asarray(native['reference_forces_hartree_per_bohr'])))),
        hessian_hartree_per_bohr2=[float(np.max(abs(matrix-phi))) for matrix in matrices],
        fd_step_change_hartree_per_bohr2=float(np.max(abs(matrices[0]-matrices[1]))))
    gates = dict(energy_hartree=2e-6, gradient_hartree_per_bohr=3e-6,
                 hessian_hartree_per_bohr2=5e-6, fd_step_change_hartree_per_bohr2=5e-6)
    branch = report['displaced_cold_warm_check']
    passed = (errors['energy_hartree'] < gates['energy_hartree']
        and errors['gradient_hartree_per_bohr'] < gates['gradient_hartree_per_bohr']
        and max(errors['hessian_hartree_per_bohr2']) < gates['hessian_hartree_per_bohr2']
        and errors['fd_step_change_hartree_per_bohr2'] < gates['fd_step_change_hartree_per_bohr2']
        and branch['energy_hartree'] < 1e-9 and branch['relative_density'] < 1e-7)
    report.update(complete=True, passed=bool(passed), errors=errors, gates=gates,
        raw_hessians_hartree_per_bohr2=[matrix.tolist() for matrix in matrices],
        raw_symmetry_error_hartree_per_bohr2=[float(np.linalg.norm(matrix-matrix.T)) for matrix in matrices],
        raw_asr_residual_hartree_per_bohr2=[float(np.linalg.norm(matrix@translations)) for matrix in matrices],
        elapsed_seconds_this_invocation=perf_counter()-started, pyscf=pyscf.__version__,
        hostname=platform.node(), python=platform.python_version(),
        scope='Independent, reconverged gradient FD at a fixed finite periodic model; no ASR or atom-only rotation projection.')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print('Independent Gamma Hessian:', report['passed'], errors, flush=True)
    assert passed, errors
    return report


if __name__ == '__main__':
    run()

# Measured on jy_gpu_1, CPU float64, 2026-10-16.
# run(): all 12 independently reconverged column/step pairs.
# Independent Gamma Hessian: True {'energy_hartree': 5.379891732104625e-08, 'gradient_hartree_per_bohr': 1.8140837598305126e-14, 'hessian_hartree_per_bohr2': [1.9429991802294433e-06, 1.9644777531566127e-06], 'fd_step_change_hartree_per_bohr2': 2.149731380407527e-08}
