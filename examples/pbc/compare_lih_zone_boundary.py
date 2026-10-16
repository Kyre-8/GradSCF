"""Independent PySCF folding and gradient-FD checks for the LiH q-half mode.

A 1x1x2 supercell has k2x2x1 and FFT(mx,my,2*mz). The raw alternating
displacements are u_l=(-1)^l u, with Phi(q)u=(g'_0-g'_1)/2. No ASR or
extra factor of two is applied. The FFT31 primitive reference is reused from
the completed independent Gamma calculation after verifying all model data.
"""
import hashlib
import json
import platform
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf.pbc import dft, gto
from pyscf.pbc.dft.gen_grid import UniformGrids


OUTPUT = Path('reproducibility/ad_showcase/P03_lih_zone_boundary')
GAMMA = Path('reproducibility/ad_showcase/P03_lih_gamma_phonons')
STEPS_BOHR = (3e-4, 1e-4)
GATES = dict(folding_energy_hartree_per_cell=1e-8, folding_relative_density=1e-7,
             native_energy_hartree=2e-6, hessian_hartree_per_bohr2=5e-6,
             fd_step_change_hartree_per_bohr2=5e-6,
             rayleigh_hartree_per_bohr2=5e-6, rayleigh_step_change_hartree_per_bohr2=5e-6)


def save(path, value):
    """Replace one complete JSON checkpoint atomically on the same filesystem."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def folded_kpoints(fractional):
    """Group primitive k points modulo the doubled cell reciprocal lattice."""
    fractional = np.asarray(fractional)
    wrapped = np.mod(np.round(fractional*np.array([1., 1., 2.]), 12), 1.)
    super_k, groups = np.unique(wrapped, axis=0, return_inverse=True)
    assert len(super_k) == 4 and np.all(np.bincount(groups) == 2)
    return super_k, groups


def calculate(native, coordinates, *, doubled=False, initial_density=None):
    """Independent Gaussian lattice integrals and reconverged analytic gradients."""
    started = perf_counter()
    settings = native['settings']
    repeats = 2 if doubled else 1
    lattice = np.diag([1, 1, repeats])@np.asarray(native['lattice_bohr'])
    fractional = np.asarray(native['fractional_kpoints'])
    if doubled:
        fractional, _ = folded_kpoints(fractional)
    mesh = np.asarray(settings['mesh'])*[1, 1, repeats]
    symbols = ['Li', 'H']*repeats
    cell = gto.M(atom=list(zip(symbols, coordinates)), a=lattice, unit='Bohr',
        basis=native['basis_data'], pseudo=dict(zip(['Li', 'H'], native['pseudopotential_data'])),
        mesh=mesh, precision=settings['precision'], cart=True, spin=0, verbose=0)
    mf = dft.KRKS(cell, kpts=cell.get_abs_kpts(fractional), exxdiv='ewald')
    mf.xc = 'pbe'
    mf.grids = UniformGrids(cell); mf.grids.mesh = mesh
    mf.with_df.mesh = mesh
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 150
    energy = mf.kernel(dm0=initial_density)
    density, overlap = np.asarray(mf.make_rdm1()), np.asarray(mf.get_ovlp())
    nocc = 2*repeats
    energies = np.asarray(mf.mo_energy)
    gap = float(energies[:, nocc:].min()-energies[:, :nocc].max())
    residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, mf.get_fock())))
    number = float(np.einsum('kij,kji->', density, overlap).real/len(overlap))
    eigenvalues = np.linalg.eigvalsh(overlap)
    assert mf.converged and residual < 2e-8 and gap > 1e-3
    assert cell.nelectron == 4*repeats
    np.testing.assert_allclose(number, cell.nelectron, atol=1e-9, rtol=0)
    assert eigenvalues.min() > 1e-8*max(1., eigenvalues.max())
    np.testing.assert_array_equal(cell.atom_charges(), native['pseudopotential_valence_charges']*repeats)
    gradient = np.asarray(mf.nuc_grad_method().kernel())
    assert np.isfinite(gradient).all()
    return dict(energy_hartree=float(energy), gradient_hartree_per_bohr=gradient.tolist(),
        converged=True, electron_number=number, orbital_gradient_norm=residual,
        sampled_indirect_gap_hartree=gap, minimum_overlap_eigenvalue=float(eigenvalues.min()),
        mesh=mesh.tolist(), fractional_kpoints=fractional.tolist(), elapsed_seconds=perf_counter()-started), density


def primitive_reference(native):
    """Reuse completed FFT31 independent work; never mix native densities into it."""
    name = 'pbe_szv_k2_fft31'
    paths = [GAMMA/(name+'.json'), GAMMA/('pyscf_'+name+'.json'), GAMMA/('pyscf_'+name+'_state.npz')]
    prior, reference = (json.loads(path.read_text()) for path in paths[:2])
    for key in ['settings', 'lattice_bohr', 'coordinates_bohr', 'fractional_kpoints',
                'basis_data', 'pseudopotential_data', 'explicit_nelectron']:
        assert native[key] == prior[key], key
    assert reference['complete'] and reference['passed']
    assert reference['native_sha256'] == hashlib.sha256(paths[0].read_bytes()).hexdigest()
    density = np.load(paths[2])['density']
    return reference['reference'], density, {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run(name='pbe_szv_k2_fft31_qhalf_probe', *, probe=True, direction=None, reference_name=None):
    """Check saved columns, or a specified unit real direction of a full matrix.

    reference_name reuses an already checked folding probe of the exact same
    physical model. A chosen Rayleigh direction has a separate output/cache;
    changing it cannot silently resume the preceding direction's endpoints.
    """
    started = perf_counter()
    native_path = OUTPUT/(name+'.json')
    native = json.loads(native_path.read_text())
    assert native['complete'] and native['passed'] and not native['asr_projected']
    projected = direction is not None
    directions = np.asarray(native['directions'])
    actions = np.asarray(native['force_constant_actions_real_hartree_per_bohr2'])
    if projected:
        assert not probe
        direction = np.asarray(direction)
        assert direction.shape == (2, 3) and not np.iscomplexobj(direction) and np.isfinite(direction).all()
        np.testing.assert_allclose(np.linalg.norm(direction), 1., atol=1e-12, rtol=0)
        np.testing.assert_array_equal(directions, np.eye(6).reshape(6, 2, 3))
        directions, actions = direction[None], (actions@direction.ravel())[:, None]
    reference_name = name if reference_name is None else reference_name
    assert not probe or reference_name == name
    probe_path = OUTPUT/('pyscf_'+reference_name+'.json')
    suffix = '_rayleigh_fd' if projected else '_fd'
    path = probe_path if probe else OUTPUT/('pyscf_'+name+suffix+'.json')
    state_path = OUTPUT/('pyscf_'+reference_name+'_state.npz')
    fingerprints = dict(native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        requested_direction=None if not projected else direction.tolist())
    lattice, coordinates = np.asarray(native['lattice_bohr']), np.asarray(native['coordinates_bohr'])
    positions = (coordinates[None]+np.arange(2)[:, None, None]*lattice[2]).reshape(4, 3)
    signs = np.array([1., -1.])
    if path.exists():
        report = json.loads(path.read_text())
        assert all(report[key] == value for key,value in fingerprints.items())
        assert report['state_sha256'] == hashlib.sha256(state_path.read_bytes()).hexdigest()
        density = np.load(state_path)['density']
    elif not probe:
        # Keep the completed folding probe immutable. The FD stage records its
        # lineage and current script, while reusing the same verified density.
        report = json.loads(probe_path.read_text())
        assert report['complete'] and report['passed'] and report['probe'] and report['folding_passed']
        previous_native = OUTPUT/(reference_name+'.json')
        assert report['native_sha256'] == hashlib.sha256(previous_native.read_bytes()).hexdigest()
        previous = json.loads(previous_native.read_text())
        for key in ['settings', 'lattice_bohr', 'coordinates_bohr', 'fractional_kpoints',
                    'basis_data', 'pseudopotential_data', 'explicit_nelectron', 'source_python_sha256']:
            assert native[key] == previous[key], key
        assert report['state_sha256'] == hashlib.sha256(state_path.read_bytes()).hexdigest()
        density = np.load(state_path)['density']
        report.update(**fingerprints, complete=False, passed=False, probe=False,
            parent_probe_sha256=hashlib.sha256(probe_path.read_bytes()).hexdigest(),
            reference_native_sha256=hashlib.sha256(previous_native.read_bytes()).hexdigest(), columns=[])
        report['errors']['native_energy_hartree'] = abs(report['primitive_reference']['energy_hartree']-native['energy_hartree'])
        report['gates'] = GATES
        save(path, report)
    else:
        primitive, small_density, previous_hashes = primitive_reference(native)
        super_k, groups = folded_kpoints(native['fractional_kpoints'])
        primitive_k = np.asarray(native['fractional_kpoints'])@(2*np.pi*np.linalg.inv(lattice).T)
        blocks = []
        for group in range(len(super_k)):
            selected = np.flatnonzero(groups == group)
            phase = np.exp(1j*np.arange(2)[:, None]*(lattice[2]@primitive_k[selected].T))/np.sqrt(2.)
            transformation = np.kron(phase, np.eye(3))
            diagonal = np.zeros((6, 6), complex)
            diagonal[:3, :3], diagonal[3:, 3:] = small_density[selected]
            blocks.append(transformation@diagonal@transformation.conj().T)
        folded_density = np.stack(blocks)
        reference, density = calculate(native, positions, doubled=True)
        errors = dict(folding_energy_hartree_per_cell=abs(reference['energy_hartree']/2-primitive['energy_hartree']),
            folding_relative_density=float(np.linalg.norm(density-folded_density)/(1+np.linalg.norm(folded_density))),
            native_energy_hartree=abs(primitive['energy_hartree']-native['energy_hartree']))
        passed = all(errors[key] < GATES[key] for key in errors)
        report = dict(complete=not passed, passed=False, folding_passed=bool(passed), **fingerprints,
            primitive_reference=primitive, reference=reference, reference_input_sha256=previous_hashes,
            errors=errors, gates=GATES, steps_bohr=STEPS_BOHR, columns=[],
            convention='u_l=(-1)^l u; Phi(q)u=(g_prime_0-g_prime_1)/2; lattice-translation phase, no atomic-position phase.',
            elapsed_seconds_this_invocation=perf_counter()-started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            note='The doubled FFT z dimension is even; its Nyquist support differs from a finite primitive G-window. Failed folding gates are retained.')
        np.savez_compressed(state_path, density=density, folded_density=folded_density)
        report['state_sha256'] = hashlib.sha256(state_path.read_bytes()).hexdigest()
        save(path, report)
        print('Independent two-cell folding:', bool(passed), errors, flush=True)
    assert report['folding_passed'], report['errors']
    if not probe:
        # A successful folding probe does not make subsequent FD complete.
        # Persist the new stage before any endpoint can time out or fail.
        report.update(complete=False, passed=False, probe=False)
        save(path, report)
        endpoint_directory = OUTPUT/(path.stem+'_endpoints')
        endpoint_directory.mkdir(exist_ok=True)

        def endpoint(step, column, sign, displaced):
            destination = endpoint_directory/f'h{step:.0e}_c{column}_{sign:+d}.json'
            identity = dict(**fingerprints, state_sha256=report['state_sha256'],
                step_bohr=step, column=column, sign=sign, coordinates_bohr=displaced.tolist())
            if destination.exists():
                saved = json.loads(destination.read_text())
                assert saved['complete'] and all(saved[key] == value for key,value in identity.items())
                return saved['result']
            result, _ = calculate(native, displaced, doubled=True, initial_density=density)
            save(destination, dict(complete=True, **identity, result=result))
            print('Saved independent endpoint:', step, column, sign, flush=True)
            return result

        for step in STEPS_BOHR:
            for column, vector in enumerate(directions):
                if any(row['step_bohr'] == step and row['column'] == column for row in report['columns']):
                    continue
                pattern = (signs[:, None, None]*vector).reshape(4, 3)
                plus = endpoint(step, column, 1, positions+step*pattern)
                minus = endpoint(step, column, -1, positions-step*pattern)
                derivative = (np.asarray(plus['gradient_hartree_per_bohr'])-np.asarray(minus['gradient_hartree_per_bohr']))/(2*step)
                finite = np.einsum('l,lai->ai', signs/2, derivative.reshape(2, 2, 3)).ravel()
                report['columns'].append(dict(step_bohr=step, column=column, plus=plus, minus=minus,
                    finite_difference_hartree_per_bohr2=finite.tolist(),
                    max_native_error_hartree_per_bohr2=float(np.max(abs(finite-actions[:, column])))))
                if projected:
                    report['columns'][-1]['rayleigh_hartree_per_bohr2'] = float(vector.ravel()@finite)
                save(path, report)
                print('Alternating gradient FD:', step, column, report['columns'][-1]['max_native_error_hartree_per_bohr2'], flush=True)
        expected = {(step, column) for step in STEPS_BOHR for column in range(actions.shape[1])}
        actual = {(row['step_bohr'], row['column']) for row in report['columns']}
        assert actual == expected and len(report['columns']) == len(expected)
        matrices = [np.stack([row['finite_difference_hartree_per_bohr2'] for row in sorted(
            (row for row in report['columns'] if row['step_bohr'] == step), key=lambda row:row['column'])],
            axis=1) for step in STEPS_BOHR]
        report['errors'].update(hessian_hartree_per_bohr2=max(row['max_native_error_hartree_per_bohr2'] for row in report['columns']),
            fd_step_change_hartree_per_bohr2=float(np.max(abs(matrices[0]-matrices[1]))))
        if projected:
            native_rayleigh = float(direction.ravel()@actions[:, 0])
            curvatures = [row['rayleigh_hartree_per_bohr2'] for row in report['columns']]
            report['native_rayleigh_hartree_per_bohr2'] = native_rayleigh
            report['errors'].update(rayleigh_hartree_per_bohr2=max(abs(value-native_rayleigh) for value in curvatures),
                rayleigh_step_change_hartree_per_bohr2=abs(curvatures[0]-curvatures[1]))
    passed = all(value < GATES[key] for key,value in report['errors'].items())
    report.update(complete=True, passed=bool(passed), probe=probe,
        elapsed_seconds_this_invocation=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hostname=platform.node(), python=platform.python_version(), pyscf=pyscf.__version__,
        scope='Independent folding/resource probe only.' if probe else 'Independent self-inverse q action check; no general conjugacy or dispersion claim.')
    save(path, report)
    assert passed, report['errors']
    return report


if __name__ == '__main__':
    run()
