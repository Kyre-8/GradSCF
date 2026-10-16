"""Independent PySCF FFTDF references for the LiH P01/P02 showcase.

Ordinary Python settings; no GradSCF physics is imported. The initial pilot
screens a two-atom rocksalt primitive cell before any lattice minimum is claimed.
"""
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf.pbc import dft, gto, scf
from pyscf.pbc.dft.gen_grid import UniformGrids


OUTPUT = Path('reproducibility/ad_showcase/P01_P02_lih_periodic')
PILOT = dict(xc='pbe', basis='gth-szv', pseudo='gth-pade',
             mesh=[31, 31, 31], kmesh=[2, 2, 2], precision=1e-10)


def primitive_geometry(lattice_constant):
    """Row FCC primitive vectors; Li/H fractional positions 0 and (1/2)^3."""
    lattice = .5*lattice_constant*np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    coordinates = np.array([[0., 0., 0.], [.5, .5, .5]])@lattice
    return coordinates, lattice


def calculate_reference(settings, *, coordinates=None, lattice=None, initial_density=None,
                        fractional_kpoints=None, properties=False):
    started = perf_counter()
    if coordinates is None or lattice is None:
        coordinates, lattice = primitive_geometry(settings['a_bohr'])
    cell = gto.M(atom=list(zip(['Li', 'H'], np.asarray(coordinates))), a=np.asarray(lattice),
                 unit='Bohr', basis=settings['basis'], pseudo=settings['pseudo'],
                 mesh=settings['mesh'], precision=settings['precision'], cart=True, spin=0, verbose=0)
    kpts = cell.make_kpts(settings['kmesh']) if fractional_kpoints is None else cell.get_abs_kpts(fractional_kpoints)
    mean_field = scf.KRHF(cell, kpts=kpts, exxdiv='ewald') if settings['xc'] == 'hf' else dft.KRKS(cell, kpts=kpts, exxdiv='ewald')
    if settings['xc'] != 'hf':
        mean_field.xc = settings['xc']
        mean_field.grids = UniformGrids(cell)
        mean_field.grids.mesh = np.asarray(settings['mesh'])
    mean_field.with_df.mesh = np.asarray(settings['mesh'])
    mean_field.conv_tol, mean_field.conv_tol_grad, mean_field.max_cycle = 1e-12, 1e-9, 150
    energy = mean_field.kernel(dm0=initial_density)
    density, overlap = np.asarray(mean_field.make_rdm1()), np.asarray(mean_field.get_ovlp())
    energies, occupations = np.asarray(mean_field.mo_energy), np.asarray(mean_field.mo_occ)
    assert cell.nelectron % 2 == 0
    nocc = cell.nelectron//2
    gap = float(np.min(energies[:, nocc])-np.max(energies[:, nocc-1]))
    residual = float(np.linalg.norm(mean_field.get_grad(mean_field.mo_coeff, mean_field.mo_occ, mean_field.get_fock())))
    number = float(np.einsum('kij,kji->', density, overlap).real/len(kpts))
    record = dict(settings=dict(settings), energy_hartree=float(energy), converged=bool(mean_field.converged),
        orbital_gradient_norm=residual, sampled_indirect_gap_hartree=gap,
        occupation_values=np.unique(occupations).tolist(), electron_number=number,
        explicit_nelectron=int(cell.nelectron), pseudopotential_valence_charges=cell.atom_charges().tolist(),
        nao=cell.nao_nr(), volume_bohr3=float(cell.vol), coordinates_bohr=np.asarray(coordinates).tolist(),
        lattice_bohr=np.asarray(lattice).tolist(), kpoints_inverse_bohr=np.asarray(kpts).tolist(),
        fractional_kpoints=(np.asarray(kpts)@np.asarray(lattice).T/(2*np.pi)).tolist(),
        basis_data=cell._basis, pseudopotential_data=[cell._pseudo[symbol] for symbol in ['Li', 'H']],
        minimum_overlap_eigenvalue=float(np.linalg.eigvalsh(overlap).min()),
        elapsed_seconds=perf_counter()-started)
    if not mean_field.converged or residual > 2e-8:
        raise RuntimeError(f'Independent SCF is not resolved: {record}')
    np.testing.assert_allclose(number, cell.nelectron, atol=1e-9, rtol=0)
    assert set(np.unique(occupations)).issubset({0., 2.})
    if properties:
        beginning = perf_counter()
        gradient = mean_field.nuc_grad_method()
        record['forces_hartree_per_bohr'] = (-np.asarray(gradient.kernel())).tolist()
        if settings['xc'] != 'hf':
            # This public PySCF routine includes numerical one-body strain
            # derivatives. Retain its raw tensor and verify it with resolved
            # energy finite differences, rather than calling it purely analytic.
            raw = np.asarray(gradient.get_stress())
            record['raw_public_stress_hartree_per_bohr3'] = raw.tolist()
            record['symmetric_public_stress_hartree_per_bohr3'] = (.5*(raw+raw.T)).tolist()
            record['raw_stress_asymmetry'] = float(np.max(abs(raw-raw.T)))
        record['property_seconds'] = perf_counter()-beginning
    record['total_seconds'] = perf_counter()-started
    return record, density


def _same_data(left, right):
    """Compare actual basis/PP definitions, not only their family labels."""
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _same_data(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _same_data(first, second)
    else:
        np.testing.assert_allclose(left, right, atol=2e-13, rtol=2e-12)


def compare_case(name, *, finite_differences=False):
    """Independent complete tensor properties, with optional multi-step FD.

    Checkpoints permit a bounded run to resume without repeating completed FD
    pairs. Every displaced/strained energy reconverges from the independent
    reference density. One displaced cold solve checks that choice of seed.
    """
    started = perf_counter()
    native_path = OUTPUT/(name+'.json')
    native = json.loads(native_path.read_text())
    assert native['complete']
    native_hash = hashlib.sha256(native_path.read_bytes()).hexdigest()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination = OUTPUT/('pyscf_'+name+'.json')
    state_file = OUTPUT/('pyscf_'+name+'_state.npz')
    settings = native['settings']
    coordinates, lattice = np.asarray(native['coordinates_bohr']), np.asarray(native['lattice_bohr'])
    fractional = np.asarray(native['fractional_kpoints'])

    def save(report):
        destination.write_text(json.dumps(report, indent=2)+'\n')

    if destination.exists():
        report = json.loads(destination.read_text())
        assert report['native_sha256'] == native_hash and report['script_sha256'] == script_hash
        reference = report['reference']
        density = np.load(state_file)['density']
    else:
        reference, density = calculate_reference(settings, coordinates=coordinates, lattice=lattice,
            fractional_kpoints=fractional, properties=True)
        _same_data(native['basis_data'], reference['basis_data'])
        _same_data(native['pseudopotential_data'], reference['pseudopotential_data'])
        np.testing.assert_array_equal(native['pseudopotential_valence_charges'], reference['pseudopotential_valence_charges'])
        assert native['explicit_nelectron'] == reference['explicit_nelectron']
        np.savez_compressed(state_file, density=density)
        report = dict(complete=False, native_sha256=native_hash, script_sha256=script_hash,
                      reference=reference, finite_difference_checks=[])
        save(report)
    volume = float(np.linalg.det(lattice))
    native_force = np.asarray(native['forces_hartree_per_bohr'])
    reference_force = np.asarray(reference['forces_hartree_per_bohr'])
    native_stress = np.asarray(native['stress_hartree_per_bohr3'])
    public_stress = reference.get('symmetric_public_stress_hartree_per_bohr3')
    errors = dict(energy_hartree=abs(native['energy_hartree']-reference['energy_hartree']),
                  forces_hartree_per_bohr=float(np.max(abs(native_force-reference_force))))
    if public_stress is not None:
        errors['stress_hartree_per_bohr3'] = float(np.max(abs(native_stress-np.asarray(public_stress))))
    gates = dict(energy_hartree=2e-6, forces_hartree_per_bohr=3e-6, stress_hartree_per_bohr3=3e-7,
                 coordinate_direction_hartree_per_bohr=3e-6, strain_direction_hartree=5e-6)

    if finite_differences:
        coordinate_directions = {
            'internal': np.array([[-.15, .10, -.05], [.15, -.10, .05]]),
            'translation': np.tile([.25, .10, -.20], (2, 1))}
        if public_stress is not None:
            strain_directions = {'isotropic': np.eye(3), 'deviatoric': np.diag([1., -.5, -.5]),
                'shear': np.array([[0., .3, .2], [.3, 0., -.1], [.2, -.1, 0.]])}
        else:
            strain_directions = {}
            for i in range(3):
                for j in range(i, 3):
                    value = np.zeros((3, 3)); value[i, j] = value[j, i] = 1.
                    strain_directions[f'{i}{j}'] = value
        for kind, directions in [('coordinate', coordinate_directions), ('strain', strain_directions)]:
            for label, direction in directions.items():
                for step in [3e-4, 1e-4]:
                    key = f'{kind}:{label}:{step}'
                    if any(row['key'] == key for row in report['finite_difference_checks']):
                        continue
                    values, endpoints = [], []
                    for sign in [1., -1.]:
                        if kind == 'coordinate':
                            position, cell = coordinates+sign*step*direction, lattice
                        else:
                            deformation = np.eye(3)+sign*step*direction
                            position, cell = coordinates@deformation.T, lattice@deformation.T
                        result, moved_density = calculate_reference(settings, coordinates=position, lattice=cell,
                            fractional_kpoints=fractional, initial_density=density)
                        assert result['sampled_indirect_gap_hartree'] > 1e-3
                        values.append(result['energy_hartree'])
                        endpoints.append({field: result[field] for field in ['energy_hartree', 'converged',
                            'orbital_gradient_norm', 'sampled_indirect_gap_hartree', 'electron_number', 'elapsed_seconds']})
                        if kind == 'coordinate' and label == 'internal' and step == 1e-4 and sign == 1.:
                            cold, cold_density = calculate_reference(settings, coordinates=position, lattice=cell,
                                                                    fractional_kpoints=fractional)
                            branch = dict(energy_difference=abs(cold['energy_hartree']-result['energy_hartree']),
                                relative_density_difference=float(np.linalg.norm(cold_density-moved_density)/(1+np.linalg.norm(cold_density))))
                            assert branch['energy_difference'] < 1e-9 and branch['relative_density_difference'] < 1e-7
                            report['displaced_cold_warm_branch_check'] = branch
                    finite = (values[0]-values[1])/(2*step)
                    predicted = (-np.sum(native_force*direction) if kind == 'coordinate'
                                 else volume*np.sum(native_stress*direction))
                    independent_prediction = (-np.sum(reference_force*direction) if kind == 'coordinate' else
                                               None if public_stress is None else volume*np.sum(np.asarray(public_stress)*direction))
                    row = dict(key=key, kind=kind, direction=label, matrix=direction.tolist(), step=step,
                               finite_difference=float(finite), native_prediction=float(predicted),
                               native_error=float(abs(predicted-finite)), endpoints=endpoints)
                    if independent_prediction is not None:
                        row['public_property_error'] = float(abs(independent_prediction-finite))
                    report['finite_difference_checks'].append(row)
                    save(report)
                    print(name, key, 'AD-FD error', row['native_error'], flush=True)
        if public_stress is None:
            tensor = np.zeros((3, 3))
            for row in report['finite_difference_checks']:
                if row['kind'] == 'strain' and row['step'] == 1e-4:
                    i, j = map(int, row['direction'])
                    tensor[i, j] = tensor[j, i] = row['finite_difference']/volume/(1 if i == j else 2)
            report['stress_from_finite_difference_hartree_per_bohr3'] = tensor.tolist()
            errors['stress_hartree_per_bohr3'] = float(np.max(abs(native_stress-tensor)))
    passed = all(value < gates[key] for key, value in errors.items())
    if finite_differences:
        for row in report['finite_difference_checks']:
            if row['step'] == 1e-4:
                limit = gates['coordinate_direction_hartree_per_bohr' if row['kind'] == 'coordinate' else 'strain_direction_hartree']
                passed &= row['native_error'] < limit
                if 'public_property_error' in row:
                    passed &= row['public_property_error'] < limit
    report.update(complete=True, passed=bool(passed), errors=errors, gates=gates,
        finite_difference_validation=bool(finite_differences),
        scope='Matched energy/atomic gradient and public PBE stress; requested FD reconverges all electrons at fixed fractional k.',
        elapsed_seconds_this_invocation=perf_counter()-started, pyscf_version=pyscf.__version__,
        hostname=platform.node(), python=platform.python_version())
    save(report)
    print(name, 'independent comparison complete:', bool(passed), errors, 'FD:', finite_differences, flush=True)
    return report


def run_pilot():
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/'pilot_pyscf.json'
    report = dict(complete=False, stage='Independent candidate screening only', points=[])
    path.write_text(json.dumps(report, indent=2)+'\n')
    density, middle_density = None, None
    for index, value in enumerate(np.linspace(7., 8.8, 7)):
        settings = dict(PILOT, a_bohr=float(value))
        record, density = calculate_reference(settings, initial_density=density)
        report['points'].append(record)
        if index == 2:
            middle_density = density.copy()
        path.write_text(json.dumps(report, indent=2)+'\n')
        print('LiH independent pilot', value, record['energy_hartree'],
              'sampled gap', record['sampled_indirect_gap_hartree'],
              'seconds', record['elapsed_seconds'], flush=True)
    middle = report['points'][2]
    cold, cold_density = calculate_reference(middle['settings'])
    branch = dict(energy_difference=abs(cold['energy_hartree']-middle['energy_hartree']),
        relative_density_difference=float(np.linalg.norm(cold_density-middle_density)/(1+np.linalg.norm(cold_density))))
    assert branch['energy_difference'] < 1e-9 and branch['relative_density_difference'] < 1e-7
    hf, _ = calculate_reference(dict(PILOT, xc='hf', kmesh=[1, 1, 1], a_bohr=7.6))
    minimum = int(np.argmin([point['energy_hartree'] for point in report['points']]))
    gaps = [point['sampled_indirect_gap_hartree'] for point in report['points']]
    report.update(complete=True, bracketed_minimum=0 < minimum < len(report['points'])-1,
        sampled_minimum_index=minimum, all_sampled_gaps_positive=min(gaps)>1e-3,
        cold_warm_branch_check=branch, hf_comparison=hf,
        model='Rocksalt LiH, Cartesian GTH-SZV, GTH-Pade for both PBE and HF; T=0 integer occupations.',
        scope='Small candidate pilot, not basis/FFT/k-converged material properties.',
        pyscf_version=pyscf.__version__, hostname=platform.node(), python=platform.python_version(),
        elapsed_seconds=perf_counter()-started, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path.write_text(json.dumps(report, indent=2)+'\n')
    print('Pilot completed; interior minimum:', report['bracketed_minimum'],
          'positive sampled gaps:', report['all_sampled_gaps_positive'], flush=True)


if __name__ == '__main__':
    run_pilot()

# Recorded on jy_gpu_1, CPU float64 (2026-10-16); see P01_P02_lih_periodic.
# Recorded compare_case("pbe_szv_k2_fft31_repaired") passed:
# Maximum errors: E 2.13828e-8 Eh, F 2.66982e-9 Eh/bohr, stress 2.68817e-9 Eh/bohr^3.
# The original PBE displacement/strain FD and repaired-source HF full-Voigt FD
# are separately archived; the full original PBE FD was not repeated after repair.
