"""Independent PySCF reconverged Berry finite differences; no GradSCF import.

Full Bloch AOs are integrated with exp(-i*q.r), independently of GradSCF's
periodic-part/folding implementation. The exact saved basis, PP, mesh, ordered
k representatives and geometry are reused. Every displaced SCF is a cold solve.
"""
import hashlib
import inspect
import json
import platform
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf.pbc import dft, gto
from pyscf.pbc.dft.gen_grid import UniformGrids
from pyscf.pbc.dft.numint import eval_ao_kpts


OUTPUT = Path('reproducibility/ad_showcase/P04_lih_born')


def calculate(native, coordinates, *, anchor=None):
    settings = native['settings']
    lattice = np.asarray(native['lattice_bohr'])
    fractional = np.asarray(native['fractional_kpoints'])
    cell = gto.M(atom=list(zip(['Li', 'H'], coordinates)), a=lattice, unit='Bohr',
                 basis=native['basis_data'], pseudo=dict(zip(['Li', 'H'], native['pseudopotential_data'])),
                 mesh=settings['mesh'], precision=settings['precision'], spin=0, cart=True, verbose=0)
    kpoints = cell.get_abs_kpts(fractional)
    mf = dft.KRKS(cell, kpts=kpoints, exxdiv='ewald')
    mf.xc = settings['xc']
    mf.grids = UniformGrids(cell)
    mf.grids.mesh = np.asarray(settings['mesh'])
    mf.with_df.mesh = np.asarray(settings['mesh'])
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 150
    energy = mf.kernel()
    occupation, eigenvalues = np.asarray(mf.mo_occ), np.asarray(mf.mo_energy)
    nocc = cell.nelectron//2
    gap = float(np.min(eigenvalues[:, nocc:])-np.max(eigenvalues[:, :nocc]))
    residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, mf.get_fock())))
    if not mf.converged or gap <= 1e-3 or residual > 2e-8:
        raise RuntimeError(f'Independent SCF is not a resolved insulator: gap={gap}, residual={residual}.')
    np.testing.assert_array_equal(occupation, np.broadcast_to(np.r_[np.ones(nocc)*2, np.zeros(cell.nao_nr()-nocc)], occupation.shape))
    np.testing.assert_allclose(cell.atom_charges(), native['pseudopotential_charges'], atol=0, rtol=0)
    mesh = tuple(settings['mesh'])
    fractions = np.stack(np.meshgrid(*[np.arange(n)/n for n in mesh], indexing='ij'), axis=-1).reshape(-1, 3)
    grid = fractions@lattice
    psi = np.asarray(eval_ao_kpts(cell, grid, kpts=kpoints))
    coefficients = np.asarray(mf.mo_coeff)[:, :, :nocc]
    volume = float(cell.vol)
    wrapped = np.mod(np.round(np.mod(fractional, 1.), 10), 1.)
    axes = [np.unique(wrapped[:, i], return_inverse=True) for i in range(3)]
    shape = tuple(len(item[0]) for item in axes)
    index_grid = np.empty(shape, dtype=int)
    index_grid[tuple(item[1] for item in axes)] = np.arange(len(kpoints))
    reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    phases, minima, changes = [], [], []
    for axis, count in enumerate(shape):
        q = np.eye(3)[axis]/count
        distances = fractional[None]-fractional[:, None]-q
        distances -= np.rint(distances)
        matches = np.max(abs(distances), axis=-1) < 1e-9
        assert np.all(matches.sum(axis=1) == 1)
        partners = matches.argmax(axis=1)
        phase_operator = np.exp(-1j*grid@(q@reciprocal))
        cross_ao = np.einsum('kgp,g,kgq->kpq', psi.conj(), phase_operator, psi[partners])*volume/len(grid)
        overlap = coefficients.conj().swapaxes(-1, -2)@cross_ao@coefficients[partners]
        singular = np.linalg.svd(overlap, compute_uv=False)
        minima.append(float(singular.min()))
        if singular.min() <= 1e-8:
            raise RuntimeError('An independent occupied Berry link is singular.')
        signs, _ = np.linalg.slogdet(overlap)
        strings = np.moveaxis(index_grid, axis, -1).reshape(-1, count)
        phase = np.angle(np.prod(signs[strings], axis=1))
        if anchor is not None:
            change = np.angle(np.exp(1j*(phase-np.asarray(anchor[axis]))))
            if np.max(abs(change)) >= np.pi/4:
                raise RuntimeError('Displacement is too large to certify this local phase branch.')
            changes.append(float(np.max(abs(change))))
            phase = np.asarray(anchor[axis])+change
        phases.append(phase)
    # Restricted PySCF orbitals represent two spins; ionic charges include the
    # actual GTH q3/q1 electron counts, not chemical oxidation-state labels.
    electronic = 2*np.array([p.mean() for p in phases])@lattice/(2*np.pi*volume)
    ionic = np.asarray(native['pseudopotential_charges'])@coordinates/volume
    return dict(polarization_e_per_bohr2=(electronic+ionic).tolist(), phases=[p.tolist() for p in phases],
                energy_hartree=float(energy), sampled_gap_hartree=gap, orbital_gradient_norm=residual,
                minimum_link_singular_value=min(minima), maximum_phase_change=max(changes, default=0.))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _signature(input_hash, coordinates, anchor):
    return dict(input_sha256=input_hash,
        calculate_sha256=hashlib.sha256(inspect.getsource(calculate).encode()).hexdigest(),
        coordinates_bohr=np.asarray(coordinates).tolist(), anchor=anchor,
        python=platform.python_version(), numpy=np.__version__, pyscf=pyscf.__version__)


def _atomic_json(path, values):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _write_endpoint(path, signature, record, seconds, origin):
    values = dict(complete=True, signature=signature, signature_sha256=_digest(signature),
                  record=record, record_sha256=_digest(record), seconds=seconds, origin=origin)
    _atomic_json(path, values)
    return values


def main():
    started = perf_counter()
    source = OUTPUT/'gradscf.json'
    native = json.loads(source.read_text())
    assert native['complete'] and native['full_tensor'] and native['validation_passed']
    input_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    directory = OUTPUT/'pyscf_endpoints'/input_hash[:16]
    directory.mkdir(parents=True, exist_ok=True)
    endpoints = {}

    def endpoint(key, positions, anchor=None):
        path = directory/(key+'.json')
        signature = _signature(input_hash, positions, anchor)
        if path.exists():
            saved = json.loads(path.read_text())
            if not saved['complete'] or saved['signature'] != signature or saved['signature_sha256'] != _digest(signature) or saved['record_sha256'] != _digest(saved['record']):
                raise RuntimeError(f'Endpoint input/function/data hash mismatch: {path}')
            print('Checkpoint reused', key, flush=True)
        else:
            before = perf_counter()
            value = calculate(native, positions, anchor=anchor)
            saved = _write_endpoint(path, signature, value, perf_counter()-before,
                                    dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
            print('Endpoint completed', key, 'seconds', saved['seconds'], flush=True)
        endpoints[key] = saved
        _atomic_json(OUTPUT/'pyscf_progress.json', dict(complete=False, input_sha256=input_hash,
            completed_endpoint_files=len(list(directory.glob('*.json'))), endpoint_files=sorted(p.name for p in directory.glob('*.json')),
            last_endpoint=key, current_process_seconds=perf_counter()-started))
        return saved['record']

    coordinates = np.asarray(native['coordinates_bohr'])
    base = endpoint('reference', coordinates)
    rows = []
    for atom in range(2):
        for axis in range(3):
            direction = np.zeros_like(coordinates)
            direction[atom, axis] = 1.
            for step in (3e-4, 1e-4):
                key = f'atom{atom}_axis{axis}_step{step:.8g}'
                plus = endpoint(key+'_plus', coordinates+step*direction, anchor=base['phases'])
                minus = endpoint(key+'_minus', coordinates-step*direction, anchor=base['phases'])
                derivative = native['volume_bohr3']*(np.asarray(plus['polarization_e_per_bohr2'])-np.asarray(minus['polarization_e_per_bohr2']))/(2*step)
                expected = np.asarray(native['born_charge_e'])[atom, :, axis]
                row = dict(atom=atom, axis=axis, step_bohr=step, charge_e=derivative.tolist(),
                           maximum_charge_error_e=float(np.max(abs(derivative-expected))), plus=plus, minus=minus)
                rows.append(row)
                print('PySCF Born', atom, axis, step, row['charge_e'], 'error', row['maximum_charge_error_e'], flush=True)
                (OUTPUT/'pyscf_partial.json').write_text(json.dumps(dict(complete=False, reference=base, finite_differences=rows), indent=2)+'\n')
    passed = all(row['maximum_charge_error_e'] < 5e-5 for row in rows)
    report = dict(complete=True, validation_passed=passed, reference=base, finite_differences=rows,
        energy_error_hartree=abs(base['energy_hartree']-native['energy_hartree']),
        maximum_charge_error_e=max(row['maximum_charge_error_e'] for row in rows),
        born_gate_e=5e-5, elapsed_seconds_this_process=perf_counter()-started,
        checkpoint_endpoint_seconds=sum(value['seconds'] for value in endpoints.values() if value['seconds'] is not None),
        legacy_endpoints_without_individual_timing=sum(value['seconds'] is None for value in endpoints.values()),
        completed_endpoint_count=len(endpoints),
        timing_scope='Process time excludes previous batches. Endpoint timing excludes legacy records without individual timings; preserve batch resource logs.',
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, hostname=platform.node(),
        python=platform.python_version(), pyscf=pyscf.__version__, numpy=np.__version__,
        input_sha256=input_hash, calculate_sha256=_signature(input_hash, coordinates, None)['calculate_sha256'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Independent PySCF SCF and full-Bloch-AO Berry quadrature at matched finite model; all displacements cold-reconverged. No external dielectric/field reference.')
    (OUTPUT/'pyscf.json').write_text(json.dumps(report, indent=2)+'\n')
    _atomic_json(OUTPUT/'pyscf_progress.json', dict(complete=True, input_sha256=input_hash,
        completed_endpoint_files=len(endpoints), validation_passed=passed,
        current_process_seconds=perf_counter()-started))
    assert passed, 'Independent Born derivatives exceed the declared 5e-5 e gate.'


if __name__ == '__main__':
    main()
