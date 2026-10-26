"""P02: optimize one cubic lattice parameter using the public AD stress.

This is a fixed-fractional-coordinate, zero-temperature LiH model. It is not a
full cell-shape/internal-coordinate relaxation. Each scalar evaluation rebuilds
the cell and reconverges SCF, so its Ewald topology is chosen at that geometry.
Run as an ordinary script with the repository root and src on PYTHONPATH.
"""
import hashlib
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import brentq

from examples.pbc.lih_forces_stress import OUTPUT, SETTINGS, run_case


SETTINGS = dict(SETTINGS, displacement_bohr=[[0., 0., 0.], [0., 0., 0.]])
BRACKET_BOHR = (7.6, 7.9, 8.2)


def _source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path('src').rglob('*.py')):
        digest.update(path.as_posix().encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def _point(name, settings, source_hash, helper_hash):
    """A fresh process bounds retained compilation state across cell rebuilds."""
    path = OUTPUT/(name+'.json')
    if path.exists():
        record = json.loads(path.read_text())
        if record.get('complete'):
            assert record['settings'] == settings
            assert record['source_python_sha256'] == source_hash
            assert record['script_sha256'] == helper_hash
            return record
    return run_case(name, settings)


def main():
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT/'lattice_optimization.json'
    source_hash = _source_hash()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    helper_hash = hashlib.sha256(Path(__file__).with_name('lih_forces_stress.py').read_bytes()).hexdigest()
    report = dict(complete=False, settings=SETTINGS, bracket_bohr=BRACKET_BOHR,
                  source_python_sha256=source_hash, script_sha256=script_hash,
                  helper_script_sha256=helper_hash, points=[])
    if destination.exists():
        previous = json.loads(destination.read_text())
        assert previous['source_python_sha256'] == source_hash
        assert previous['script_sha256'] == script_hash and previous['settings'] == SETTINGS
        assert previous['helper_script_sha256'] == helper_hash
        if previous.get('complete') and previous.get('passed'):
            print('Existing validated cubic minimum:', previous['minimum']['a_bohr'], 'bohr')
            return
        report['points'] = previous['points']

    def save():
        destination.write_text(json.dumps(report, indent=2)+'\n')

    def point(a):
        for row in report['points']:
            if row['a_bohr'] == float(a):
                return row
        name = f'cubic_eos_{len(report["points"]):03d}'
        settings = dict(SETTINGS, a_bohr=float(a))
        with multiprocessing.get_context('spawn').Pool(1, maxtasksperchild=1) as pool:
            state = pool.apply_async(_point, (name, settings, source_hash, helper_hash)).get(timeout=1200)
        path = OUTPUT/(name+'.json')
        stress = np.asarray(state['stress_hartree_per_bohr3'])
        # V(a)=a^3/4, dE/d(log a)=V Tr(sigma), tensile-positive sigma.
        slope = state['volume_bohr3']*np.trace(stress)/a
        row = dict(a_bohr=float(a), energy_hartree=state['energy_hartree'],
                   derivative_hartree_per_bohr=float(slope),
                   pressure_hartree_per_bohr3=state['pressure_hartree_per_bohr3'],
                   volume_bohr3=state['volume_bohr3'],
                   gap_hartree=state['sampled_indirect_gap_hartree'],
                   source_python_sha256=state['source_python_sha256'],
                   state_file=path.name, state_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report['points'].append(row); save()
        print('cubic LiH', a, row['energy_hartree'], 'dE/da', slope, flush=True)
        return row

    save()
    lower, middle, upper = [point(a) for a in BRACKET_BOHR]
    assert middle['energy_hartree'] < min(lower['energy_hartree'], upper['energy_hartree'])
    assert lower['derivative_hartree_per_bohr'] < 0 < upper['derivative_hartree_per_bohr']
    optimum, solver = brentq(lambda a: point(a)['derivative_hartree_per_bohr'],
                            BRACKET_BOHR[0], BRACKET_BOHR[-1], xtol=1e-5,
                            rtol=1e-12, maxiter=12, full_output=True)
    assert solver.converged
    minimum = point(optimum)
    step = .005
    minus, plus = point(optimum-step), point(optimum+step)
    curvature_energy = (plus['energy_hartree']-2*minimum['energy_hartree']+minus['energy_hartree'])/step**2
    curvature_stress = (plus['derivative_hartree_per_bohr']-minus['derivative_hartree_per_bohr'])/(2*step)
    assert curvature_energy > 1e-4
    np.testing.assert_allclose(curvature_energy, curvature_stress, atol=1e-5, rtol=0)
    assert abs(minimum['derivative_hartree_per_bohr']) < 2e-6
    assert min(plus['energy_hartree'], minus['energy_hartree']) > minimum['energy_hartree']
    bulk_modulus = (optimum**2*curvature_energy-2*optimum*minimum['derivative_hartree_per_bohr'])/(9*minimum['volume_bohr3'])
    report.update(complete=True, passed=True, minimum=minimum,
        optimizer='SciPy brentq of AD dE/da=V Tr(stress)/a after a measured energy/stress bracket',
        optimizer_iterations=solver.iterations, curvature_step_bohr=step,
        curvature_from_energy_hartree_per_bohr2=float(curvature_energy),
        curvature_from_stress_hartree_per_bohr2=float(curvature_stress),
        cubic_bulk_modulus_hartree_per_bohr3=float(bulk_modulus),
        scope='Cubic zero-pressure minimum of this finite-basis/FFT/k model only; no cell-shape or phonon-stability claim.',
        elapsed_seconds_this_invocation=perf_counter()-started)
    save()
    print('cubic lattice optimization PASS', optimum, 'bohr', flush=True)


if __name__ == '__main__':
    main()

# Recorded on jy_gpu_1, CPU float64 (2026-10-16); see P01_P02_lih_periodic.
# cubic lattice optimization PASS: a = 7.780032276700287 bohr.
# Nine native EOS points; symmetric fractional sites, PBE/GTH-Pade/SZV/k2^3/FFT31^3.
# A scalar cubic minimum does not certify shape/internal-coordinate/phonon stability.
