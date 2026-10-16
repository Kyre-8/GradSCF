"""Independent PySCF energy minimization and scalar EOS checks for P02."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize_scalar

from examples.pbc.compare_lih_forces_stress import OUTPUT, calculate_reference


def main():
    started = perf_counter()
    native_path = OUTPUT/'lattice_optimization.json'
    native = json.loads(native_path.read_text())
    assert native['complete'] and native['passed']
    native_hash = hashlib.sha256(native_path.read_bytes()).hexdigest()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    helper_hash = hashlib.sha256(Path(__file__).with_name('compare_lih_forces_stress.py').read_bytes()).hexdigest()
    destination = OUTPUT/'pyscf_lattice_optimization.json'
    minimum = native['minimum']
    state_path = OUTPUT/minimum['state_file']
    assert hashlib.sha256(state_path.read_bytes()).hexdigest() == minimum['state_sha256']
    fractional = np.asarray(json.loads(state_path.read_text())['fractional_kpoints'])
    report = dict(complete=False, native_sha256=native_hash, script_sha256=script_hash,
                  helper_script_sha256=helper_hash, settings=native['settings'], points=[])
    if destination.exists():
        previous = json.loads(destination.read_text())
        assert previous['native_sha256'] == native_hash and previous['script_sha256'] == script_hash
        assert previous['helper_script_sha256'] == helper_hash
        if previous.get('complete') and previous.get('passed'):
            print('Existing independent cubic minimum:', previous['minimum_a_bohr'], 'bohr')
            return
        report['points'] = previous['points']

    def save():
        destination.write_text(json.dumps(report, indent=2)+'\n')

    def point(a):
        for row in report['points']:
            if row['settings']['a_bohr'] == float(a):
                return row
        # Every energy is a fresh-cell, independently reconverged PySCF solve.
        row, _ = calculate_reference(dict(native['settings'], a_bohr=float(a)),
                                      fractional_kpoints=fractional)
        assert row['sampled_indirect_gap_hartree'] > 1e-3
        report['points'].append(row); save()
        print('independent cubic LiH', a, row['energy_hartree'], flush=True)
        return row

    def energy(a):
        return point(a)['energy_hartree']

    save()
    lower, middle, upper = native['bracket_bohr']
    assert energy(middle) < min(energy(lower), energy(upper))
    optimized = minimize_scalar(energy, bounds=(lower, upper), method='bounded',
                                options=dict(xatol=1e-5, maxiter=18))
    assert optimized.success
    native_a = minimum['a_bohr']
    common_energy = energy(native_a)
    slope_checks = []
    for step in [1e-3, 3e-4]:
        finite = (energy(native_a+step)-energy(native_a-step))/(2*step)
        slope_checks.append(dict(step_bohr=step, derivative_hartree_per_bohr=finite,
            native_error=abs(finite-minimum['derivative_hartree_per_bohr'])))
    step = native['curvature_step_bohr']
    curvature = (energy(native_a+step)-2*common_energy+energy(native_a-step))/step**2
    errors = dict(lattice_constant_bohr=abs(float(optimized.x)-native_a),
        common_geometry_energy_hartree=abs(common_energy-minimum['energy_hartree']),
        curvature_hartree_per_bohr2=abs(curvature-native['curvature_from_energy_hartree_per_bohr2']),
        derivative_hartree_per_bohr=slope_checks[-1]['native_error'])
    gates = dict(lattice_constant_bohr=2e-3, common_geometry_energy_hartree=2e-6,
                 curvature_hartree_per_bohr2=2e-5, derivative_hartree_per_bohr=3e-6)
    passed = all(value < gates[key] for key, value in errors.items())
    report.update(complete=True, passed=passed, minimum_a_bohr=float(optimized.x),
        minimum_energy_hartree=float(optimized.fun), optimizer_iterations=int(optimized.nfev),
        optimizer='Independent SciPy bounded minimization of cold-reconverged PySCF energies',
        energy_at_native_minimum_hartree=common_energy,
        curvature_at_native_minimum_hartree_per_bohr2=curvature,
        slope_checks=slope_checks, errors=errors, gates=gates,
        scope='Same finite Gaussian/GTH-Pade/FFT/k model and fixed fractional coordinates/kpoints; no experimental lattice prediction.',
        elapsed_seconds_this_invocation=perf_counter()-started)
    save()
    assert passed, errors
    print('independent lattice/EOS PASS', errors, flush=True)


if __name__ == '__main__':
    main()

# Recorded on jy_gpu_1, CPU float64 (2026-10-16); see P01_P02_lih_periodic.
# Independent lattice/EOS PASS: a = 7.780045423140039 bohr.
# Lattice difference 1.31464e-5 bohr; common-geometry energy error 5.37989e-8 Eh.
# Curvature error 1.12060e-6 Eh/bohr^2; largest slope error 2.61907e-7 Eh/bohr.
# Independent numerical minimum of the same declared finite model.
