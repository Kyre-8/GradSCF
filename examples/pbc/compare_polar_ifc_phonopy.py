"""Independent pinned Phonopy reciprocal kernel check; no GradSCF import.

Use the isolated Phonopy2.44/Spglib2.5/Symfc1.5 oracle environment recorded in
the results. Its private C-kernel API is pinned explicitly for this comparison.
"""
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import phonopy
import spglib
import symfc
from phonopy.harmonic.dynamical_matrix import DynamicalMatrixGL
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_primitive, get_supercell


OUTPUT = Path('reproducibility/ad_showcase/P05_polar_interpolation')


def run():
    started = perf_counter()
    report_path = OUTPUT/'gradscf.json'
    native = json.loads(report_path.read_text())
    assert native['complete'] and native['passed']
    if (phonopy.__version__, spglib.__version__, symfc.__version__) != ('2.44.0', '2.5.0', '1.5.0'):
        raise RuntimeError('Use the recorded pinned oracle versions, not a changed private API.')
    lattice, positions = np.asarray(native['lattice_bohr']), np.asarray(native['positions_bohr'])
    born, epsilon = np.asarray(native['born_charges']), np.asarray(native['epsilon_infinity'])
    atoms = PhonopyAtoms(symbols=['Li', 'H'], cell=lattice, positions=positions, masses=[7., 1.])
    supercell = get_supercell(atoms, np.eye(3, dtype=int))
    primitive = get_primitive(supercell, np.eye(3))
    np.testing.assert_allclose(primitive.cell, lattice, atol=1e-12, rtol=0)
    np.testing.assert_allclose(primitive.positions, positions, atol=1e-12, rtol=0)
    # Numbers are bohr/Eh throughout. factor=1 removes the default eV/Angstrom
    # conversion. Phonopy uses reciprocal cycles, so Lambda carries 1/(2*pi).
    params = dict(born=born, dielectric=epsilon, factor=1., G_cutoff=2.,
                  Lambda=native['smearing_inverse_bohr']/(2*np.pi))
    reference = DynamicalMatrixGL(supercell, primitive, np.zeros((2, 2, 3, 3)),
                                 nac_params=params, use_openmp=False)
    integers = np.array(list(itertools.product(range(-native['cutoff'], native['cutoff']+1), repeat=3)))
    reciprocal = np.linalg.inv(lattice).T
    reference._G_list = np.ascontiguousarray(integers@reciprocal, dtype=float)
    # Compare the raw reciprocal model; the onsite convention cancels when
    # consistently subtracted and restored in total IFC interpolation.
    reference._dd_q0 = np.zeros((2, 3, 3), dtype=complex)

    def matrix(q, direction=None):
        cartesian = np.ascontiguousarray(q@reciprocal, dtype=float)
        direction = None if direction is None else np.ascontiguousarray(direction, dtype=float)
        atomic_gauge = reference._get_c_recip_dipole_dipole(cartesian, direction).reshape(6, 6)
        phase = np.repeat(np.exp(2j*np.pi*(positions@cartesian)), 3)
        return phase[:, None]*atomic_gauge*phase.conj()[None, :]

    with np.load(OUTPUT/'native.npz') as saved:
        qpoints, target = saved['qpoints'], saved['interpolation_target']
        values = np.array([matrix(q) for q in qpoints])
        gamma = matrix(np.zeros(3))
        directional = matrix(np.zeros(3), saved['gamma_direction'])
        short = (np.asarray(native['spring'])-np.asarray(native['onsite'])
                 +native['short_range_cosine_coefficient']*np.sum(1-np.cos(2*np.pi*target))*np.eye(6))
        errors = dict(non_gamma=float(np.max(abs(values-saved['matrices']))),
            gamma=float(np.max(abs(gamma-saved['gamma']))),
            directional_gamma=float(np.max(abs(directional-saved['directional_gamma']))),
            total_interpolation=float(np.max(abs(short+matrix(target)-saved['interpolated_matrix']))))
        steps = []
        for step in (1e-3, 3e-4, 1e-4):
            delta = step*saved['derivative_direction']
            finite = (matrix(qpoints[0]+delta)-matrix(qpoints[0]-delta))/(2*step)
            steps.append(dict(step=step, derivative_error=float(np.max(abs(finite-saved['derivative'])))))
    gates = dict(matrix=1e-11, derivative=2e-8)
    passed = max(errors.values()) < gates['matrix'] and steps[-1]['derivative_error'] < gates['derivative']
    report = dict(complete=True, passed=passed, errors=errors, finite_differences=steps, gates=gates,
        native_json_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        native_arrays_sha256=hashlib.sha256((OUTPUT/'native.npz').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions=dict(phonopy=phonopy.__version__, spglib=spglib.__version__, symfc=symfc.__version__),
        elapsed_seconds=perf_counter()-started,
        convention='Phonopy reciprocal cycles/Lambda converted to angular units; raw unweighted C kernel; atomic to lattice phase U=exp(i q.tau).',
        scope='Independent implementation of a fixed reciprocal dipole model and manufactured short-range part; not independent electronic structure.')
    (OUTPUT/'phonopy.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Phonopy comparison:',passed,errors,steps,flush=True)
    assert passed, report
    return report


if __name__ == '__main__':
    run()

# Measured on jy_gpu_1 with the pinned oracle environment (2026-10-16):
# Maximum native/Phonopy matrix error: 5.55e-17 Eh/bohr^2.
# Directional FD errors at h=1e-3 / 3e-4 / 1e-4:
# 2.17464e-8 / 1.95717e-9 / 2.17449e-10 Eh/bohr^2.
# All declared gates passed; wrapper wall 1.50 s, RSS 69,244 KiB.
