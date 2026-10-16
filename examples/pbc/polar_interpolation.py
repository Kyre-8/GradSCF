"""Polar IFC subtraction/interpolation/restoration on a manufactured crystal.

This isolates phase, cutoff and interpolation conventions before a material
calculation. The supplied charges/epsilon are model inputs, not SCF results.
"""
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)
from gradscf.pbc.polar_ifc import DipoleCorrection, PolarForceConstants


OUTPUT = Path('reproducibility/ad_showcase/P05_polar_interpolation')
LATTICE = np.diag([5., 6., 7.])
POSITIONS = np.array([[.2, .3, .4], [.7, 1., 1.2]])
CHARGE = np.array([[1.1, .2, -.1], [.3, 1.4, .2], [.1, -.2, .9]])
BORN = np.stack((CHARGE, -CHARGE))
EPSILON = np.array([[2.4, .1, .2], [.1, 3.1, -.1], [.2, -.1, 2.8]])
TARGET = np.array([.12, -.09, .17])
SPRING = .2*np.block([[np.eye(3), -np.eye(3)], [-np.eye(3), np.eye(3)]])
ONSITE = np.diag([.1, .2, .3, .3, .4, .5])


def short_range(q):
    return SPRING-ONSITE+.03*np.sum(1-np.cos(2*np.pi*q))*np.eye(6)


def run():
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cell = SimpleNamespace(lattice=LATTICE, coords=POSITIONS)
    model = DipoleCorrection(LATTICE, POSITIONS, BORN, EPSILON, smearing=.35, cutoff=5)
    qpoints = np.array([[.13, -.17, .21], [.21, .08, -.16], [0., 0., .5], [1.13, -1.17, .21]])
    matrices = np.array([model.matrix(q) for q in qpoints])
    direction = np.array([.2, -.1, .3])
    derivative = np.asarray(jax.jit(lambda q:jax.jvp(model.matrix, (q,),
        (jnp.asarray(direction),))[1])(jnp.asarray(qpoints[0])))
    gamma_direction = np.array([1., .2, .3])
    gamma, directional = np.asarray(model.gamma()), np.asarray(model.gamma(gamma_direction))

    fine = DipoleCorrection(LATTICE, POSITIONS, BORN, EPSILON, smearing=.35, cutoff=6)
    reference = np.asarray(fine.matrix(TARGET))
    cutoff_records = []
    for cutoff in range(1, 6):
        current = DipoleCorrection(LATTICE, POSITIONS, BORN, EPSILON, smearing=.35, cutoff=cutoff)
        value = np.asarray(current.matrix(TARGET))
        cutoff_records.append(dict(cutoff=cutoff, error=float(np.max(abs(value-reference))),
            periodic_error=float(np.max(abs(current.matrix(TARGET+[1., -1., 0.])-value))),
            self_inverse_imaginary=float(np.max(abs(current.matrix([0., 0., .5]).imag)))))
    assert max(cutoff_records[-1][key] for key in ['error', 'periodic_error', 'self_inverse_imaginary']) < 1e-11

    expected = short_range(TARGET)+model.matrix(TARGET)
    alternate = DipoleCorrection(LATTICE, POSITIONS, BORN, EPSILON, smearing=.45, cutoff=5)
    grid_records = []
    for size in (3, 5, 7):
        axis = np.arange(-(size//2), size//2+1)/size
        grid = np.array(list(itertools.product(axis, repeat=3)))
        long_range = np.array([model.gamma() if np.all(q == 0) else model.matrix(q) for q in grid])
        sampled = np.array([short_range(q) for q in grid])+long_range
        same = PolarForceConstants.from_qmesh(cell, grid, sampled, masses=[7., 1.], correction=model, tolerance=1e-9)
        varied = PolarForceConstants.from_qmesh(cell, grid, sampled, masses=[7., 1.], correction=alternate, tolerance=1e-9)
        row = dict(qmesh=size, same_model_error=float(np.max(abs(same.matrix(TARGET)-expected))),
            changed_smearing_error=float(np.max(abs(varied.matrix(TARGET)-expected))),
            sample_reconstruction_error=varied.sample_reconstruction_error,
            total_raw_asr=float(varied.raw_asr_residual), short_range_raw_asr=float(varied.short_range.raw_asr_residual))
        grid_records.append(row)
        print('Interpolation:', row, flush=True)
        assert row['same_model_error'] < 1e-11 and row['sample_reconstruction_error'] < 1e-11
    assert grid_records[-1]['changed_smearing_error'] < 2e-6
    assert grid_records[-1]['changed_smearing_error'] < grid_records[0]['changed_smearing_error']/10

    np.savez_compressed(OUTPUT/'native.npz', qpoints=qpoints, matrices=matrices,
        derivative=derivative, derivative_direction=direction, gamma=gamma,
        gamma_direction=gamma_direction, directional_gamma=directional,
        interpolation_target=TARGET, interpolated_matrix=np.asarray(same.matrix(TARGET)))
    report = dict(complete=True, passed=True, lattice_bohr=LATTICE.tolist(), positions_bohr=POSITIONS.tolist(),
        born_charges=BORN.tolist(), epsilon_infinity=EPSILON.tolist(), smearing_inverse_bohr=.35, cutoff=5,
        spring=SPRING.tolist(), onsite=ONSITE.tolist(), short_range_cosine_coefficient=.03,
        cutoff_records=cutoff_records, qmesh_records=grid_records, elapsed_seconds=perf_counter()-started,
        source_sha256=hashlib.sha256(Path('src/gradscf/pbc/polar_ifc.py').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Manufactured unweighted force constants, not a material. No ASR projection; smearing and reciprocal/q-mesh convergence are distinct.')
    (OUTPUT/'gradscf.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Polar model gates PASS',report['elapsed_seconds'],'s',flush=True)
    return report


if __name__ == '__main__':
    run()

# Measured on jy_gpu_1, CPU float64 (2026-10-16):
# Native model gates passed in 24.14 s; process peak RSS 1,638,228 KiB.
# Changed-width interpolation errors, q grids 3^3 / 5^3 / 7^3:
# 2.05725e-4 / 5.57214e-7 / 5.04893e-11 Eh/bohr^2.
# Manufactured IFCs; raw translational defect retained, no material claim.
