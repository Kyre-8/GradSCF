"""Separate FFT and basis effects on the silicon L-point curvature.

Run example 06 first in the same working directory. Each case changes one
setting from its fixed-geometry PBE/GTH-SZV, k=2^3, FFT=31^3 baseline. These
are sensitivity checks, not a complete convergence study or relaxed phonons.
The k=4^3 check remains outside this default example: the observed
memory limit must be resolved before treating it as a routine material case.
Run material checks on a suitably sized machine; the output records cold
times including compilation. No ASR projection is used at this finite q.
"""

from pathlib import Path
import json
import time
import jax
import numpy as np
from scipy.constants import physical_constants, h, c
from gradscf.pbc import gto, dft
from gradscf.pbc.phonon import Phonons

jax.config.update('jax_enable_x64', True)
output = Path('silicon_convergence')
output.mkdir(exist_ok=True)
with np.load('silicon_phonons/qmesh.npz') as archive:
    baseline = dict(archive)
q = np.array([0.0, 0.0, 0.5])
index = np.flatnonzero(np.all(np.isclose(baseline['qpoints'], q), axis=1))[0]
reference = baseline['force_constants'][index]
conversion = physical_constants['Hartree energy'][0] / (h * c * 100)

# Change one setting at a time; all retain the same commensurate L point.
cases = [('fft41', 41, 2, 'gth-szv'), ('dzvp', 31, 2, 'gth-dzvp')]
for name, grid, nk, basis in cases:
    start = time.perf_counter()
    print('Starting', name, 'FFT', grid, 'k', nk, 'basis', basis, flush=True)
    cell = gto.M(
        atom=[('Si', r) for r in baseline['coordinates']],
        a=baseline['lattice'],
        unit='Bohr',
        basis=basis,
        pseudo='gth-pbe',
        mesh=(grid,) * 3,
        precision=1e-10,
    )
    mf = dft.KRKS(cell, kpts=cell.make_kpts((nk,) * 3), xc='pbe').run(
        conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-9, max_cycle=150, damping=0.1
    )
    assert mf.converged
    scf_seconds = time.perf_counter() - start
    print(name, 'SCF finished in', scf_seconds, 's; starting force constants', flush=True)
    # This explicit conservative workspace ceiling accommodates the selected
    # small-cell material cases; it does not preallocate AO-pair tensors.
    phonons = Phonons(
        mf, masses=baseline['masses'], max_grid=100_000, max_pair_elements=2_000_000_000
    )
    result = phonons.kernel(q, method='primitive')
    frequencies = np.asarray(result.frequencies) * conversion
    row = dict(
        case=name,
        basis=basis,
        mesh=list(cell.mesh),
        kmesh=[nk] * 3,
        nao=cell.topology.nao,
        energy_eh=float(mf.e_tot),
        max_force_constant_change=float(np.max(np.abs(result.force_constants - reference))),
        frequencies_cm1=frequencies.tolist(),
        response_residual=float(result.residual_norm),
        hermiticity_error=float(result.hermiticity_error),
        max_reference_force=float(np.max(np.abs(phonons.reference_forces))),
        scf_seconds=scf_seconds,
        total_seconds=time.perf_counter() - start,
        backend=jax.default_backend(),
        jax_version=jax.__version__,
    )
    np.savez(
        output / (name + '.npz'),
        force_constants=np.asarray(result.force_constants),
        frequencies_cm1=frequencies,
        q=q,
    )
    (output / (name + '.json')).write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row), flush=True)
    jax.clear_caches()

# Measured on jy_gpu_1, CPU float64, JAX 0.10.0, a=5.43 Angstrom, k=2^3:
# SZV / FFT31 baseline L modes (cm^-1):
# [-71.47785, -71.47785, 437.14172, 444.84998, 591.10305, 591.10305]
# SZV / FFT41:
# [-71.54189, -71.54189, 437.13139, 444.84071, 591.09707, 591.09707]
# DZVP / FFT31 (28 Cartesian AOs):
# [-84.84185, -84.84185, 378.65517, 417.87819, 492.79469, 492.79469]
# DZVP energy: -7.7688173116059405 Eh/cell
# DZVP response residual: 2.8950e-10; Hermiticity error: 5.4466e-10 Eh/Bohr^2
# Neither change removes the negative pair. k/geometry convergence is pending.
# The separate k=4^3 run exceeded the memory guard and produced no IFC result.
