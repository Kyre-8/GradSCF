"""Independent PySCF check of silicon raw Gamma and zone-boundary curvature.

Run 06_silicon_phonons.py first in the same working directory. This script uses
PySCF alone and reconverges every displaced state at matched basis, FFT and k
settings. The displacement coordinate and finite-difference steps are in Bohr;
curvatures are in Eh/Bohr**2 per primitive cell. The zone boundary uses a
1x1x2 supercell and alternating real displacements, including the lowest raw
curvature direction. No ASR projection is used. Displaced SCFs start from the
undisplaced density and are independently reconverged.
"""

import json
from pathlib import Path
import numpy as np
from pyscf.pbc import gto, dft

with np.load('silicon_phonons/qmesh.npz') as archive:
    data = dict(archive)
coordinates = data['coordinates']
lattice = data['lattice']
optical = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]) / np.sqrt(2.0)


def energy(displacement, repetitions, direction, initial=None):
    translations = np.arange(repetitions)[:, None] * lattice[2]
    positions = coordinates[None] + translations[:, None]
    signs = (-1.0) ** np.arange(repetitions)
    positions = positions + displacement * signs[:, None, None] * direction[None]
    cell = gto.M(
        atom=[('Si', r) for r in positions.reshape(-1, 3)],
        a=np.diag([1, 1, repetitions]) @ lattice,
        unit='Bohr',
        basis='gth-szv',
        pseudo='gth-pbe',
        mesh=(31, 31, 31 * repetitions),
        precision=1e-10,
        cart=True,
        verbose=0,
    )
    ref = dft.KRKS(cell, kpts=cell.make_kpts((2, 2, 2 // repetitions)))
    ref.xc = 'pbe'
    ref.conv_tol = 1e-12
    ref.conv_tol_grad = 1e-9
    ref.max_cycle = 150
    ref.kernel(dm0=initial)
    assert ref.converged
    return float(ref.e_tot) / repetitions, ref.make_rdm1()


rows = []
for q, repetitions in (([0.0, 0.0, 0.0], 1), ([0.0, 0.0, 0.5], 2)):
    index = np.flatnonzero(np.all(np.isclose(data['qpoints'], q), axis=1))[0]
    matrix = data['force_constants'][index]
    assert np.isfinite(matrix).all(), 'Finish the corresponding native q calculation first'
    initial = data['density'].sum(axis=0) if repetitions == 1 else None
    base, seed = energy(0.0, repetitions, optical, initial)
    np.testing.assert_allclose(base, data['energy'], atol=1e-7, rtol=0)
    directions = [('optical', optical)]
    if repetitions == 2:
        # This self-inverse q has a real IFC. Either vector in its lowest
        # degenerate pair is a valid direction for the independent energy test.
        np.testing.assert_allclose(matrix.imag, 0.0, atol=1e-10)
        lowest = np.linalg.eigh(matrix.real)[1][:, 0].reshape(2, 3)
        directions.append(('lowest', lowest))
    for label, direction in directions:
        expected = float(np.vdot(direction.ravel(), matrix @ direction.ravel()).real)
        for step in (0.002, 0.001):
            plus = energy(step, repetitions, direction, seed)[0]
            minus = energy(-step, repetitions, direction, seed)[0]
            value = (plus + minus - 2 * base) / step**2
            rows.append(
                dict(
                    q=q,
                    direction=label,
                    step_bohr=step,
                    native_curvature=expected,
                    pyscf_curvature=value,
                    error=value - expected,
                    base_energy_eh_per_cell=base,
                )
            )
            print(rows[-1], flush=True)
            np.testing.assert_allclose(value, expected, atol=2e-5, rtol=0)
            Path('silicon_phonons/pyscf_reference.json').write_text(json.dumps(rows, indent=2))

# Measured on jy_gpu_1, CPU, matched fixed-geometry PBE/GTH-SZV settings:
#                         native       PySCF h=.002   PySCF h=.001 (Eh/Bohr^2)
# Gamma optical direction 0.479329851   0.479339410    0.479332936
# q=(0,0,1/2) optical     0.315277406   0.315289377    0.315281410
# q=(0,0,1/2) lowest     -0.005430210  -0.005413797   -0.005424312
# Absolute errors at h=.001 Bohr: 3.0850e-6, 4.0039e-6, 5.8983e-6 Eh/Bohr^2.
# Undisplaced energies (Eh/primitive cell):
# Native -7.712547969705708; PySCF primitive -7.712547969705225;
# PySCF doubled cell / 2: -7.712547969705636.
# Agreement at these settings is an implementation check, not basis/k convergence.
# The independently negative curvature confirms this coarse calculation's
# negative mode; it does not establish a physical instability of bulk silicon.
