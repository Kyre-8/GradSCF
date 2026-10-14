"""Independent PySCF check of silicon raw Gamma and zone-boundary curvature.

Run 06_silicon_phonons.py first in the same working directory. This script uses
PySCF alone and reconverges every displaced state at matched basis, FFT and k
settings. The displacement coordinate and finite-difference steps are in Bohr;
curvatures are in Eh/Bohr**2 per primitive cell. The zone boundary uses a
1x1x2 supercell and alternating real displacements. No ASR projection is used.
"""

import json
from pathlib import Path
import numpy as np
from pyscf.pbc import gto, dft

with np.load('silicon_phonons/qmesh.npz') as archive:
    data = dict(archive)
coordinates = data['coordinates']
lattice = data['lattice']
direction = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]) / np.sqrt(2.0)


def energy(displacement, repetitions):
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
    initial = data['density'].sum(axis=0) if repetitions == 1 else None
    ref.kernel(dm0=initial)
    assert ref.converged
    return float(ref.e_tot) / repetitions


rows = []
for q, repetitions in (([0.0, 0.0, 0.0], 1), ([0.0, 0.0, 0.5], 2)):
    index = np.flatnonzero(np.all(np.isclose(data['qpoints'], q), axis=1))[0]
    matrix = data['force_constants'][index]
    assert np.isfinite(matrix).all(), 'Finish the corresponding native q calculation first'
    expected = float(np.vdot(direction.ravel(), matrix @ direction.ravel()).real)
    base = energy(0.0, repetitions)
    np.testing.assert_allclose(base, data['energy'], atol=1e-7, rtol=0)
    for step in (0.002, 0.001):
        value = (energy(step, repetitions) + energy(-step, repetitions) - 2 * base) / step**2
        rows.append(
            dict(
                q=q,
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
# q=(0,0,1/2) direction   0.315277406   0.315289378    0.315281413
# Absolute errors at h=.001 Bohr: 3.0850e-6 and 4.0075e-6 Eh/Bohr^2.
# Undisplaced energies (Eh/primitive cell):
# Native -7.712547969705708; PySCF primitive -7.712547969705225;
# PySCF doubled cell / 2: -7.712547969705636.
# Agreement at these settings is an implementation check, not basis/k convergence.
