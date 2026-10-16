"""Plot completed silicon sensitivity and independent curvature checks.

Run examples 06, 07 and 08 in the same working directory first. All points
describe the fixed-geometry pilot, not a converged material prediction.
"""

from pathlib import Path
import json
import jax
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants, h, c
from gradscf.pbc import gto
from gradscf.pbc.ifc import ForceConstants

jax.config.update('jax_enable_x64', True)
output = Path('silicon_convergence')
with np.load('silicon_phonons/qmesh.npz') as archive:
    data = dict(archive)
cell = gto.M(
    atom=[('Si', r) for r in data['coordinates']],
    a=data['lattice'],
    unit='Bohr',
    basis='gth-szv',
    pseudo='gth-pbe',
    mesh=(31,) * 3,
)
model = ForceConstants.from_qmesh(
    cell, data['qpoints'], data['force_constants'], masses=data['masses']
)
conversion = physical_constants['Hartree energy'][0] / (h * c * 100)
frequencies = [np.asarray(model.bands(np.array([[0.0, 0.0, 0.5]])).frequencies[0]) * conversion]
for name in ('fft41', 'dzvp'):
    row = json.loads((output / (name + '.json')).read_text())
    frequencies.append(np.asarray(row['frequencies_cm1']))
checks = json.loads(Path('silicon_phonons/pyscf_reference.json').read_text())
lowest = sorted((r for r in checks if r['direction'] == 'lowest'), key=lambda r: r['step_bohr'])
assert len(lowest) == 2

fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
for x, values in enumerate(frequencies):
    axes[0].scatter(np.full(6, x), values, color='tab:blue', s=25)
axes[0].axhline(0.0, color='black', linewidth=0.7)
axes[0].set_xticks([0, 1, 2], ['SZV\nFFT 31³', 'SZV\nFFT 41³', 'DZVP\nFFT 31³'])
axes[0].set_ylabel(r'Signed L-point frequency (cm$^{-1}$)')
axes[0].set_title('Six modes; degenerate values overlap')
steps = np.array([r['step_bohr'] for r in lowest])
values = np.array([r['pyscf_curvature'] for r in lowest])
axes[1].plot(steps**2 * 1e6, values * 1e3, 'o-', label='PySCF energy differences')
axes[1].axhline(lowest[0]['native_curvature'] * 1e3, color='tab:orange', label='GradSCF DFPT')
axes[1].set_xlabel(r'Step squared ($10^{-6}$ Bohr$^2$)')
axes[1].set_ylabel(r'Curvature ($10^{-3}$ Eh/Bohr$^2$)')
axes[1].set_title('Independent negative-curvature check')
axes[1].legend(fontsize=8)
fig.suptitle('Si pilot: PBE/GTH-PBE, a = 5.43 Å, k = 2³; no finite-q projection')
fig.savefig(output / 'checks.png', dpi=180)
fig.savefig(output / 'checks.pdf')
