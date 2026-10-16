"""Silicon PBE/GTH-SZV phonons and short-range Fourier interpolation.

Fixed a=5.43 Angstrom, k/q=2x2x2, FFT=31^3, natural-isotope masses.
This is a pilot at a specified geometry, not a basis/k/q-converged material
prediction. Raw force constants and ASR diagnostics are saved before the
optional Gamma projection. Negative frequencies are kept visible.
"""

from pathlib import Path
import itertools
import json
import time
import jax
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants, h, c
from gradscf.pbc import gto, dft
from gradscf.pbc.phonon import Phonons
from gradscf.pbc.ifc import ForceConstants

jax.config.update('jax_enable_x64', True)
output = Path('silicon_phonons')
output.mkdir(exist_ok=True)
a = 5.43
lattice = a / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
cell = gto.M(
    atom=[('Si', [0.0, 0.0, 0.0]), ('Si', [a / 4, a / 4, a / 4])],
    a=lattice,
    unit='Angstrom',
    basis='gth-szv',
    pseudo='gth-pbe',
    mesh=(31,) * 3,
    precision=1e-10,
)
mf = dft.KRKS(cell, kpts=cell.make_kpts((2, 2, 2)), xc='pbe').run(
    conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-9, max_cycle=150, damping=0.1
)
assert mf.converged
print('PBE energy (Eh/cell):', float(mf.e_tot), flush=True)

masses = np.array([28.0855, 28.0855])
phonons = Phonons(mf, masses=masses, max_grid=100_000, max_pair_elements=200_000_000)
qpoints = np.array(list(itertools.product([0.0, 0.5], repeat=3)))
matrices = np.full((len(qpoints), 6, 6), np.nan + 0j)
seconds = np.full(len(qpoints), np.nan)
residuals = np.full(len(qpoints), np.nan)
for index, q in enumerate(qpoints):
    start = time.perf_counter()
    result = phonons.kernel(q, method='primitive')
    matrices[index] = np.asarray(result.force_constants)
    seconds[index] = time.perf_counter() - start
    residuals[index] = float(result.residual_norm)
    checkpoint = output / 'qmesh.tmp.npz'
    np.savez(
        checkpoint,
        qpoints=qpoints,
        force_constants=matrices,
        masses=masses,
        lattice=np.asarray(cell.lattice),
        coordinates=np.asarray(cell.coords),
        kpoints=np.asarray(mf.kpts),
        density=np.asarray(mf.result.density_spin),
        energy=float(mf.e_tot),
        seconds=seconds,
        residuals=residuals,
    )
    checkpoint.replace(output / 'qmesh.npz')
    print('q:', q, 'seconds:', seconds[index], 'residual:', residuals[index], flush=True)

# For small calculations, phonons.mesh((2,2,2),asr=True) performs this sampling
# directly. The explicit loop above retains a checkpoint after every q point.
model = ForceConstants.from_qmesh(cell, qpoints, matrices, masses=masses, asr=True)
print('Raw Gamma ASR residual:', float(model.raw_asr_residual), flush=True)
print('Gamma projection norm:', float(model.asr_correction_norm), flush=True)
(output / 'diagnostics.json').write_text(
    json.dumps(
        {
            'raw_asr_residual_eh_bohr2': float(model.raw_asr_residual),
            'asr_correction_norm_eh_bohr2': float(model.asr_correction_norm),
            'max_response_residual': float(np.max(residuals)),
            'qpoint_seconds': seconds.tolist(),
            'basis': 'gth-szv',
            'pseudo': 'gth-pbe',
            'xc': 'pbe',
            'lattice_parameter_angstrom': a,
            'fft_mesh': list(cell.mesh),
            'kmesh': [2, 2, 2],
            'qmesh': [2, 2, 2],
            'jax_version': jax.__version__,
            'backend': jax.default_backend(),
        },
        indent=2,
    )
    + '\n'
)

vertices = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])
path = np.concatenate(
    [np.linspace(left, right, 31)[:-1] for left, right in zip(vertices[:-1], vertices[1:])]
    + [vertices[-1:]]
)
cartesian = path @ np.asarray(cell.reciprocal_vectors())
distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(cartesian, axis=0), axis=1))]
bands = model.bands(path)
conversion = physical_constants['Hartree energy'][0] / (h * c * 100)
frequencies = np.asarray(bands.frequencies) * conversion
dos_points = np.array(list(itertools.product(np.arange(10) / 10, repeat=3)))
energy_grid = np.linspace(-300.0, 900.0, 1201) / conversion
dos = np.asarray(model.dos(dos_points, energy_grid, width=8.0 / conversion)) / conversion

fig, axes = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
axes[0].plot(distance, frequencies, color='tab:blue', linewidth=1.1)
axes[0].axhline(0, color='black', linewidth=0.6)
axes[0].set_xticks(distance[[0, 30, 60, 90]], [r'$\Gamma$', 'X', 'L', r'$\Gamma$'])
axes[0].set_ylabel(r'Signed frequency (cm$^{-1}$)')
axes[1].plot(dos, energy_grid * conversion, color='tab:orange')
axes[1].set_xlabel(r'DOS (states/cm$^{-1}$/cell)')
fig.suptitle('Si PBE/GTH-SZV pilot: k/q=2x2x2, FFT=31³; Gamma ASR projected')
fig.tight_layout()
fig.savefig(output / 'dispersion.png', dpi=180)
fig.savefig(output / 'dispersion.pdf')
np.savez(
    output / 'dispersion.npz',
    qpoints=path,
    distance=distance,
    frequencies_cm1=frequencies,
    dos_cm1=dos,
    energy_cm1=energy_grid * conversion,
)

# Measured on jy_gpu_1, Intel Xeon Platinum 8457C, CPU float64, JAX 0.10.0:
# PBE energy: -7.7125479697057076 Eh/cell
# Eight q points: 847.39 s in total, including per-q compilation (not warm calls).
# Maximum response residual: 4.0763e-12
# Raw Gamma ASR residual: 5.2561e-5 Eh/Bohr^2
# Gamma ASR projection norm: 4.4521e-5 Eh/Bohr^2
# Gamma modes after projection: three numerical zeros, 671.5532 x 3 cm^-1
# L modes: -71.4779 x 2, 437.1417, 444.8500, 591.1031 x 2 cm^-1
# Minimum along this path: -72.2133 cm^-1; DOS integral: 6.000000000000001
# The negative branches belong to this coarse fixed-geometry calculation.
# They require basis/k/q/geometry convergence checks before a physical claim.
