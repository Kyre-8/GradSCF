"""Compare graphite free energies across FFT meshes at fixed 300 K electrons.

Both programs independently converge the same geometry, basis, pseudopotential
and absolute k vectors. GradSCF uses Fourier-truncated AO grids and overlap /
kinetic integrals; PySCF uses real-space AO grids and analytic overlap / kinetic
integrals. Matching FFT dimensions therefore does not imply identical numerical
discretization. This checks energies, not phonon or anharmonic convergence.
"""
import json
import time
import jax
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyscf
from pyscf.pbc import gto, dft, df, scf
from pyscf.pbc.dft.gen_grid import UniformGrids
from settings import OUTPUT, BASIS, KB_HARTREE, make_cell, mean_field, calculation_settings

jax.config.update('jax_enable_x64', True)
OUTPUT.mkdir(exist_ok=True)
results = []
for mesh in ((15, 15, 31), (21, 21, 41), (27, 27, 51)):
    start = time.perf_counter()
    cell = make_cell(mesh=mesh)
    native = mean_field(cell, 300.).run()
    native_seconds = time.perf_counter()-start

    start = time.perf_counter()
    reference_cell = gto.Cell()
    reference_cell.atom = [('C', r) for r in np.asarray(cell.coords)]
    reference_cell.a = np.asarray(cell.lattice)
    reference_cell.unit = 'Bohr'
    reference_cell.basis = BASIS
    reference_cell.pseudo = 'gth-pbe'
    reference_cell.cart = True
    reference_cell.mesh = mesh
    reference_cell.precision = 1e-10
    reference_cell.verbose = 0
    reference_cell.build()
    kpoints = np.asarray(native.kpts)
    reference = dft.KRKS(reference_cell, kpoints, xc='pbe', exxdiv=None)
    reference.with_df = df.FFTDF(reference_cell, kpoints)
    reference.grids = UniformGrids(reference_cell)
    reference.grids.mesh = mesh
    reference = scf.addons.smearing_(reference, sigma=300.*KB_HARTREE, method='fermi')
    reference.conv_tol = 1e-12
    reference.max_cycle = 180
    reference.kernel()
    initial_converged = bool(reference.converged)
    # A bounded retry uses only PySCF's own density, never the native solution.
    if not initial_converged:
        reference.kernel(dm0=reference.make_rdm1())
    reference_seconds = time.perf_counter()-start

    settings = calculation_settings()
    settings['fft_mesh'] = list(mesh)
    both_converged = bool(native.converged and reference.converged)
    result = dict(
        settings=settings, electron_temperature_K=300.,
        kpoints_bohr_inverse=kpoints.tolist(),
        pyscf_controls=dict(conv_tol=1e-12, max_cycle=180, maximum_attempts=2),
        native_converged=bool(native.converged), pyscf_converged=bool(reference.converged),
        pyscf_initial_converged=initial_converged, pyscf_self_restart=not initial_converged,
        native_energy_Eh_cell=float(native.e_tot), native_free_energy_Eh_cell=float(native.e_free),
        pyscf_energy_Eh_cell=float(reference.e_tot), pyscf_free_energy_Eh_cell=float(reference.e_free),
        energy_difference_Eh_cell=float(native.e_tot-reference.e_tot) if both_converged else None,
        free_energy_difference_Eh_cell=float(native.e_free-reference.e_free) if both_converged else None,
        native_seconds=native_seconds, pyscf_seconds=reference_seconds,
    )
    results.append(result)
    print('FFT / native converged / PySCF converged / delta F (Eh/cell):',
          mesh, result['native_converged'], result['pyscf_converged'],
          result['free_energy_difference_Eh_cell'], flush=True)
    (OUTPUT/'grid_convergence.json').write_text(json.dumps(dict(
        calculations=results, jax_version=jax.__version__, pyscf_version=pyscf.__version__,
        backend=jax.default_backend(),
        scope='fixed-geometry electronic free energies only; no phonon convergence claim',
    ), indent=2)+'\n')

if not all(row['native_converged'] and row['pyscf_converged'] for row in results):
    raise RuntimeError('Some references did not converge; inspect grid_convergence.json before comparison.')

fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
labels = ['x'.join(map(str, row['settings']['fft_mesh'])) for row in results]
for key, label in (('native_free_energy_Eh_cell', 'GradSCF'),
                   ('pyscf_free_energy_Eh_cell', 'PySCF')):
    axes[0].plot(labels, [row[key] for row in results], 'o-', label=label)
axes[0].set(ylabel='Free energy (Eh/cell)', title='Independent electronic references')
axes[0].legend()
axes[1].semilogy(labels, [abs(row['free_energy_difference_Eh_cell']) for row in results], 'o-')
axes[1].set(ylabel='Absolute free-energy difference (Eh/cell)', title='Finite AO/FFT discretization error')
for ax in axes:
    ax.set_xlabel('FFT mesh')
    ax.tick_params(axis='x', labelrotation=15)
fig.suptitle('AB graphite: same geometry, k vectors and 300 K electrons; phonons not converged by this check')
fig.savefig(OUTPUT/'grid_convergence.png', dpi=180)
fig.savefig(OUTPUT/'grid_convergence.pdf')

# Measured CPU float64, JAX 0.10.0 / PySCF 2.13.0; both references converged:
# FFT (15,15,31): delta F = 4.959816929770e-3 Eh/cell
# FFT (21,21,41): delta F = 1.380159797648e-4 Eh/cell
# FFT (27,27,51): delta F = 7.829949275617e-7 Eh/cell
# The last PySCF reference needed its bounded self-restart. No native seed was used.
