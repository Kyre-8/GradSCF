"""Converge the fixed-lattice graphite reference before calculating phonons.

Run from the repository root with PYTHONPATH=src. No command-line interface.
All energies are per four-carbon primitive cell, in Hartree.
"""
import json
import time
import jax
import numpy as np
from settings import OUTPUT, BASIS, FFT_MESH, K_MESH, make_cell, mean_field, calculation_settings

jax.config.update('jax_enable_x64', True)
OUTPUT.mkdir(exist_ok=True)
cell = make_cell()
start = time.perf_counter()
mf = mean_field(cell, 300.).run()
assert mf.converged, 'Converge the electronic reference before computing derivatives.'
elapsed = time.perf_counter()-start

print('Basis / FFT / k mesh:', BASIS, FFT_MESH, K_MESH)
print('SCF converged / iterations:', mf.converged, mf.result.cycles)
print('Internal energy (Eh/cell):', float(mf.e_tot))
print('Free energy (Eh/cell):', float(mf.e_free))
occupations = np.asarray(mf.result.mo_occ_spin[0])
print('Fractionally occupied spin orbitals:', np.count_nonzero((occupations > 1e-8) & (occupations < 1-1e-8)))
print('SCF time including compilation (s):', elapsed)

np.savez(OUTPUT/'reference.npz', lattice=np.asarray(cell.lattice),
         coordinates=np.asarray(cell.coords), density=np.asarray(mf.result.density_spin),
         kpoints=np.asarray(mf.kpts), energies=np.asarray(mf.result.mo_energy_spin),
         occupations=np.asarray(mf.result.mo_occ_spin), settings=json.dumps(calculation_settings(), sort_keys=True))
(OUTPUT/'reference.json').write_text(json.dumps({
    'settings': calculation_settings(), 'electron_temperature_K': 300., 'geometry_relaxed': False,
    'total_energy_Eh_cell': float(mf.e_tot), 'free_energy_Eh_cell': float(mf.e_free),
    'seconds': elapsed, 'jax_version': jax.__version__, 'backend': jax.default_backend(),
    'scope': 'fixed-geometry pilot; basis/FFT/k convergence is not established',
}, indent=2)+'\n')

# Measured on jy_gpu_1, CPU JAX 0.10.0, float64, default pilot settings:
# Internal energy (Eh/cell): -22.472858173664125
# Free energy (Eh/cell): -22.473441428470178
# One cold SCF run took 23.33 s; this is not an isolated performance benchmark.
