"""r2SCAN optical Gamma modes at the same settings as the refined PBE baseline.

Only the XC functional changes. The GTH-PBE pseudopotential and fixed lattice
are retained as declared approximations, not called an r2SCAN-consistent
pseudopotential calculation. Basis/k convergence and bulk anharmonicity remain
separate tasks. Run 09 afterwards; it reports resolved multipeak spectra
separately instead of forcing a single-peak experimental comparison.
"""
import json
import time
import jax
import numpy as np
from gradscf.pbc.phonon import Phonons
from settings import (OUTPUT, MASSES, HARTREE_TO_MEV, make_cell, mean_field,
                      inplane_directions, calculation_settings)

jax.config.update('jax_enable_x64', True)
CASE = ('r2scan_szv_fft33_k3', (33, 33, 63), (300., 1377.))
# The separately recorded finer-grid diagnostic:
# CASE = ('r2scan_szv_fft39_k3', (39, 39, 75), (300.,))
name, mesh, temperatures = CASE
output = OUTPUT/'gamma_precision'/name
output.mkdir(parents=True, exist_ok=True)
cell = make_cell(mesh=mesh)
settings = dict(calculation_settings(), xc='r2scan', fft_mesh=list(mesh))
directions = inplane_directions()
records = []

for temperature in temperatures:
    start = time.perf_counter()
    print('r2SCAN electron temperature (K):', temperature, flush=True)
    mf = mean_field(cell, temperature, xc='r2scan').run()
    assert mf.converged, 'Do not compute response from an unconverged reference.'
    scf_seconds = time.perf_counter()-start
    print('SCF converged; free energy / seconds:', float(mf.e_free), scf_seconds, flush=True)
    result = Phonons(mf, masses=MASSES, max_grid=150_000,
                     max_pair_elements=3_000_000_000).project(directions)
    assert result.converged
    all_energies = np.asarray(result.frequencies)
    optical = np.argsort(all_energies)[-4:]
    energies = all_energies[optical]
    leakage = float(result.subspace_residual_norm)/np.linalg.norm(all_energies**2)
    assert np.all(energies > 0) and leakage < 1e-3
    np.savez(output/f'gamma_{temperature:.0f}K.npz', energies=energies,
             modes=np.asarray(result.modes)[:, optical],
             all_inplane_energies=all_energies,
             force_constants=np.asarray(result.force_constants),
             coordinates=np.asarray(cell.coords), lattice=np.asarray(cell.lattice),
             masses=MASSES, directions=directions, electron_temperature=temperature,
             relative_subspace_residual=leakage,
             settings=json.dumps(settings, sort_keys=True))
    row = dict(electron_temperature_K=temperature, free_energy_Eh_cell=float(mf.e_free),
               optical_energies_meV=(energies*HARTREE_TO_MEV).tolist(),
               all_inplane_energies_meV=(all_energies*HARTREE_TO_MEV).tolist(),
               response_residual=float(result.residual_norm),
               relative_subspace_residual=leakage, scf_seconds=scf_seconds,
               seconds_including_compilation=time.perf_counter()-start, settings=settings)
    records.append(row)
    (output/'gamma_temperature.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(row), flush=True)
