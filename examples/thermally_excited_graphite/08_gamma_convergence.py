"""Refine Gamma optical modes before comparing a spectral peak to experiment.

Change one discretization at a time. Each case has its own output directory;
the coarse pilot is never overwritten. This is a convergence study, not a
prediction of the bulk anharmonic thermal redshift. Edit CASES to extend it.
"""
import json
import time

import jax
import numpy as np

from gradscf.pbc.phonon import Phonons
from settings import (OUTPUT, MASSES, HARTREE_TO_MEV, make_cell, mean_field,
                      inplane_directions, calculation_settings)

jax.config.update('jax_enable_x64', True)

# Start with the FFT error at the same SZV basis and electronic k mesh as 01.
# A converged electronic energy does not guarantee converged phonon curvature.
CASES = [
    ('szv_fft27_k3', 'gth-szv', (27, 27, 51), (3, 3, 1), (300., 1377.)),
    ('szv_fft21_k3', 'gth-szv', (21, 21, 41), (3, 3, 1), (300.,)),
    ('szv_fft33_k3', 'gth-szv', (33, 33, 63), (3, 3, 1), (300., 1377.)),
]

directions = inplane_directions()
for name, basis, fft_mesh, k_mesh, temperatures in CASES:
    output = OUTPUT/'gamma_precision'/name
    output.mkdir(parents=True, exist_ok=True)
    settings = calculation_settings()
    settings.update(basis=basis, fft_mesh=list(fft_mesh), k_mesh=list(k_mesh))
    cell = make_cell(basis=basis, mesh=fft_mesh)
    records = []

    for temperature in temperatures:
        start = time.perf_counter()
        print(name, 'electron temperature (K):', temperature, flush=True)
        mf = mean_field(cell, temperature, kmesh=k_mesh).run()
        assert mf.converged, 'SCF must converge before evaluating its response.'
        scf_seconds = time.perf_counter()-start
        result = Phonons(mf, masses=MASSES, max_grid=100_000,
                         max_pair_elements=2_000_000_000).project(directions)
        assert result.converged, 'The response solve did not converge.'
        all_energies = np.asarray(result.frequencies)
        optical = np.argsort(all_energies)[-4:]
        energies = all_energies[optical]
        leakage = float(result.subspace_residual_norm)/np.linalg.norm(all_energies**2)
        assert np.all(energies > 0), 'Inspect the signed optical curvatures.'
        assert leakage < 1e-3, 'The in-plane space is not invariant.'

        np.savez(output/f'gamma_{temperature:.0f}K.npz', energies=energies,
                 modes=np.asarray(result.modes)[:, optical],
                 all_inplane_energies=all_energies,
                 force_constants=np.asarray(result.force_constants),
                 coordinates=np.asarray(cell.coords), lattice=np.asarray(cell.lattice),
                 masses=MASSES, directions=directions, electron_temperature=temperature,
                 subspace_residual=float(result.subspace_residual_norm),
                 relative_subspace_residual=leakage,
                 settings=json.dumps(settings, sort_keys=True))
        row = dict(case=name, electron_temperature_K=temperature,
                   free_energy_Eh_cell=float(mf.e_free),
                   optical_energies_meV=(energies*HARTREE_TO_MEV).tolist(),
                   all_inplane_energies_meV=(all_energies*HARTREE_TO_MEV).tolist(),
                   response_residual=float(result.residual_norm),
                   relative_subspace_residual=leakage, scf_seconds=scf_seconds,
                   seconds_including_compilation=time.perf_counter()-start,
                   settings=settings,
                   scope='fixed lattice; adiabatic electronic temperature; no phonon self-energy')
        records.append(row)
        (output/'gamma_temperature.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    # Release per-case compiled objects before changing AO/grid dimensions.
    del mf, result, cell
    jax.clear_caches()
