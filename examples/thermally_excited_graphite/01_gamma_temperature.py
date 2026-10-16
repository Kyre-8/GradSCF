"""Fixed-lattice Gamma optical frequencies versus ELECTRON temperature.

This isolates the electronic occupation contribution. It does not yet include
thermal expansion or the Brillouin-zone phonon-phonon self-energy. A separate
Bose temperature is used when synthesizing a spectrum in example 04.
"""
import json
import time
import jax
import numpy as np
from gradscf.pbc.phonon import Phonons
from settings import (OUTPUT, MASSES, HARTREE_TO_MEV, ELECTRON_TEMPERATURES,
                      make_cell, mean_field, inplane_directions, calculation_settings)

jax.config.update('jax_enable_x64', True)
OUTPUT.mkdir(exist_ok=True)
cell = make_cell()
directions = inplane_directions()
records = []

for temperature in ELECTRON_TEMPERATURES:
    start = time.perf_counter()
    mf = mean_field(cell, temperature).run()
    assert mf.converged
    driver = Phonons(mf, masses=MASSES, max_grid=100_000,
                     max_pair_elements=2_000_000_000)
    result = driver.project(directions)
    assert result.converged
    all_energies = np.asarray(result.frequencies)
    # Diagonalize the full in-plane space first, then retain its four highest
    # modes as the optical cluster. Do not assume a smaller optical subspace
    # is closed under the finite-grid dynamical matrix.
    optical = np.argsort(all_energies)[-4:]
    energies = all_energies[optical]
    # A negative mode is an instability, not a positive excitation to broaden.
    assert np.all(energies > 0), 'The optical reference is unstable; inspect raw curvatures.'
    relative_leakage = float(result.subspace_residual_norm)/np.linalg.norm(all_energies**2)
    print('Electron temperature (K):', temperature, flush=True)
    print('Projected optical energies (meV):', energies*HARTREE_TO_MEV, flush=True)
    print('Full-space relative residual:', relative_leakage, flush=True)
    assert relative_leakage < 1e-3, 'Optical subspace is not closed: refine the FFT grid or calculate full modes.'

    # Retain full modes, geometry and diagnostics for reproducible spectra.
    np.savez(OUTPUT/f'gamma_{temperature:.0f}K.npz', energies=energies,
             modes=np.asarray(result.modes)[:, optical], all_inplane_energies=all_energies,
             force_constants=np.asarray(result.force_constants),
             coordinates=np.asarray(cell.coords), lattice=np.asarray(cell.lattice),
             masses=MASSES, directions=directions, electron_temperature=temperature,
             subspace_residual=float(result.subspace_residual_norm),
             relative_subspace_residual=relative_leakage,
             settings=json.dumps(calculation_settings(), sort_keys=True))
    records.append({
        'electron_temperature_K': temperature, 'free_energy_Eh_cell': float(mf.e_free),
        'optical_energies_meV': (energies*HARTREE_TO_MEV).tolist(),
        'all_inplane_energies_meV': (all_energies*HARTREE_TO_MEV).tolist(),
        'response_residual': float(result.residual_norm),
        'relative_subspace_residual': relative_leakage,
        'seconds_including_compilation': time.perf_counter()-start,
        'scope': 'fixed geometry; electronic temperature only; projected harmonic modes',
        'settings': calculation_settings(),
    })
    (OUTPUT/'gamma_temperature.json').write_text(json.dumps(records, indent=2)+'\n')

# CPU float64, default pilot: four highest in-plane energies in meV.
#  300 K: [175.08639040, 175.45328125, 178.55994709, 178.92268846]
#  800 K: [176.22750911, 176.59097584, 178.76226937, 179.12551640]
# 1377 K: [178.06478917, 178.42422264, 180.41597195, 180.77618448]
# Full-space relative residuals: 3.67e-13, 9.23e-14, 1.02e-13.
# These electronic-temperature shifts are NOT the experimental thermal redshift.
