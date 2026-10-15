"""Relative momentum-resolved loss/gain maps from the completed q samples.

Run 05 first. Electronic modes stay at 300 K; only Bose temperature changes.
The displayed window contains modes above 100 meV. All signed low-frequency
modes, including instabilities, remain in the separate dispersion plot/data.
This positive-mode illustration is not a stable bulk thermodynamic prediction.
No momentum interpolation or detector-acceptance model is introduced.
"""
import json
import jax
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from gradscf.pbc.vibrational import mode_scattering_weights, one_phonon_spectrum
from settings import OUTPUT, MASSES, HARTREE_TO_MEV, make_cell, calculation_settings

jax.config.update('jax_enable_x64', True)
with np.load(OUTPUT/'dispersion_samples.npz') as archive:
    data = dict(archive)
expected = dict(calculation_settings(), k_mesh=[6, 6, 1], electron_temperature_K=300.)
assert json.loads(str(data['settings'])) == expected, 'The dispersion uses different settings.'
assert len(data['qpoints']) == 6, 'Complete all six direct q samples before plotting.'
assert np.max(data['relative_subspace_residuals']) < 1e-3

cell = make_cell()
reciprocal = np.asarray(cell.reciprocal_vectors())
qpoints = np.vstack((data['qpoints'], data['qpoints'][:1]))
energies = np.vstack((data['energies'], data['energies'][:1]))
modes = np.concatenate((data['modes'], data['modes'][:1]))
distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(qpoints @ reciprocal, axis=0), axis=1))]
grid_mev = np.linspace(-270., 270., 1801)
temperatures = (300., 1377.)
maps = []

for temperature in temperatures:
    intensities = []
    for q, energy, eigenvectors in zip(qpoints, energies, modes):
        # A fixed reciprocal offset G, plus the actual reduced q along the path.
        momentum = (np.array([1., 0., 0.])+q) @ reciprocal
        selected = energy*HARTREE_TO_MEV >= 100.
        weights = mode_scattering_weights(momentum, cell.coords, eigenvectors[:, selected],
            MASSES, form_factors=np.ones(4))/(2*energy[selected])
        spectrum = one_phonon_spectrum(energy[selected], weights, grid_mev/HARTREE_TO_MEV,
                                       temperature=temperature, fwhm=11./HARTREE_TO_MEV)
        intensities.append(np.asarray(spectrum.total)/HARTREE_TO_MEV)
    maps.append(np.asarray(intensities))

maximum = max(np.max(values) for values in maps)
normalization = LogNorm(vmin=maximum*1e-5, vmax=maximum)
fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True, sharey=True)
for row, (temperature, values) in enumerate(zip(temperatures, maps)):
    for column, (bounds, title) in enumerate((((-260., -100.), 'Energy gain'),
                                              ((100., 260.), 'Energy loss'))):
        selected = (grid_mev >= bounds[0]) & (grid_mev <= bounds[1])
        artist = axes[row, column].pcolormesh(grid_mev[selected], distance,
            np.ma.masked_less_equal(values[:, selected], 0.), shading='nearest',
            norm=normalization, cmap='magma')
        axes[row, column].set(title=f'{title}; phonon T = {temperature:.0f} K',
                              xlabel='Energy transfer (meV)')
        axes[row, column].set_yticks(distance[[0, 2, 3, 6]],
                                     [r'$\Gamma$', 'K', 'M', r'$\Gamma$'])
fig.colorbar(artist, ax=axes, label='Relative kinematic intensity / meV', shrink=.85)
fig.suptitle('Graphite pilot: direct q samples, fixed electronic T = 300 K\n'
             'Modes above 100 meV; Gaussian FWHM 11 meV; no thermal frequency renormalization')
fig.savefig(OUTPUT/'momentum_spectrum.png', dpi=180)
fig.savefig(OUTPUT/'momentum_spectrum.pdf')
np.savez(OUTPUT/'momentum_spectrum.npz', energy_meV=grid_mev, qpoints=qpoints,
         distance_bohr_inverse=distance, temperatures_K=temperatures, spectra_per_meV=maps,
         reciprocal_offset=[1., 0., 0.], form_factors=np.ones(4), minimum_mode_energy_meV=100.,
         fwhm_meV=11., settings=data['settings'])
print('Saved:', OUTPUT/'momentum_spectrum.png')
