"""Separate occupation, electronic-frequency and instrumental effects.

Run 01 and 02 first. The spectrum is a relative single-scattering illustration,
not an absolute EELS cross section. No empirical thermal frequency shift or
temperature-dependent lifetime is inserted to match the experimental markers.
"""
import json
import jax
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gradscf.pbc.vibrational import one_phonon_spectrum, mode_scattering_weights
from settings import OUTPUT, HARTREE_TO_MEV, calculation_settings

jax.config.update('jax_enable_x64', True)
records = json.loads((OUTPUT/'gamma_temperature.json').read_text())
grid_mev = np.linspace(-270., 270., 1801)
fig, panels = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
axes = panels.ravel()
with np.load(OUTPUT/'gamma_300K.npz') as archive:
    cold = dict(archive)

# Full Q at a reciprocal lattice vector, not q=0, gives a nonzero probe vertex.
# Equal unit form factors are a stated model choice, not screened atomic data.
for record in records:
    temperature = record['electron_temperature_K']
    with np.load(OUTPUT/f'gamma_{temperature:.0f}K.npz') as archive:
        data = dict(archive)
    assert json.loads(str(data['settings'])) == calculation_settings()
    assert float(data['relative_subspace_residual']) < 1e-3
    reciprocal = 2*np.pi*np.linalg.inv(data['lattice']).T
    weights = mode_scattering_weights(reciprocal[0], data['coordinates'],
        data['modes'], data['masses'], form_factors=np.ones(len(data['masses'])))
    strengths = weights/(2*data['energies'])
    spectrum = one_phonon_spectrum(data['energies'], strengths, grid_mev/HARTREE_TO_MEV,
                                   temperature=temperature, fwhm=11./HARTREE_TO_MEV)
    cold_weights = mode_scattering_weights(reciprocal[0], cold['coordinates'],
        cold['modes'], cold['masses'], form_factors=np.ones(len(cold['masses'])))
    occupation_only = one_phonon_spectrum(cold['energies'], cold_weights/(2*cold['energies']),
        grid_mev/HARTREE_TO_MEV, temperature=temperature, fwhm=11./HARTREE_TO_MEV)
    axes[0].scatter(np.full(4, temperature), data['energies']*HARTREE_TO_MEV,
                    s=22, label=f'Electronic T = {temperature:.0f} K')
    axes[1].plot(grid_mev, np.asarray(spectrum.total)/HARTREE_TO_MEV,
                 label=f'Te = Tph = {temperature:.0f} K')
    axes[2].plot(grid_mev, np.asarray(occupation_only.total)/HARTREE_TO_MEV,
                 label=f'Te = 300 K; Tph = {temperature:.0f} K')
    np.savez(OUTPUT/f'spectrum_{temperature:.0f}K.npz', energy_meV=grid_mev,
             loss_per_meV=np.asarray(spectrum.loss)/HARTREE_TO_MEV,
             gain_per_meV=np.asarray(spectrum.gain)/HARTREE_TO_MEV,
             occupation_only_per_meV=np.asarray(occupation_only.total)/HARTREE_TO_MEV,
             momentum_bohr_inverse=reciprocal[0], form_factors=np.ones(len(data['masses'])),
             fwhm_meV=11., electron_temperature_K=temperature, phonon_temperature_K=temperature,
             settings=data['settings'])

# Extracted from arXiv:2610.18561v1, Sec. III.A (not digitized from a plot).
# The unresolved experimental peak is not assigned to an individual Ritz mode;
# the experimental momentum acceptance and instrument geometry are not modeled.
axes[0].errorbar([300., 1377.], [193.7, 185.2], yerr=[.2, .3],
                 fmt='kx', capsize=3, label='Experiment: unresolved optical peak')
axes[0].set(xlabel='Temperature (K)', ylabel='Energy (meV)',
            title='Fixed-lattice electronic contribution')
axes[0].legend(fontsize=7)
axes[1].set(xlabel='Energy transfer (meV)', ylabel='Relative intensity / meV',
            title='Loss (+) and gain (-); Gaussian FWHM 11 meV')
axes[1].legend(fontsize=7)
axes[2].set(xlabel='Energy transfer (meV)', ylabel='Relative intensity / meV',
            title='Bose occupation only; fixed 300 K modes')
axes[2].legend(fontsize=7)

with np.load(OUTPUT/'frozen_mode.npz') as archive:
    scan = dict(archive)
assert str(scan['settings']) == str(cold['settings']), 'Do not combine different electronic settings.'
points = np.linspace(scan['displacements'].min(), scan['displacements'].max(), 201)
baseline = scan['energies'][np.argmin(abs(scan['displacements']))]
axes[3].scatter(scan['displacements'], (scan['energies']-baseline)*HARTREE_TO_MEV,
                s=18, label='Relaxed SCF')
for order in (4, 6):
    fitted = np.polynomial.polynomial.polyval(points, scan[f'coefficients{order}'])
    axes[3].plot(points, (fitted-baseline)*HARTREE_TO_MEV, label=f'Order {order}')
axes[3].set(xlabel='Mode displacement (Bohr)', ylabel='Free-energy change (meV/cell)',
            title='Local potential; no bulk thermal correction')
axes[3].legend(fontsize=8)
fig.suptitle('AB graphite pilot: fixed geometry, unconverged basis/k/FFT\n'
             'Electronic-temperature effects and local potential diagnostics')
fig.savefig(OUTPUT/'graphite_pilot.png', dpi=180)
fig.savefig(OUTPUT/'graphite_pilot.pdf')
print('Saved:', OUTPUT/'graphite_pilot.png')
