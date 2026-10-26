"""Cu(111) photoemission broadening with literature inputs and experiment.

Native GradSCF Fan linewidth + scalar spectral function. alpha2F and measured
points are vector-digitized from Eiguren et al., PRL 88, 066805 (2002).
This is a literature-input benchmark, not a new slab GW/DFPT calculation.
No widths/couplings are fitted to experiment. Run with JAX_PLATFORMS=cpu.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from gradscf.gw.ep_coupling import fan_linewidth, spectral_function
from gradscf.tools.spectra import HARTREE_TO_EV

jax.config.update('jax_enable_x64', True)
folder = Path(__file__).resolve().parents[2] / 'reproducibility/gw_bse/cu111_photoemission'
source = json.loads((folder/'sources.json').read_text())
a2f = np.loadtxt(folder/'alpha2f.csv', delimiter=',', skiprows=1)
experiment = np.loadtxt(folder/'experiment_linewidth.csv', delimiter=',', skiprows=1)
edc = np.loadtxt(folder/'experiment_edc.csv', delimiter=',', skiprows=1)
kb_ev = 8.617333262145e-5
gamma_ee = source['electronic_baseline_mev']


def quadrature(nmode=480, nelectron=801):
    """Represent the local flat-continuum alpha2F integral by Fan vertices.

    g_lm^2 = alpha2F(Omega_l) dOmega_l dE_m. No extra DOS/spin factor:
    the published alpha2F already represents this particular hole state.
    The 1 meV Gaussian is numerical delta integration, not an extra PES width.
    """
    dw = .030 / nmode
    omega = (np.arange(nmode) + .5) * dw
    alpha = np.interp(omega * 1000, a2f[:, 0], a2f[:, 1], left=0., right=0.)
    internal = np.linspace(-.52, -.36, nelectron)
    weights = np.full(nelectron, internal[1]-internal[0])
    weights[[0, -1]] *= .5
    g = np.sqrt(alpha[:, None] * dw * weights[None, :])[:, None, :] / HARTREE_TO_EV
    return tuple(map(jnp.asarray, (omega/HARTREE_TO_EV, internal/HARTREE_TO_EV, g)))


omega, internal, couplings = quadrature()


@jax.jit
def phonon_width(temperature, omega=omega, internal=internal, couplings=couplings):
    beta = HARTREE_TO_EV / (kb_ev * jnp.maximum(temperature, 1e-6))
    return fan_linewidth(jnp.array([-.44/HARTREE_TO_EV]), internal, omega, couplings,
        mu=0., beta=beta, eta=.001/HARTREE_TO_EV, smearing='gaussian')[0] * HARTREE_TO_EV * 1000


temperatures = np.linspace(0., 320., 65)
widths = np.array([float(phonon_width(t)) for t in temperatures])
mode_ev = np.asarray(omega) * HARTREE_TO_EV
alpha = np.interp(mode_ev*1000, a2f[:, 0], a2f[:, 1], left=0., right=0.)
continuum = []
for temperature in temperatures:
    occupation = np.zeros_like(mode_ev) if temperature == 0 else 1/np.expm1(mode_ev/(kb_ev*temperature))
    continuum.append(2*np.pi*np.sum(alpha*(1+2*occupation))*.030/len(mode_ev)*1000)
continuum_error = float(np.max(np.abs(widths-continuum)))
assert continuum_error < .001  # meV: independent deep-hole continuum formula.
predicted = gamma_ee + np.array([float(phonon_width(t)) for t in experiment[:, 0]])
fine = quadrature(960, 1601)
quadrature_error = max(abs(float(phonon_width(t, *fine)-phonon_width(t))) for t in (0., 55., 160., 285.))
assert quadrature_error < .025  # meV, phonon-grid discretization check.

# The intrinsic Gamma points in the paper have already been extracted by
# line-shape fits. Instrument convolution is applied only to plotted spectra.
offset = np.linspace(-160., 160., 6401)  # meV relative to each peak center.


def profile(width):
    sigma = jnp.full((len(offset), 1, 1), -.5j * width / (1000 * HARTREE_TO_EV))
    a = np.asarray(spectral_function(jnp.zeros((1, 1)), sigma,
        jnp.asarray(offset)/(1000*HARTREE_TO_EV), eta=1e-12))[:, 0, 0].real / (1000*HARTREE_TO_EV)
    instrument_sigma = source['instrument_energy_resolution_mev'] / np.sqrt(8*np.log(2))
    return gaussian_filter1d(a, instrument_sigma/(offset[1]-offset[0]), mode='constant')


plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False,
                     'axes.spines.right':False, 'axes.titleweight':'bold', 'savefig.dpi':180})
colors = {55:'#2166ac', 160:'#b87916', 285:'#b44242'}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), constrained_layout=True)
ax = axes[0]
ax.plot(offset, profile(gamma_ee), '--', color='#61656b', lw=1.8,
        label=f'Electronic baseline: {gamma_ee:.0f} meV')
for t, color in colors.items():
    total = gamma_ee + float(phonon_width(t))
    ax.plot(offset, profile(total), color=color, lw=2,
            label=f'+ phonons, {t} K: {total:.1f} meV')
ax.set(xlim=(-80, 80), ylim=(0, None), xlabel='Energy relative to peak center (meV)',
       ylabel='Spectral density (meV$^{-1}$)', title='A  Phonons broaden the photoemission peak')
ax.legend(fontsize=9.5, frameon=False, loc='upper right')
ax.text(.02, .97, '3 meV instrument FWHM\napplied to every curve', va='top',
        transform=ax.transAxes, fontsize=9, color='#555555')
ax = axes[1]
ax.axhline(gamma_ee, color='#61656b', ls='--', lw=1.6, label='Electronic baseline (literature)')
ax.plot(temperatures, gamma_ee+widths, color='#2166ac', lw=2, label='GradSCF Fan + electronic baseline')
ax.scatter(experiment[:, 0], experiment[:, 1], facecolors='white', edgecolors='#b44242', s=55, lw=1.5,
           zorder=5, label='Experiment: intrinsic FWHM (Fig. 3)')
known = np.isfinite(experiment[:, 2])
ax.errorbar(experiment[known, 0], experiment[known, 1], yerr=experiment[known, 2],
            fmt='none', color='#b44242', capsize=4, zorder=4)
ax.set(xlim=(0, 320), ylim=(10, 46), xlabel='Temperature (K)', ylabel='Intrinsic FWHM (meV)',
       title='B  Comparison with measured linewidths')
ax.legend(fontsize=9.5, frameon=False, loc='upper left')
ax.text(.97, .04, 'Only the error bar reported in Fig. 3 is shown', ha='right',
        transform=ax.transAxes, fontsize=9, color='#555555')
fig.suptitle('Cu(111) Shockley surface state: electron–phonon broadening', fontsize=15)
fig.savefig(folder/'cu111_broadening.png')
fig.savefig(folder/'cu111_broadening.pdf')
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), sharex=True, sharey=True, constrained_layout=True)
for ax, (temperature, color) in zip(axes, colors.items()):
    points = edc[edc[:, 0] == temperature]
    center = points[points[:, 2].argmax(), 1]
    measured = points[:, 2] / points[:, 2].max()
    total = gamma_ee + float(phonon_width(temperature))
    line, bare = profile(total), profile(gamma_ee)
    ax.plot(offset, bare/bare.max(), color='#777777', ls='--', lw=1.2, label='Without phonons')
    ax.plot(offset, line/line.max(), color=color, lw=2, label='With phonons')
    ax.scatter(points[:, 1]-center, measured, facecolors='none', edgecolors='black', s=22,
               lw=.8, label='Experiment')
    ax.set(title=f'{temperature} K', xlim=(-60, 60), ylim=(0, 1.12), xlabel='Binding energy relative to peak (meV)')
    ax.text(.96, .96, f'$\\Gamma$ = {total:.1f} meV', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, color=color)
axes[0].set_ylabel('Peak-normalized intensity')
axes[0].legend(frameon=False, fontsize=8.5, loc='upper left', bbox_to_anchor=(0, 1.03))
fig.suptitle('Experimental line shapes: each trace centered on its own maximum\nNo coupling or linewidth fit; thermal peak shifts are not modeled', fontsize=13)
fig.savefig(folder/'cu111_experimental_profiles.png')
fig.savefig(folder/'cu111_experimental_profiles.pdf')
plt.close(fig)

np.savetxt(folder/'linewidth_comparison.csv', np.column_stack((experiment[:, 0], experiment[:, 1], predicted,
           predicted-experiment[:, 1])), delimiter=',', comments='',
           header='temperature_k,experiment_fwhm_mev,calculated_fwhm_mev,residual_mev')
np.savetxt(folder/'calculated_temperature_curve.csv', np.column_stack((temperatures, widths, gamma_ee+widths)),
           delimiter=',', comments='', header='temperature_k,phonon_fwhm_mev,total_fwhm_mev')
summary = dict(backend=jax.default_backend(), jax_version=jax.__version__, dtype=str(couplings.dtype),
    zero_temperature_ep_mev=float(phonon_width(0.)),
    extracted_lambda=float(2*np.trapezoid(a2f[:, 1]/a2f[:, 0], a2f[:, 0])),
    quadrature_max_difference_mev=quadrature_error,
    independent_continuum_error_mev=continuum_error,
    experiment_rmse_mev=float(np.sqrt(np.mean((predicted-experiment[:, 1])**2))),
    widths_mev={str(t):float(gamma_ee+phonon_width(t)) for t in colors},
    input_scope='Published alpha2F and electronic baseline; no new slab GW/DFPT or experimental width fit.',
    approximation='Local flat electronic continuum, Fan on-shell FWHM, Lorentzian quasiparticle line plus 3 meV Gaussian.',
    experimental_data='Vector-digitized published markers, not original instrument files.')
(folder/'results.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))

# CPU float64 results (2026-10-04):
# Phonon contribution at T -> 0: 6.88655 meV; graph-derived lambda: 0.162906.
# Total FWHM at 55,160,285 K: 22.02591,29.41693,39.88765 meV.
# Seven-point experimental FWHM RMSE: 2.40329 meV; no linewidth fit.
# Doubling both quadratures changes widths by at most 0.000694 meV.
