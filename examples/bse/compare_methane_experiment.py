"""Compare the archived methane BSE spectra with gas-phase absorption data.

No electronic-structure calculation, network request, CLI, energy shift or
intensity fit. Download the uncommitted experimental tables and verify their
checksums using experiment/README.md beside the archived spectrum.json.
The calculated HWHM stays at 0.15 eV. Cross sections are functions
of energy: converting wavelength to energy does not introduce a Jacobian.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import h, c, e
from scipy.integrate import trapezoid

folder = Path(__file__).resolve().parents[2] / 'reproducibility/gw_bse/methane_aug_cc_pvdz'
report = json.loads((folder / 'spectrum.json').read_text())
sources = json.loads((folder / 'experiment/sources.json').read_text())
experiments = []
for source in sources['datasets']:
    path = folder / 'experiment' / source['file']
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing experimental data: {path}. Download the exact sources.json '
            f'URLs and verify SHA256 checksums as described in {folder / "experiment/README.md"}.'
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
    data = np.loadtxt(path)
    assert data.ndim == 2 and data.shape[1] == 2
    assert np.all(np.isfinite(data)) and np.all(data > 0)
    energy = h * c / e * 1e9 / data[:, 0]
    cross_section = data[:, 1] * 1e18  # cm^2 -> Mb
    order = np.argsort(energy)
    energy, cross_section = energy[order], cross_section[order]
    assert np.all(np.diff(energy) > 0)
    experiments.append((energy, cross_section))

methods = ('g0w0', 'evgw0', 'evgw')
labels = (r'$G_0W_0$ + BSE', r'ev$GW_0$ + BSE', r'ev$GW$ + BSE')
colors = ('#0072B2', '#D55E00', '#009E73')
styles = ('-', '--', '-.')
energy_calc = np.asarray(report['energy_ev'])
energy_exp, sigma_exp = experiments[0]  # Kameta 2002 is the main reference.
common = np.linspace(10., 23.8, 2761)
assert energy_exp[0] <= common[0] < common[-1] <= energy_exp[-1]
reference = np.interp(common, energy_exp, sigma_exp)
metrics = {'comparison_interval_ev': [10., 23.8],
           'theory_hwhm_ev': report['spectral_hwhm_ev'], 'energy_shift_ev': 0.,
           'intensity_scale': 1., 'reference': sources['datasets'][0], 'methods': {}}
peak = int(np.argmax(sigma_exp))
metrics['experimental_maximum'] = {'energy_ev': float(energy_exp[peak]), 'cross_section_mb': float(sigma_exp[peak])}
point_energies = [10.2, 13.5, 16.5, 21.2]
metrics['point_energies_ev'] = point_energies
metrics['experimental_cross_sections_mb'] = np.interp(point_energies, energy_exp, sigma_exp).tolist()
for method in methods:
    curve = np.asarray(report['methods'][method]['cross_section_mb'])
    calculated = np.interp(common, energy_calc, curve)
    j = int(np.argmax(calculated))
    metrics['methods'][method] = {
        'rmse_mb': float(np.sqrt(trapezoid((calculated-reference)**2, common)/(common[-1]-common[0]))),
        'mae_mb': float(trapezoid(abs(calculated-reference), common)/(common[-1]-common[0])),
        'point_cross_sections_mb': np.interp(point_energies, energy_calc, curve).tolist(),
        'maximum_energy_ev': float(common[j]), 'maximum_cross_section_mb': float(calculated[j]),
        'integrated_intervals_mb_ev': {},
    }
    for lo, hi in ((10., 12.5), (12.61, 23.8), (10., 23.8)):
        x = np.linspace(lo, hi, 4001)
        expected = trapezoid(np.interp(x, energy_exp, sigma_exp), x)
        actual = trapezoid(np.interp(x, energy_calc, curve), x)
        metrics['methods'][method]['integrated_intervals_mb_ev'][f'{lo}-{hi}'] = {
            'experiment': float(expected), 'theory': float(actual), 'ratio': float(actual / expected),
        }
# Independent experimental overlap; do not splice different datasets together.
x, y = experiments[1]
inside = (x >= 13.1) & (x <= 23.8)
metrics['samson_vs_kameta_overlap'] = {
    'number_of_points': int(inside.sum()),
    'mean_absolute_difference_mb': float(np.mean(abs(y[inside]-np.interp(x[inside],energy_exp,sigma_exp)))),
}
(folder / 'experimental_comparison.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')

plt.rcParams.update({'font.size': 10.5, 'axes.labelsize': 11, 'axes.titlesize': 12,
                     'axes.linewidth': .8, 'pdf.fonttype': 42,
                     'xtick.direction': 'in', 'ytick.direction': 'in'})
fig, axes = plt.subplots(2, 1, figsize=(8.3, 7.), gridspec_kw={'height_ratios': [1.25, 1.]})
for ax in axes:
    for method, label, color, style in zip(methods, labels, colors, styles):
        ax.plot(energy_calc, report['methods'][method]['cross_section_mb'],
                color=color, ls=style, lw=1.6, label=label)
    ax.plot(*experiments[0], color='black', lw=1.7, label='Kameta 2002, 298 K', zorder=5)
    ax.plot(*experiments[1], 'o', color='#555555', mfc='white', ms=3.5,
            label='Samson 1989, 298 K', zorder=6)
    ax.plot(*experiments[2], color='#888888', ls=':', lw=1.4,
            label='Lee & Chiang 1983, 293 K', zorder=4)
    ax.set_ylabel('Absorption cross section (Mb)')
    ax.set_xlabel('Photon energy (eV)')
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set(xlim=(8.5,24), ylim=(0,565), title='Methane: absolute absorption, without alignment or rescaling')
axes[0].legend(frameon=False, ncol=2, fontsize=9, loc='upper left')
axes[1].set(xlim=(8.5,16), ylim=(0,140), title='Low-energy region (expanded vertical scale)')
axes[1].axvline(12.61, color='#777777', ls=':', lw=.9)
axes[1].text(12.72, 130, 'Experimental ionization onset: 12.61 eV', color='#555555', fontsize=9)
fig.subplots_adjust(left=.12,right=.98,top=.94,bottom=.17,hspace=.44)
fig.text(.12,.035, 'Theory: RHF / aug-cc-pVDZ / Weigend RI; windowed full BSE; HWHM 0.15 eV\n'
         'Experiment: MPI-Mainz UV/VIS Atlas; individual datasets shown without stitching',fontsize=9,color='#444444')
for suffix in ('png','pdf'):
    fig.savefig(folder / f'methane_bse_experiment.{suffix}', dpi=220)
plt.close(fig)
print(json.dumps(metrics, indent=2))
