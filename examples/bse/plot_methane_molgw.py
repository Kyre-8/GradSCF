"""Overlay the archived GradSCF/MolGW methane reference and experimental data.

MolGW's BSE input adapter and RI cutoff alignment are documented in the
reproducibility directory. No calculation, download or spectral fit is run.
Download the uncommitted experimental table and verify its SHA256 checksum
first, following experiment/README.md beside the archived spectrum.json.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import h, c, e

folder = Path(__file__).resolve().parents[2] / 'reproducibility/gw_bse/methane_aug_cc_pvdz'
native = json.loads((folder / 'spectrum.json').read_text())
reference = json.loads((folder / 'molgw_comparison.json').read_text())
methods = ('g0w0', 'evgw0', 'evgw')
labels = (r'$G_0W_0$', r'ev$GW_0$', r'ev$GW$')
colors = ('#0072B2', '#D55E00', '#009E73')
styles = ('-', '--', '-.')
experimental_path = folder / 'experiment/CH4_Kameta(2002)_298K_52-125nm.txt'
if not experimental_path.is_file():
    raise FileNotFoundError(
        f'Missing experimental data: {experimental_path}. Download the exact '
        f'sources.json URL and verify SHA256 as described in {folder / "experiment/README.md"}.'
    )
experimental = np.loadtxt(experimental_path)
energy_exp = h * c / e * 1e9 / experimental[:, 0]

plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.linewidth': .8,
                     'pdf.fonttype': 42, 'xtick.direction': 'in', 'ytick.direction': 'in'})
fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.8), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1.2], 'hspace': .18})
for method, label, color, style in zip(methods, labels, colors, styles):
    record = reference['methods'][method]['matched_cutoff']
    x = np.asarray(record['spectrum_energy_ev'])
    y = np.asarray(record['cross_section_mb'])
    axes[0].plot(native['energy_ev'], native['methods'][method]['cross_section_mb'],
                 color=color, ls=style, lw=1.7, label='GradSCF ' + label)
    selected = np.flatnonzero((x >= 8) & (x <= 24))[::2]
    axes[0].plot(x[selected], y[selected], 'o', ms=3.1, mfc='none', mew=.8,
                 color=color, label='MolGW ' + label)
    axes[1].plot(x, record['cross_section_difference_mb'], color=color,
                 ls=style, lw=1.4, label=label)
axes[0].plot(energy_exp, experimental[:, 1] * 1e18, color='black', lw=1.5,
             label='Experiment: Kameta 2002')
axes[0].set_title('Methane: GradSCF and MolGW with matched screening and RI rank', fontsize=12, pad=12)
axes[0].set_ylabel('Absorption cross section (Mb)')
axes[0].set_ylim(0, 565)
axes[0].legend(frameon=False, ncol=2, fontsize=9, loc='upper left')
axes[1].axhline(0, color='#777777', lw=.8)
axes[1].set_ylabel('GradSCF − MolGW\n(Mb)')
axes[1].set_xlabel('Photon energy (eV)')
axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
for ax in axes:
    ax.set_xlim(8, 24)
    ax.spines[['top', 'right']].set_visible(False)
fig.subplots_adjust(left=.13, right=.98, top=.93, bottom=.17)
fig.text(.13, .05, 'Both: 109 RI directions; 5 occupied + 29 optical virtuals; full G/W sums; HWHM 0.15 eV\n'
         'MolGW: documented BSE input adapter and auxiliary metric cutoff 10⁻¹⁰ (default: 10⁻⁶)',
         fontsize=8.5, color='#444444')
for suffix in ('png', 'pdf'):
    fig.savefig(folder / f'methane_molgw_comparison.{suffix}', dpi=220)
plt.close(fig)
print(folder / 'methane_molgw_comparison.png')
