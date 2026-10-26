"""Plot the saved methane GW/BSE calculation; no SCF or GW is rerun.

Run examples/bse/methane_gw_spectrum.py first. The spectra are absolute
orientationally averaged cross sections, with no per-method normalization.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

output = Path('outputs/methane_gw_bse')
report = json.loads((output / 'spectrum.json').read_text())
methods = ('g0w0', 'evgw0', 'evgw')
labels = (r'$G_0W_0$ + BSE', r'ev$GW_0$ + BSE', r'ev$GW$ + BSE')
colors = ('#0072B2', '#D55E00', '#009E73')
styles = ('-', '--', '-.')
energy = np.asarray(report['energy_ev'])

# Sum strengths inside numerically degenerate manifolds; individual
# eigenvectors within a manifold have an arbitrary orientation.
groups = {}
for method in methods:
    roots = np.asarray(report['methods'][method]['excitation_ev'])
    strengths = np.asarray(report['methods'][method]['oscillator_strength'])
    bands = []
    for root, strength in zip(roots, strengths):
        if bands and abs(root - bands[-1][0]) < 1e-5:
            bands[-1][1] += strength
            bands[-1][2] += 1
        else:
            bands.append([root, strength, 1])
    groups[method] = np.asarray(bands)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.labelsize': 12,
    'axes.linewidth': .8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
})
fig, axes = plt.subplots(4, 1, figsize=(8.4, 8.0), sharex=True,
                         gridspec_kw={'height_ratios': [3.5, 1, 1, 1], 'hspace': .12})
for method, label, color, style in zip(methods, labels, colors, styles):
    axes[0].plot(energy, report['methods'][method]['cross_section_mb'],
                 color=color, ls=style, lw=1.9, label=label)
axes[0].set_ylabel('Absorption cross section (Mb)')
axes[0].set_ylim(bottom=0)
axes[0].legend(frameon=False, fontsize=11, loc='upper left')
axes[0].set_title(r'Methane: GW self-consistency and the BSE absorption spectrum',
                  loc='left', fontsize=13, pad=14)
visible_max = max(np.max(b[(b[:, 0] >= energy[0]) & (b[:, 0] <= energy[-1]), 1])
                  for b in groups.values())
for ax, method, label, color in zip(axes[1:], methods, labels, colors):
    bands = groups[method]
    ax.vlines(bands[:, 0], 0, bands[:, 1], color=color, lw=1.5)
    ax.set_ylim(0, visible_max * 1.18)
    ax.text(.015, .73, label, color=color, transform=ax.transAxes, fontsize=11)
    ax.set_ylabel(r'$\sum f$')
for ax in axes:
    ax.set_xlim(energy[0], energy[-1])
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=10)
axes[-1].set_xlabel('Photon energy (eV)')
fig.subplots_adjust(left=.11, right=.97, top=.92, bottom=.145)
fig.text(.11, .055,
         f"RHF / {report['basis']} / {report['auxbasis']} RI; full singlet BSE\n"
         f"QP: {report['nocc']} occupied + {len(report['optical_virtual'])} virtual; "
         f"G/W sums: all {report['nmo']} MOs; spectral HWHM = {report['spectral_hwhm_ev']:.2f} eV",
         fontsize=9.4, color='#444444', linespacing=1.6)
for suffix in ('png', 'pdf', 'svg'):
    fig.savefig(output / f'methane_gw_bse.{suffix}', dpi=220)
plt.close(fig)
print(output / 'methane_gw_bse.png')

# Show the small method-dependent shifts on a common, unnormalized scale.
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), constrained_layout=True)
for ax, limits, title in zip(axes, ((10.4, 11.3), (16., 17.)),
                            ('First bright band', 'Strongest band in the plotted range')):
    for method, label, color, style in zip(methods, labels, colors, styles):
        ax.plot(energy, report['methods'][method]['cross_section_mb'],
                color=color, ls=style, lw=1.9, label=label)
    ax.set(xlim=limits, xlabel='Photon energy (eV)', title=title)
    selected = (energy >= limits[0]) & (energy <= limits[1])
    ax.set_ylim(0, 1.08 * max(np.max(np.asarray(report['methods'][m]['cross_section_mb'])[selected])
                             for m in methods))
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Absorption cross section (Mb)')
axes[0].legend(frameon=False, fontsize=9)
for suffix in ('png', 'pdf'):
    fig.savefig(output / f'methane_gw_bse_zoom.{suffix}', dpi=220)
plt.close(fig)
