"""Isolated adamantane PES: redraw a published experimental/theory benchmark.

All three curves are from Gali et al., Nat. Commun. 7,11327 (2016), Fig.1b.
This script performs no GradSCF electronic-structure or EP calculation.
It deliberately distinguishes static HR from dynamical SF+HR broadening.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parents[2] / 'reproducibility/gw_bse/adamantane_photoemission'
data = {name:np.loadtxt(folder/(name+'.csv'), delimiter=',', skiprows=1)
        for name in ('experiment','qp_hr','dynamic_sf_hr')}
poles = np.loadtxt(folder/'qp_positions.csv', skiprows=1)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                     'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(1, 2, figsize=(12.5,4.5), constrained_layout=True,
                         gridspec_kw={'width_ratios':[1.45,1]})
for ax in axes:
    for name, label, color, style in (
        ('qp_hr', 'Published GW poles + static HR envelope', '#2166ac', '--'),
        ('dynamic_sf_hr', 'Published dynamical e-vib SF + HR', '#b44242', '-'),
        ('experiment', 'Published experiment', '#222222', '-')):
        ax.plot(*data[name].T, label=label, color=color, ls=style, lw=1.7 if name!='experiment' else 1.3)
    ax.vlines(poles, -.10, -.01, color='#737373', lw=1.4)
    ax.set(xlabel='Ionization energy (eV)', ylim=(-.14,2.25))
axes[0].set(xlim=(8.6,15.8), ylabel='Intensity (common relative scale)', title='Gas-phase adamantane, C$_{10}$H$_{16}$')
axes[0].plot([], [], color='#737373', marker='|', ls='none', label='GW pole positions (ticks)')
axes[0].legend(frameon=False, fontsize=8.5, loc='upper right')
axes[1].set(xlim=(8.8,11.1), title='First photoemission feature')
axes[1].text(.97,.95,'Both colored curves include vibrations.\nRed additionally includes dynamical coupling.',
             ha='right', va='top', transform=axes[1].transAxes, fontsize=9)
fig.suptitle('Isolated-molecule benchmark: literature theory and experiment\nVector-redrawn from Gali et al. (2016), Fig. 1b — not a GradSCF calculation', fontsize=13)
fig.savefig(folder/'adamantane_literature.png', dpi=180)
fig.savefig(folder/'adamantane_literature.pdf')
plt.close(fig)
print(folder/'adamantane_literature.png')
