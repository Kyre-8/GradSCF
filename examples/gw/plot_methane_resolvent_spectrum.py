"""Plot only freshly computed native-resolvent and pinned MolGW references."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FOLDER = Path('outputs/methane_resolvent_spectrum')
reference = json.loads((FOLDER/'molgw_comparison.json').read_text())
native = json.loads((FOLDER/'spectrum.json').read_text())
assert native['complete']
assert reference['native_json_sha256'] == hashlib.sha256((FOLDER/'spectrum.json').read_bytes()).hexdigest()
plt.rcParams.update({'font.size':10, 'axes.linewidth':.8, 'pdf.fonttype':42,
                     'xtick.direction':'in', 'ytick.direction':'in'})
fig, axes = plt.subplots(2, 3, figsize=(12, 6.7), sharex=True,
                          gridspec_kw={'height_ratios':[2.3, 1], 'hspace':.14, 'wspace':.28})
for column, (method, label) in enumerate(zip(('g0w0','evgw0','evgw'),
                                            (r'$G_0W_0$',r'ev$GW_0$',r'ev$GW$'))):
    matched = reference['methods'][method]['matched_cutoff']
    default = reference['methods'][method]['default_cutoff']
    x = np.asarray(matched['spectrum_energy_ev'])
    y = np.asarray(matched['cross_section_mb'])
    difference = np.asarray(matched['cross_section_difference_mb'])
    axes[0,column].plot(x, y+difference, color='#0072B2', lw=1.8, label='GradSCF, existing resolvent')
    axes[0,column].plot(x, y, color='#D55E00', lw=1.2, ls='--', label='MolGW, RI rank 109')
    axes[0,column].plot(default['spectrum_energy_ev'], default['cross_section_mb'],
                        color='#777777', lw=.9, ls=':', label='MolGW default, rank 108')
    axes[0,column].set_title(label)
    axes[1,column].plot(x,difference,color='#0072B2',lw=1.2)
    axes[1,column].axhline(0,color='#777777',lw=.7)
    axes[1,column].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axes[1,column].set_xlabel('Photon energy (eV)')
    for axis in axes[:,column]:
        axis.set_xlim(8,24)
        axis.spines[['top','right']].set_visible(False)
    axes[0,column].set_ylim(bottom=0)
axes[0,0].set_ylabel('Absorption cross section (Mb)')
axes[1,0].set_ylabel('GradSCF − matched MolGW\n(Mb)')
axes[0,0].legend(frameon=False,fontsize=7.8,loc='upper left')
fig.suptitle('Methane: independent implementations of the same static BSE model',y=.98,fontsize=13)
fig.subplots_adjust(left=.08,right=.98,top=.91,bottom=.19)
fig.text(.08,.07,'Cartesian aug-cc-pVDZ / Weigend RI; all 61 G/W orbitals; QP 1–34; 145 singlet full-BSE roots.\n'
         'Same MolGW frequency grid for both curves; HWHM 0.15 eV. This is implementation consistency, not experimental validation.\n'
         'The separately recorded native resolvent branch passed; the original facade runtime failure remains archived.',
         fontsize=8.7,color='#444444',linespacing=1.5)
for suffix in ('png','pdf'):
    fig.savefig(FOLDER/f'methane_reference_replay.{suffix}',dpi=220)
plt.close(fig)
print(FOLDER/'methane_reference_replay.png',flush=True)
