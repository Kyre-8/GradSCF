"""Plot recorded 100-step H2 FCI-supervised losses; no calculations rerun."""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = Path(__file__).with_name('h2_fci_results')
report = json.loads((folder/'summary.json').read_text())
styles = [('fixed_density', 'Fixed HF density', '#365F9E', '-', None, 0),
          ('explicit', 'Self-consistent / explicit', '#CB6A28', '-', 'o', 0),
          ('implicit', 'Self-consistent / implicit', '#167F70', '--', '^', 5)]
plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
    'axes.spines.right':False, 'pdf.fonttype':42, 'figure.facecolor':'white'})
fig, axes = plt.subplots(1,3,figsize=(12.3,4.2))
for name, label, color, linestyle, marker, offset in styles:
    with (folder/f'{name}.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    x = np.array([int(r['step']) for r in rows])
    if not np.array_equal(x,np.arange(101)):
        raise ValueError('Expected steps 0 through 100 in every curve')
    for ax, key, scale in zip(axes,('loss','mse','mae'),(1.,1.,1000.)):
        y = np.array([float(r[key]) for r in rows])*scale
        ax.plot(x,y,label=label,color=color,ls=linestyle,lw=1.65,
            marker=marker,markevery=(offset,10),ms=4.2,mfc='white',mew=1.)
        ax.set_xlim(0,100)
        ax.set_xlabel('Completed optimizer updates')
        ax.grid(axis='y',alpha=.18)
for ax,title,ylabel in zip(axes,('Total loss','MSE component','MAE component'),
    ('δ² + |δ|,  δ = (E − EFCI) / (1 Ha)','Energy MSE / Ha²','Energy MAE / mHa')):
    ax.set_title(title,loc='left',fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
fig.suptitle('H₂ / 6-31G* · Four-layer density-matrix MLP · FCI supervision',fontsize=14,y=.99)
handles,labels = axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False)
fig.subplots_adjust(left=.075,right=.985,bottom=.23,top=.75,wspace=.31)
fig.text(.075,.09,
    f"EFCI = {report['fci_energy_hartree']:.12f} Ha. Adam lr=0.002; seed=0; CPU float64; 100/100 updates accepted per mode.",fontsize=9)
fig.text(.075,.035,
    'Explicit and implicit curves overlap. Fixed-density loss uses the unchanged HF state; the other curves use converged SCF states.',fontsize=9,color='#444444')
for extension in ('png','pdf'):
    fig.savefig(folder/f'loss_curves.{extension}',dpi=240,bbox_inches='tight')
plt.close(fig)
print('Saved',folder/'loss_curves.png')
