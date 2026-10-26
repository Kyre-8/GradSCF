"""Plot verified X13 full-wavefunction NAC and complete-pair response data."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

folder=Path('reproducibility/ad_showcase/X13_h3_fci_nac')
native=json.loads((folder/'native.json').read_text())
reference=json.loads((folder/'pyscf.json').read_text())
assert native['passed'] and reference['passed']
assert reference['native_json_sha256']==hashlib.sha256((folder/'native.json').read_bytes()).hexdigest()
radius=np.asarray([row['radius_bohr'] for row in native['rows']])
gap=np.asarray([row['gap_hartree'] for row in native['rows']])
coupling=np.asarray([np.linalg.norm(row['mode_nac_per_bohr']) for row in native['rows']])
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.linewidth':.8})
fig,axes=plt.subplots(2,2,figsize=(10,7.8))
axes[0,0].plot(np.r_[0.,radius[::-1]],np.r_[0.,gap[::-1]],'-o',color='#0072B2',ms=4)
axes[0,0].axvline(0,color='#777777',ls=':',lw=1)
axes[0,0].set(xlabel=r'Distortion radius $\rho$ (bohr)',ylabel=r'$E_1-E_0$ (Eh)',
              title='Two lowest doublets meet at the symmetric geometry')
axes[0,1].loglog(gap,coupling,'o-',color='#D55E00',ms=4,label='Full wavefunction NAC')
axes[0,1].loglog(gap,coupling[-1]*gap[-1]/gap,'--',color='#777777',lw=1,label=r'$1/\Delta E$ guide')
axes[0,1].invert_xaxis()
axes[0,1].set(xlabel='Energy gap (Eh)',ylabel=r'$\Vert d_{01,Q}\Vert$ (bohr$^{-1}$)',
              title=f'Near-crossing slope: {native["near_crossing_log_slope"]:.6f}')
axes[0,1].legend(frameon=False,fontsize=9)
point=native['rows'][1]
values=np.asarray(point['nac_01_per_bohr'])
limit=np.max(abs(values))
im=axes[1,0].imshow(values,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
axes[1,0].set_xticks(range(3),['x','y','z'])
axes[1,0].set_yticks(range(3),['H1','H2','H3'])
axes[1,0].set_title(r'Cartesian $d_{01}$ at $q=(0.02,0.01)$ bohr')
for i in range(3):
    for j in range(3):
        label=f'{values[i,j]:.3f}'
        label='0.000' if label=='-0.000' else label
        axes[1,0].text(j,i,label,ha='center',va='center',
                       color='white' if abs(values[i,j])>.6*limit else 'black')
fig.colorbar(im,ax=axes[1,0],label=r'bohr$^{-1}$',fraction=.05,pad=.03)
for mode,color in enumerate(('#0072B2','#D55E00')):
    rows=[row for row in reference['complete_pair_checks'] if row['mode']==mode]
    axes[1,1].loglog([r['step_bohr'] for r in rows],[r['projector_error'] for r in rows],
                      'o-',color=color,ms=4,label=f'Distortion {mode+1}')
axes[1,1].axhline(reference['gates']['complete_pair_fd_atol'],color='#555555',ls='--',lw=1,label='FD gate')
axes[1,1].set(xlabel='PySCF difference step (bohr)',ylabel='Projector derivative error (bohr$^{-1}$)',
              title='Exact crossing: finite complete-pair response')
axes[1,1].legend(frameon=False,fontsize=8)
for axis in (axes[0,0],axes[0,1],axes[1,1]):
    axis.spines[['top','right']].set_visible(False)
fig.suptitle('H3 / STO-3G: nonadiabatic coupling in a controlled full-FCI model',fontsize=13,y=.98)
fig.tight_layout(rect=(0,.12,1,.95))
fig.text(.075,.035,'Three AOs, (2α, 1β), nine determinants; no HF. CI, moving orbital frame and AO cross overlap are differentiated.\n'
         'Lab-frame BO couplings: no ETF/ERF. Individual NACs at the exact crossing are undefined; only the complete pair is differentiated.',fontsize=8.8)
for suffix in ('png','pdf'):
    fig.savefig(folder/f'h3_nonadiabatic_coupling.{suffix}',dpi=220)
plt.close(fig)
print(folder/'h3_nonadiabatic_coupling.png')
