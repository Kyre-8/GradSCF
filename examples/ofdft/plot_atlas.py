"""Plot measured ATLAS-style OFDFT comparisons; no calculations or CLI.

Run after atlas_compare.py. Reads JSON only; exports PNG and vector PDF figures
under examples/ofdft/figures. No smoothing or fabricated uncertainty is used.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ase import units

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'
COLORS={1.:'#285E8E', .2:'#CF7929', 1/9:'#218577'}
LABELS={1.:r'$\lambda=1$', .2:r'$\lambda=1/5$', 1/9:r'$\lambda=1/9$'}
INK='#213448'
STYLES={1.:'-', .2:'--', 1/9:':'}
MARKERS={1.:'o', .2:'s', 1/9:'^'}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.titlesize':12,'axes.labelsize':10,'axes.labelcolor':INK,'text.color':INK,
        'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':'#A8B4BE',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'lines.linewidth':1.8,'figure.facecolor':'white','axes.facecolor':'white',
        'savefig.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none',
        'legend.frameon':False,'legend.fontsize':9})


def finish(fig,name,caption):
    fig.text(.055,.025,caption,fontsize=8.3,color='#586975',va='bottom')
    for suffix in ('png','pdf'):
        fig.savefig(OUT/f'{name}.{suffix}',dpi=220,bbox_inches='tight',pad_inches=.16)
    plt.close(fig)


def lambda_legend(fig):
    fig.legend([Line2D([],[],color=c,marker=MARKERS[w],ls=STYLES[w],markersize=5) for w,c in COLORS.items()],
               LABELS.values(),loc='upper center',bbox_to_anchor=(.52,.91),ncol=3)


def convergence(records):
    fig,axes=plt.subplots(1,2,figsize=(11.4,5.1),gridspec_kw={'width_ratios':[1.6,1.]})
    fig.subplots_adjust(left=.095,right=.97,bottom=.21,top=.76,wspace=.30)
    fig.suptitle('Grid convergence of TF + λvW energies',x=.055,ha='left',y=.99,
                 fontsize=16,fontweight='semibold')
    lambda_legend(fig)
    for panel,(ax,symbol) in enumerate(zip(axes,('Al','Mg'))):
        for weight,color in COLORS.items():
            rows=sorted([r for r in records if r['symbol']==symbol and r['lambda_vw']==weight],
                        key=lambda r:-r['spacing_angstrom'])
            differences=np.abs(np.diff([r['gradscf_energy_ha_per_atom'] for r in rows]))*units.Hartree*1000
            x=np.arange(len(differences))
            ax.plot(x,differences,marker=MARKERS[weight],ls=STYLES[weight],color=color,markersize=6,zorder=3)
            for i,value in enumerate(differences):
                if i==len(differences)-1:
                    ax.annotate(f'{value:.2g}',(i,value),xytext=(8,({1.:-4,.2:10,1/9:-5}[weight] if symbol=='Al' else 0)),textcoords='offset points',
                                fontsize=8.3,color=color,va='center')
        ticks=[f"{a['spacing_angstrom']:.2f} → {b['spacing_angstrom']:.2f}"
               for a,b in zip(rows[:-1],rows[1:])]
        ax.set_xticks(np.arange(len(ticks)),ticks)
        ax.set_yscale('log');ax.set_ylim(3e-5,4)
        ax.set_xlim(-.25,max(len(ticks)-1,0)+.8)
        ax.axhline(.1,color='#8B969E',ls='--',lw=1)
        ax.text(.025,.1,'0.1 meV/atom',transform=ax.get_yaxis_transform(),va='bottom',fontsize=8.5,color='#697780')
        ax.grid(axis='y',which='major',color='#E5E9ED',lw=.7,zorder=0)
        ax.set_title(f'({chr(97+panel)})  {symbol} · '+('fcc' if symbol=='Al' else 'hcp'),loc='left')
        ax.set_xlabel('Target spacing refinement (Å)')
        ax.set_ylabel(r'$|\Delta E|$ (meV / atom)')
    finish(fig,'atlas_mesh_convergence',
        'Successive GradSCF energies at fixed cell. TF + λvW / OEPP / PZ-LDA; Mg uses ideal c/a.\n'
        'Dashed line: comparison scale of 0.1 meV/atom. A successive difference is not a rigorous discretization-error bound.')


def agreement(records):
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.9))
    fig.subplots_adjust(left=.065,right=.985,bottom=.23,top=.77,wspace=.36)
    fig.suptitle('Independent implementations agree on the same grid',x=.05,ha='left',y=.99,
                 fontsize=16,fontweight='semibold')
    groups=list(dict.fromkeys((r['symbol'],r['spacing_angstrom']) for r in records))
    group_labels=[f'{s}\n{h:.2f} Å' for s,h in groups]
    for j,(weight,color) in enumerate(COLORS.items()):
        rows=[r for r in records if r['lambda_vw']==weight]
        x=np.array([groups.index((r['symbol'],r['spacing_angstrom'])) for r in rows])+(j-1)*.16
        axes[0].scatter(x,[r['energy_error_ha_per_atom'] for r in rows],c=color,s=30,marker=MARKERS[weight],label=LABELS[weight],zorder=3)
        axes[1].scatter(x,[r['relative_density_l2_error'] for r in rows],c=color,s=30,marker=MARKERS[weight],zorder=3)
        axes[2].scatter(x,[r['gradscf_residual'] for r in rows],c='#285E8E',s=23,zorder=3)
        axes[2].scatter(x,[r['dftpy_residual'] for r in rows],edgecolors='#CF7929',facecolors='none',s=32,marker='s',zorder=3)
    labels=[r'$|E_{\rm GradSCF}-E_{\rm DFTpy}|$ (Ha / atom)',r'Relative density $L^2$ error','Constrained residual norm']
    titles=['(a)  Total energy','(b)  Electron density','(c)  Stationarity checks']
    for ax,label,title in zip(axes,labels,titles):
        ax.set_xticks(np.arange(len(groups)),group_labels,fontsize=8)
        ax.set_yscale('log');ax.grid(axis='y',color='#E5E9ED',lw=.7)
        ax.set_ylabel(label);ax.set_title(title,loc='left')
        ax.set_xlim(-.45,len(groups)-.55)
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,1.20),ncol=3,handletextpad=.25,columnspacing=.8)
    axes[2].legend([Line2D([],[],color='#285E8E',marker='o',ls='none'),
                    Line2D([],[],color='#CF7929',marker='s',mfc='none',ls='none')],
                   ['GradSCF','DFTpy'],loc='upper right',bbox_to_anchor=(1,1.2),ncol=2)
    axes[2].axhline(1e-8,color='#285E8E',ls=':',lw=.9)
    axes[2].axhline(5e-5,color='#CF7929',ls=':',lw=.9)
    finish(fig,'atlas_code_agreement',
        '18 paired calculations. GradSCF: stationarity tolerance 10⁻⁸. DFTpy: energy tolerance 10⁻¹² Ha; independently checked residual < 5×10⁻⁵.\n'
        'Different stopping rules are intentional; these panels are not a speed comparison or a test against WGC Table 1.')


def closed_plane(a):
    a=np.asarray(a)
    return np.pad(a,((0,1),(0,1)),mode='wrap')


def density_planes(slices):
    fig=plt.figure(figsize=(11.8,8.6))
    grid=fig.add_gridspec(4,3,height_ratios=[1,.065,1,.065],left=.075,right=.965,
                        bottom=.155,top=.835,wspace=.32,hspace=.62)
    fig.suptitle('Density slices: GradSCF, DFTpy, and their difference',x=.055,ha='left',y=.985,
                 fontsize=16,fontweight='semibold')
    fig.text(.055,.914,r'TF + $\frac{1}{9}$vW · OEPP · PZ-LDA · target spacing 0.18 Å',fontsize=11)
    for row,symbol in enumerate(('Al','Mg')):
        record=next(r for r in slices if r['symbol']==symbol and r['lambda_vw']==1/9)
        a,b=closed_plane(record['gradscf_plane']),closed_plane(record['dftpy_plane'])
        x,y=np.linspace(0,1,a.shape[0]),np.linspace(0,1,a.shape[1])
        vmin,vmax=min(a.min(),b.min()),max(a.max(),b.max())
        axes=[fig.add_subplot(grid[2*row,k]) for k in range(3)]
        for ax,values,title in zip(axes[:2],(a,b),('GradSCF','DFTpy')):
            im=ax.pcolormesh(x,y,values.T,cmap='viridis',vmin=vmin,vmax=vmax,shading='auto',rasterized=True)
            ax.set_title(f'{symbol} · {title}',loc='left',fontsize=11)
        delta=(a-b)*1e8
        limit=max(abs(delta.min()),abs(delta.max()))
        difference=axes[2].pcolormesh(x,y,delta.T,cmap='RdBu_r',vmin=-limit,vmax=limit,
                                     shading='auto',rasterized=True)
        axes[2].set_title(f'{symbol} · GradSCF − DFTpy',loc='left',fontsize=11)
        for ax in axes:
            ax.set_aspect('equal');ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
            ax.set_xlabel('Fractional u');ax.set_ylabel('Fractional v')
        cbar=fig.colorbar(im,cax=fig.add_subplot(grid[2*row+1,:2]),orientation='horizontal')
        cbar.set_label('n (e / Bohr³)',fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        cb=fig.colorbar(difference,cax=fig.add_subplot(grid[2*row+1,2]),orientation='horizontal')
        cb.set_label('Δn (10⁻⁸ e / Bohr³)',fontsize=9)
        cb.ax.tick_params(labelsize=8)
    finish(fig,'atlas_density_slices',
        'Actual mesh values on the fractional w=0 plane, spanned by primitive a₁ and a₂; axes are fractional, not Cartesian.\n'
        'Each row shares its absolute-density scale. Difference maps use a separate symmetric scale. Periodic endpoints are repeated; no smoothing.')


def profiles(slices):
    fig,axes=plt.subplots(2,2,figsize=(11.4,7.1),sharex=True,gridspec_kw={'height_ratios':[1.6,1]})
    fig.subplots_adjust(left=.09,right=.97,bottom=.16,top=.78,wspace=.26,hspace=.22)
    fig.suptitle('Planar density profiles for TF + λvW',x=.055,ha='left',y=.99,
                 fontsize=16,fontweight='semibold')
    handles=[Line2D([],[],color=c,ls=STYLES[w],lw=2,label=LABELS[w]) for w,c in COLORS.items()]
    handles += [Line2D([],[],color=INK,lw=1.7,label='GradSCF'),
                Line2D([],[],color=INK,marker='o',mfc='none',ls='none',label='DFTpy')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.925),ncol=5)
    for col,symbol in enumerate(('Al','Mg')):
        for weight,color in COLORS.items():
            r=next(r for r in slices if r['symbol']==symbol and r['lambda_vw']==weight)
            a,b=np.asarray(r['gradscf_planar_mean']),np.asarray(r['dftpy_planar_mean'])
            x=np.linspace(0,1,len(a)+1)
            a,b=np.r_[a,a[0]],np.r_[b,b[0]]
            axes[0,col].plot(x,a,color=color,ls=STYLES[weight])
            axes[0,col].plot(x[::2],b[::2],'o',mfc='none',mec=color,ms=5,lw=0)
            axes[1,col].plot(x,(a-b)*1e8,color=color,ls=STYLES[weight])
        axes[0,col].set_title(f'({chr(97+col)})  {symbol} · planar mean density',loc='left')
        axes[0,col].set_ylabel('Mean n (e / Bohr³)')
        axes[1,col].set_ylabel('Δ mean n (10⁻⁸ e / Bohr³)')
        axes[1,col].set_xlabel('Fractional u')
        axes[1,col].axhline(0,color='#A8B4BE',lw=.7)
        for ax in axes[:,col]:
            ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1]);ax.grid(axis='y',color='#E5E9ED',lw=.7)
    finish(fig,'atlas_density_profiles',
        'Planar mean over fractional v,w at each u, on the same 0.18 Å target mesh. Lines join measured mesh points.\n'
        'The lower panels resolve differences hidden by the overlaid density curves. Different λ values define different kinetic functionals.')


def main():
    style();OUT.mkdir(exist_ok=True)
    records=json.loads((ROOT/'atlas_results.json').read_text())['calculations']
    slices=json.loads((ROOT/'atlas_density_slices.json').read_text())['density_slices']
    convergence(records);agreement(records);density_planes(slices);profiles(slices)
    print('Saved four figures as PNG and PDF:',OUT)


if __name__=='__main__':
    main()
