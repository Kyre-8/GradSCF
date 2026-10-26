"""Paper-style smooth density contours on Al/Mg crystallographic sections.

Reads actual periodic density volumes exported by atlas_compare.py. Smoothness
comes from exact evaluation of the density's discrete Fourier series on a finer
plane grid, not Gaussian filtering or a changed density. Exports one 2x2 figure
for each lambda. GradSCF is red solid; DFTpy is blue dotted.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

ROOT=Path(__file__).resolve().parent


def fourier_plane(density, lattice, translation_u, translation_v, *, origin=None, npoints=256):
    """Evaluate a periodic volume on a plane spanning two lattice translations.

    Lattice/translation/origin share units. Restriction maps each 3D Fourier mode
    to an integer 2D mode. Coefficients with the same projected mode are summed,
    then a zero-padded inverse FFT evaluates the plane without changing them.
    """
    density=np.asarray(density)
    mesh=density.shape
    if len(mesh)!=3 or any(n%2==0 for n in mesh):
        raise ValueError('Expected a 3D odd-mesh periodic density.')
    inverse=np.linalg.inv(np.asarray(lattice))
    spans=np.stack([translation_u,translation_v])@inverse
    integer_spans=np.rint(spans).astype(int)
    if not np.allclose(spans,integer_spans,atol=1e-10,rtol=0):
        raise ValueError('Plane edges must be lattice translations.')
    modes=np.stack(np.meshgrid(*[np.rint(np.fft.fftfreq(n)*n).astype(int) for n in mesh],
                               indexing='ij'),axis=-1).reshape(-1,3)
    projected=modes@integer_spans.T
    if npoints<=2*np.max(np.abs(projected)):
        raise ValueError('Plane sampling would alias projected Fourier modes.')
    coefficients=np.fft.fftn(density).ravel()/density.size
    if origin is not None:
        coefficients*=np.exp(2j*np.pi*(modes@(np.asarray(origin)@inverse)))
    spectrum=np.zeros((npoints,npoints),dtype=complex)
    np.add.at(spectrum,(projected[:,0]%npoints,projected[:,1]%npoints),coefficients)
    return np.fft.ifft2(spectrum*npoints**2).real


def plot_contours(volumes, weight, tag, *, reference='dftpy',
                  labels=('GradSCF', 'DFTpy'), subtitle=None, note=None,
                  filename=None, contour_levels=None, label_positions=None):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'axes.linewidth':1.2,
        'pdf.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})
    al=volumes[f'Al_{tag}_lattice'];mg=volumes[f'Mg_{tag}_lattice']
    a=2*al[0,1]
    sections=[
        ('Al',r'(a) Al (001)',np.array([a,0,0]),np.array([0,a,0]),r'$x$ (Å)',r'$y$ (Å)'),
        ('Al',r'(b) Al (011)',np.array([0,a,-a]),np.array([a,0,0]),r'Along $[01\bar{1}]$ (Å)',r'Along [100] (Å)'),
        ('Mg',r'(c) Mg (0001)',mg[0],mg[1],r'$a_1$ coordinate (Å)',r'$a_2$ coordinate (Å)'),
        ('Mg',r'(d) Mg $(01\bar{1}0)$',mg[0],mg[2],r'Along $a_1$ (Å)',r'Along $c$ (Å)'),
    ]
    planes=[]
    for symbol,title,u,v,xlabel,ylabel in sections:
        lattice=volumes[f'{symbol}_{tag}_lattice']
        pair=[100*fourier_plane(volumes[f'{symbol}_{tag}_{code}'],lattice,u,v)
              for code in ('gradscf',reference)]
        planes.append(pair)
    levels={}
    for symbol in ('Al','Mg'):
        selected=[a for section,pair in zip(sections,planes) if section[0]==symbol for a in pair]
        lo=min(a.min() for a in selected);hi=max(a.max() for a in selected)
        levels[symbol]=(MaxNLocator(nbins=9).tick_values(lo,hi)
                        if contour_levels is None else np.asarray(contour_levels[symbol]))
        levels[symbol]=levels[symbol][(levels[symbol]>lo)&(levels[symbol]<hi)]

    fig,axes=plt.subplots(2,2,figsize=(10.2,10.0))
    fig.subplots_adjust(left=.095,right=.97,bottom=.18,top=.84,hspace=.34,wspace=.26)
    fig.suptitle('Electron-density contours',fontsize=16,y=.98)
    if subtitle is None:
        lam={1.:'1',.2:'1/5',1/9:'1/9'}[weight]
        subtitle=f'TF + λvW, λ = {lam}  ·  OEPP  ·  PZ-LDA'
    fig.text(.5,.94,subtitle,ha='center',fontsize=11)
    fig.legend([Line2D([],[],color='#D72828',lw=1.35),
                Line2D([],[],color='#153ADF',lw=1.6,ls=(0,(1.4,2.4)))],
               labels,loc='upper center',bbox_to_anchor=(.5,.915),ncol=2,frameon=False)
    for panel,(ax,section,pair) in enumerate(zip(axes.ravel(),sections,planes)):
        symbol,title,u,v,xlabel,ylabel=section
        x=np.linspace(0,np.linalg.norm(u),pair[0].shape[0]+1)
        y=np.linspace(0,np.linalg.norm(v),pair[0].shape[1]+1)
        values=[np.pad(a,((0,1),(0,1)),mode='wrap').T for a in pair]
        red=ax.contour(x,y,values[0],levels=levels[symbol],colors='#D72828',linewidths=1.)
        ax.contour(x,y,values[1],levels=levels[symbol],colors='#153ADF',
                   linewidths=1.2,linestyles=[(0,(1.4,2.4))])
        manual = False if label_positions is None else [
            (px*x[-1], py*y[-1]) for px,py in label_positions[panel]]
        selected_levels = levels[symbol][::2] if label_positions is None else levels[symbol]
        labels=ax.clabel(red,levels=selected_levels,inline=True,fontsize=8.7,
                         fmt='%g',colors='black',manual=manual)
        for label in labels:
            label.set_bbox(dict(facecolor='white',edgecolor='none',pad=.2))
            label.set_zorder(5)
        ax.set_xlim(0,x[-1]);ax.set_ylim(0,y[-1]);ax.set_box_aspect(.9)
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
        ax.set_title(title,fontsize=12,pad=9)
        ax.xaxis.set_major_locator(MultipleLocator(.8 if x[-1]<4.5 else 1.))
        ax.yaxis.set_major_locator(MultipleLocator(.8 if y[-1]<4.5 else 1.))
        ax.tick_params(direction='in',top=True,right=True,length=7,width=1.1)
        for spine in ax.spines.values():
            spine.set_visible(True);spine.set_color('black')
    fig.text(.055,.087,
        'Contour labels show 100n, with n in electron/Bohr³; coordinates are in Å.\n'
        'The same contour levels are used for both codes and both sections of each material.\n'
        'Periodic Fourier interpolation adds sampling points without blurring the calculated density.',fontsize=9,va='bottom')
    if note is None:
        note='These are TF + λvW results at the stated reference volumes, not the paper’s WGC Table 1.'
    fig.text(.055,.033,
        'Panels have independent axis scales. Mg basal coordinates follow a₁,a₂ (120° apart); ideal c/a is assumed.\n'
        +note,fontsize=8.5,color='#444',va='bottom')
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    if filename is None:
        filename=f'atlas_density_contours_lambda_{tag}'
    for suffix in ('png','pdf'):
        fig.savefig(out/f'{filename}.{suffix}',dpi=240,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def main():
    with np.load(ROOT/'atlas_density_volumes.npz',allow_pickle=False) as volumes:
        for weight,tag in ((1.,'1'),(.2,'1_5'),(1/9,'1_9')):
            plot_contours(volumes,weight,tag)
    print('Saved smooth contour figures for lambda=1, 1/5, 1/9.')


if __name__=='__main__':
    main()
