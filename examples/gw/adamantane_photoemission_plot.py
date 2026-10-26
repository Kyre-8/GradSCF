"""Plot calculated adamantane spectra against the separately sourced experiment.

Run adamantane_photoemission.py first. No energy alignment or linewidth fit is
performed. A common Gaussian sigma=0.05 eV is applied only for display; raw
spectral densities remain in spectra.csv. This is not an experimental
instrument-response model or a Franck-Condon calculation.
"""
from pathlib import Path
import json

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
folder = root/'reproducibility/gw_bse/adamantane_native'
reference = root/'reproducibility/gw_bse/adamantane_photoemission'
data = np.genfromtxt(folder/'spectra.csv', delimiter=',', names=True)
inputs = np.load(folder/'phonon_inputs.npz')
experiment = np.loadtxt(reference/'experiment.csv', delimiter=',', skiprows=1)
summary = json.loads((folder/'results.json').read_text())
x = data['binding_energy_ev']
sigma = .05
smooth = lambda values: gaussian_filter1d(values, sigma/(x[1]-x[0]), mode='constant')
colors = {'gw':'#2166ac', 'fan':'#ce8426', 'fan_dw':'#a63d40'}
labels = {'gw':r'G$_0$W$_0$ poles', 'fan':'+ Fan', 'fan_dw':'+ Fan + DW'}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(2, 2, figsize=(12,8), constrained_layout=True)
for name in colors:
    axes[0,0].plot(x, smooth(data[name]), color=colors[name], lw=1.4, label=labels[name])
    axes[0,1].plot(x, smooth(data[name+'_homo']), color=colors[name], lw=1.6, label=labels[name])
axes[0,0].set(xlim=(8,18), ylabel='Spectral density (eV$^{-1}$)',
              title='Calculated valence spectrum')
axes[0,0].legend(frameon=False)
ip = summary['gw_homo_ip_ev']
hf_ip = summary['hf_homo_ip_ev']
axes[0,1].set(xlim=(min(ip,hf_ip)-1,max(ip,hf_ip)+1.5),
              ylabel='Projected density (eV$^{-1}$)', title='Three-dimensional HOMO subspace')
# Compare relative shapes on the same absolute energy axis. Area normalization
# changes intensity units only; no computed peak is moved toward experiment.
window = (x>=8.)&(x<=15.8)
measured = np.interp(x, experiment[:,0], experiment[:,1], left=0., right=0.)
area = lambda curve: np.trapezoid(curve[window], x[window])
axes[1,0].plot(x, measured/area(measured), color='#252525', lw=1.5,
              label='Experiment: Gali et al., Fig. 1b')
for name in ('gw','fan_dw'):
    curve = smooth(data[name])
    axes[1,0].plot(x, curve/area(curve), color=colors[name], lw=1.4,
                  ls='--' if name=='gw' else '-', label=labels[name])
axes[1,0].set(xlim=(8.,15.8), ylabel='Area-normalized intensity (eV$^{-1}$)',
              title='Experiment comparison; no energy shift')
axes[1,0].legend(frameon=False, fontsize=8)
# The Frobenius norm over the whole degenerate electronic subspace is invariant
# under its basis rotations. Sum over degenerate vibrational subspaces too.
omega = inputs['energies']*219474.6313632
strength = np.sum(abs(inputs['couplings'][:,35:38,35:38])**2,axis=(1,2))/3
strength *= (27.211386245988*1000)**2
groups = np.split(np.arange(len(omega)),np.flatnonzero(np.diff(omega)>1.)+1)
centers = [omega[g].mean() for g in groups]
weights = [strength[g].sum() for g in groups]
axes[1,1].vlines(centers, 0., weights, color='#217a72', lw=2)
axes[1,1].scatter(centers, weights, color='#217a72', s=12)
axes[1,1].set(xlabel='Vibrational wavenumber (cm$^{-1}$)',
              ylabel=r'$\sum_{\nu\in group}\|g^\nu_{HOMO}\|_F^2/3$ (meV$^2$)',
              title='HOMO coupling; modes grouped within 1 cm$^{-1}$')
for ax in axes.flat:
    ax.set_ylim(bottom=0)
for ax in (axes[0,0],axes[0,1],axes[1,0]):
    ax.set_xlabel('Binding energy (eV)')
fig.suptitle('Isolated adamantane: native GradSCF RHF/STO-3G + GW + vibrations\n'
             r'300 K; numerical $\eta=10$ meV; display Gaussian $\sigma=50$ meV',fontsize=13)
fig.savefig(folder/'adamantane_native.png',dpi=180)
fig.savefig(folder/'adamantane_native.pdf')
plt.close(fig)
print(folder/'adamantane_native.png')

resolution = np.genfromtxt(folder/'resolution_check.csv', delimiter=',', names=True)
def weight_interval(curve):
    cumulative = cumulative_trapezoid(curve,x,initial=0.)
    low,high = np.interp([.1,.9],cumulative/cumulative[-1],x)
    return {'integrated_weight_7_19_ev':float(cumulative[-1]),
            'q10_ev':float(low),'q90_ev':float(high),'central_80_percent_width_ev':float(high-low)}

first_band = (x>=8.8)&(x<=10.7)
metrics = {'display_gaussian_sigma_ev':sigma,'comparison_area_window_ev':[8.,15.8],
           'experimental_first_band_peak_ev':float(x[first_band][np.argmax(measured[first_band])]),
           'homo':{name:dict(weight_interval(data[name+'_homo']),
                    displayed_peak_ev=float(x[np.argmax(smooth(data[name+'_homo']))])) for name in colors},
           'resolution':{field:weight_interval(resolution[field]) for field in resolution.dtype.names[1:]}}
(folder/'comparison_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
fig, ax = plt.subplots(figsize=(7.5,4), constrained_layout=True)
for field,color,style in (('eta_5_mev','#217a72','-'),('eta_10_mev','#a63d40','--'),
                          ('eta_20_mev','#2166ac',':')):
    ax.plot(x,resolution[field],color=color,ls=style,lw=1.3,
            label=field.replace('eta_','eta = ').replace('_mev',' meV'))
ax.set(xlim=(ip-1.5,ip+2.),ylim=(0,None),xlabel='Binding energy (eV)',
       ylabel='HOMO projected density (eV$^{-1}$)',
       title='Numerical resolution check: GW + Fan + DW\nRaw spectrum; no Gaussian display smoothing')
ax.legend(frameon=False)
fig.savefig(folder/'resolution_check.png',dpi=180)
fig.savefig(folder/'resolution_check.pdf')
plt.close(fig)
