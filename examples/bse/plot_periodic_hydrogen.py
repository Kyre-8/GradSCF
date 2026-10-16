"""Compare per-cell IP and BSE-body strengths on the identical QP spectrum."""
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gradscf.pbc.optics import broaden_spectrum, HARTREE_TO_EV
from periodic_hydrogen import OUTPUT


def plot():
    report = json.loads((OUTPUT/'results.json').read_text())
    assert report['complete'] and not report['gw_fc']
    with np.load(OUTPUT/'arrays.npz') as data:
        energies = [data['ip_energies'], data['bse_energies']]
        strengths = [data['ip_strengths'], data['bse_strengths']]
    limits = np.concatenate(energies)*HARTREE_TO_EV
    grid = np.linspace(max(0., float(limits.min())-1.5), float(limits.max())+1.5, 1601)
    width = .3  # Display FWHM in eV, not an electronic lifetime.
    spectra = [np.asarray(broaden_spectrum(e, f, grid, fwhm_ev=width))
               for e, f in zip(energies, strengths)]
    area_errors = [float(abs(np.trapezoid(y, grid)-f.sum())) for y, f in zip(spectra, strengths)]
    assert max(area_errors) < 1e-6

    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for label, color, e, f, y in zip(['Independent particle', 'Static TDA-BSE body'],
            ['#6b7280', '#1261a0'], energies, strengths, spectra):
        axes[0].plot(grid, y, color=color, label=label, lw=1.8)
        axes[1].vlines(e*HARTREE_TO_EV, 0., f, color=color, lw=1.6)
        axes[1].plot(e*HARTREE_TO_EV, f, 'o', color=color, label=label)
    axes[0].set_ylabel('Oscillator strength per cell / eV')
    axes[0].set_title(f'Gaussian display broadening: FWHM = {width:.2f} eV')
    axes[1].set_ylabel('Oscillator strength per cell')
    axes[1].set_title('Unbroadened transitions; no rescaling')
    for ax in axes:
        ax.set_xlabel('Excitation energy / eV')
        ax.set_xlim(grid[0], grid[-1]); ax.set_ylim(bottom=0.)
        ax.grid(alpha=.18); ax.legend(frameon=False, fontsize=9)
    fig.suptitle('3D H₂ array: PBE → G₀W₀ → static TDA-BSE body\n'
                 'Same QP energies and KS vertices; 2 × 1 × 1 k points; no direct q → 0 correction', fontsize=11)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUTPUT/('periodic_hydrogen_spectrum.'+suffix))
    plt.close(fig)
    with (OUTPUT/'spectrum.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['energy_ev', 'ip_strength_per_cell_per_ev', 'bse_body_strength_per_cell_per_ev'])
        writer.writerows(zip(grid, *spectra))
    validation = dict(complete=True, fwhm_ev=width, integrated_strength_errors=area_errors,
        results_sha256=hashlib.sha256((OUTPUT/'results.json').read_bytes()).hexdigest(),
        arrays_sha256=hashlib.sha256((OUTPUT/'arrays.npz').read_bytes()).hexdigest(),
        scope='Finite-model per-cell oscillator strengths, not a macroscopic absorption coefficient.')
    (OUTPUT/'plot_validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print('Saved periodic_hydrogen_spectrum.png/.pdf; area errors:', area_errors)


if __name__ == '__main__':
    plot()
