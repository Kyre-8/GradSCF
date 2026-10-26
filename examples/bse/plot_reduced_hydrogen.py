"""Plot saved line/sheet polarizabilities from the finite-model H2 examples."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from reduced_hydrogen import OUTPUT


def plot():
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for dimension, ax in zip((1, 2), axes):
        folder = OUTPUT/f'dimension_{dimension}'
        report = json.loads((folder/'results.json').read_text())
        assert report['complete'] and not report['gw_fc']
        with np.load(folder/'arrays.npz') as data:
            grid = data['frequency_ev']
            for key, label, color in [('ip_alpha', 'Independent particle', '#6b7280'),
                                     ('bse_alpha', 'Static TDA-BSE', '#1261a0')]:
                ax.plot(grid, data[key][:, 0, 0].imag, label=label, color=color, lw=1.8)
        kind, unit = ('line', r'bohr$^2$') if dimension == 1 else ('sheet', 'bohr')
        ax.set_title(f'H₂ {"wire" if dimension == 1 else "layer"}; η = {report["broadening_eta_ev"]:.2f} eV')
        ax.set_ylabel(f'Im α$_{{xx}}$ ({kind}) / {unit}')
        ax.set_xlabel('Photon energy / eV')
        ax.set_xlim(0., grid[-1]); ax.set_ylim(bottom=0.)
        ax.grid(alpha=.18); ax.legend(frameon=False, fontsize=9)
    fig.suptitle('LDA → G₀W₀ → static TDA-BSE: finite Coulomb model\n'
                 'Same QP energies and KS vertices; finite-part gauge; no q-cell integration', fontsize=11)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUTPUT/f'reduced_hydrogen_polarizability.{suffix}')
    plt.close(fig)
    print('Saved reduced_hydrogen_polarizability.png/.pdf')


if __name__ == '__main__':
    plot()
