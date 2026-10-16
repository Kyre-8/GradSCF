"""Show a verified band case and its independent numerical discrepancy."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants


OUTPUT = Path('reproducibility/ad_showcase/PBC_material_bands')
HARTREE_TO_EV = physical_constants['Hartree energy in eV'][0]


def plot(case='silicon_pbe_pilot'):
    folder = OUTPUT/case
    path = folder/'native.json'
    native = json.loads(path.read_text())
    reference = json.loads((folder/'pyscf.json').read_text())
    assert native['passed'] and reference['passed']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference['native_sha256']
    with np.load(folder/'bands.npz') as stored:
        energy, distance = stored['energies_hartree'], stored['path_distance']
        zero = float(stored['energy_zero_hartree'])
    with np.load(folder/'pyscf_bands.npz') as stored:
        oracle = stored['energies_hartree']
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True,
                              gridspec_kw={'width_ratios': [2.3, 1.]})
    for band in range(energy.shape[1]):
        axes[0].plot(distance, (energy[:, band]-zero)*HARTREE_TO_EV,
                     color='#2166ac', linewidth=1.1, label='GradSCF' if band == 0 else None)
        axes[0].plot(distance, (oracle[:, band]-zero)*HARTREE_TO_EV,
                     color='#d95f02', linestyle='--', linewidth=.65,
                     label='Independent PySCF' if band == 0 else None)
    labels = ['Γ' if label == 'G' else label for label in native['tick_labels']]
    for axis in axes:
        for tick in native['tick_positions']:
            axis.axvline(tick, color='#aaaaaa', linewidth=.5)
        axis.set_xticks(native['tick_positions'], labels)
        axis.set_xlim(distance[0], distance[-1])
        axis.set_xlabel('Wavevector path')
    axes[0].axhline(0., color='#555555', linewidth=.6)
    axes[0].set(ylim=(-12, 10), ylabel='KS/GKS energy relative to declared zero (eV)',
                title='Native and independent bands')
    axes[0].legend(frameon=False, fontsize=9)
    frontier = np.asarray(reference['frontier_indices'])
    discrepancy = np.max(abs(energy[:, frontier]-oracle[:, frontier]), axis=1)*HARTREE_TO_EV*1000
    axes[1].plot(distance, discrepancy, color='#348741')
    axes[1].axhline(reference['gates']['frontier_bands_hartree']*HARTREE_TO_EV*1000,
                    color='black', linestyle=':', label='Numerical gate')
    axes[1].set(ylabel='Maximum frontier-band difference (meV)',
                title='Same-input numerical comparison')
    axes[1].legend(frameon=False, fontsize=8)
    options = native['settings']
    subtitle = (f"{native['material']} / {native['method'].upper()} / {options['basis']}\n"
        f"k={tuple(options['kmesh'])}, FFT={tuple(options['mesh'])}; "
        f"sampled path gap {native['path_gap_ev']:.3f} eV; discretization not yet converged")
    if native['physical_dimension'] < 3:
        subtitle += (f"\n{native['electrostatic_boundary']}, cell height {options['vacuum']:g} Å; "
                     'vacuum convergence not established')
    fig.suptitle(subtitle, fontsize=10)
    for suffix in ('png', 'pdf'):
        fig.savefig(folder/f'bands.{suffix}')
    plt.close(fig)


def compare_methods(material='silicon', methods=('lda', 'pbe', 'r2scan'), *, cases=None,
                    output_name=None):
    """Compare validated cases at the same geometry, basis, grid and k path."""
    records = []
    for method in methods:
        folder = OUTPUT/(f'{material}_{method}_pilot' if cases is None else cases[method])
        path = folder/'native.json'
        native = json.loads(path.read_text())
        reference = json.loads((folder/'pyscf.json').read_text())
        assert native['passed'] and reference['passed']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference['native_sha256']
        with np.load(folder/'bands.npz') as stored:
            data = {key: stored[key].copy() for key in stored.files}
        if records:
            previous = records[0][0]
            assert native['settings'] == previous['settings']
            assert native['lattice_angstrom'] == previous['lattice_angstrom']
            assert native['coordinates_angstrom'] == previous['coordinates_angstrom']
            assert native.get('electrostatic_dimension', 3) == previous.get('electrostatic_dimension', 3)
            np.testing.assert_array_equal(data['fractional_kpoints'], records[0][1]['fractional_kpoints'])
        records.append((native, data))
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(1, len(methods), figsize=(4*len(methods), 4.9),
        sharey=True, squeeze=False, constrained_layout=True)
    colors = ('#2166ac', '#b35806', '#348741', '#762a83')
    for index, (native, data) in enumerate(records):
        axis = axes[0, index]
        energy = (data['energies_hartree']-float(data['energy_zero_hartree']))*HARTREE_TO_EV
        axis.plot(data['path_distance'], energy, color=colors[index % len(colors)], linewidth=.9)
        for tick in native['tick_positions']:
            axis.axvline(tick, color='#aaaaaa', linewidth=.5)
        axis.axhline(0., color='#666666', linewidth=.6)
        axis.set_xticks(native['tick_positions'], ['Γ' if x == 'G' else x for x in native['tick_labels']])
        axis.set(xlim=(data['path_distance'][0], data['path_distance'][-1]), ylim=(-12, 10),
            xlabel='Wavevector path', title=f"{native['method'].upper()}\nSampled path gap {native['path_gap_ev']:.3f} eV")
    axes[0, 0].set_ylabel('Band energy relative to each declared reference (eV)')
    native = records[0][0]
    options = native['settings']
    fig.suptitle(f"{material}: fixed structure / {options['basis']} / {options['pseudo']}\n"
        f"k={tuple(options['kmesh'])}, FFT={tuple(options['mesh'])}; numerical cross-checks passed, discretization not converged",
        fontsize=10)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUTPUT/f'{output_name or material+"_methods"}.{suffix}')
    plt.close(fig)


if __name__ == '__main__':
    plot()
