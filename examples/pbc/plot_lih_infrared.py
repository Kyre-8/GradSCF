"""Plot raw-weight Gamma ionic dielectric loss and directional LO-TO limits."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path('reproducibility/ad_showcase/P05_lih_infrared')


def main(mesh=71):
    path = OUTPUT/f'fft{mesh}.json'
    data = json.loads(path.read_text())
    if not data['complete']:
        raise RuntimeError('The infrared calculation did not finish.')
    for source, expected in data['input_sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise RuntimeError('An upstream response file changed: '+source)
    with np.load(OUTPUT/f'fft{mesh}.npz') as archive:
        grid, epsilon, directions = archive['grid_cm1'], archive['epsilon'], archive['directions']
    labels = ['[100]', '[110]', '[111]']
    colors = ['#2166ac', '#b2182b', '#348741']
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for width_index, width in enumerate(data['widths_cm1']):
        for index, direction in enumerate(directions):
            response = np.einsum('i,wij,j->w', direction, epsilon[width_index], direction)
            style = '-' if width_index == 0 else '--'
            label = labels[index]+f', η = {width:g} cm⁻¹'
            axes[0].plot(grid, response.imag, style, color=colors[index], label=label, linewidth=1.3)
            axes[1].plot(grid, (-1/response).imag, style, color=colors[index], linewidth=1.3)
    axes[0].set_ylabel('Im [e · ε(ω) · e]')
    axes[1].set_ylabel('Im [−1 / (q̂ · ε(ω) · q̂)]')
    axes[0].set_title('Transverse ionic dielectric loss')
    axes[1].set_title('Longitudinal macroscopic loss')
    for ax in axes:
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_xlim(250, grid[-1])
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    verdict = 'raw gates passed' if data['passed'] else 'raw gates FAILED; diagnostic only'
    fig.suptitle(f'⁷Li¹H, PBE / GTH-Pade / SZV, k2³ / FFT{mesh}³\n'
                 f'{verdict}; harmonic Γ model, imposed resolution, no weight normalization', fontsize=11)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUTPUT/f'lih_infrared_fft{mesh}.{suffix}')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    transverse = np.asarray(data['transverse_cm1'])
    longitudinal = np.asarray(data['longitudinal_optical_cm1'])
    for index, label in enumerate(labels):
        axes[0].scatter(np.full(3, index)-.08, transverse, marker='_', s=160,
                        color='#666666', label='Transverse' if index == 0 else None)
        axes[0].scatter(np.full(3, index)+.08, longitudinal[index], marker='o', s=30,
                        color=colors[index], label=label+' directional')
    axes[0].set_xticks(range(3), labels)
    axes[0].set_ylabel('Optical frequency (cm⁻¹)')
    axes[0].set_title('Optical-subspace nonanalytic correction')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=.2)
    keys = list(data['gates'])
    ratios = [data['metrics'][key]/data['gates'][key] for key in keys]
    axes[1].barh(range(len(keys)), np.maximum(ratios, 1e-16),
                 color=['#348741' if value < 1 else '#b2182b' for value in ratios])
    axes[1].set_yticks(range(len(keys)), ['Raw acoustic', 'Raw ASR', 'Charge sum',
                                        'Mode sum', 'Optical LST', 'Resolvent', 'Grid integral', 'Spectral sum'])
    axes[1].axvline(1., color='black', linestyle='--', linewidth=1)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Measured error / predeclared threshold')
    axes[1].set_title('Raw defects and numerical checks')
    fig.suptitle('No acoustic or charge-neutrality projection; raw full-direction modes remain in JSON', fontsize=10)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUTPUT/f'lih_infrared_checks_fft{mesh}.{suffix}')
    plt.close(fig)


if __name__ == '__main__':
    main()

# Saved on jy_gpu_1 (2026-10-16), using the matched FFT71 result:
# lih_infrared_fft71.png/.pdf and lih_infrared_checks_fft71.png/.pdf.
# TO = 441.4103 cm^-1; highest directional LO = 1490.0793--1492.6685 cm^-1.
# All raw gates pass; no ASR, charge projection, intensity normalization or lifetime fit.
