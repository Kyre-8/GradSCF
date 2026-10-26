"""Plot raw Born tensors and independent displacement checks, without projection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('reproducibility/ad_showcase/P04_lih_born')
plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def save(figure, name):
    figure.savefig(ROOT/(name+'.png'), dpi=180, bbox_inches='tight')
    figure.savefig(ROOT/(name+'.pdf'), bbox_inches='tight')
    plt.close(figure)


def main(native_name='gradscf', reference_name='pyscf', prefix=''):
    native = json.loads((ROOT/(native_name+'.json')).read_text())
    reference = None if reference_name is None else json.loads((ROOT/(reference_name+'.json')).read_text())
    assert native['full_tensor'] and native['validation_passed']
    assert reference is None or reference['validation_passed']
    tensor = np.asarray(native['born_charge_e'])
    charge_sum = np.asarray(native['raw_acoustic_charge_sum_e'])
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    scale = np.max(abs(tensor))
    for axis, matrix, title in zip(axes, [tensor[0], tensor[1], charge_sum], ['Li', 'H', 'Raw sum over atoms']):
        bound = scale if title != 'Raw sum over atoms' else np.max(abs(matrix))
        image = axis.imshow(matrix, cmap='RdBu_r', vmin=-bound, vmax=bound)
        for i in range(3):
            for j in range(3):
                label = f'{matrix[i,j]:.6f}' if abs(matrix[i,j]) > .01 else f'{matrix[i,j]:.2e}'
                axis.text(j, i, label, ha='center', va='center', fontsize=8,
                          color='white' if abs(matrix[i,j]) > .55*bound else 'black')
        axis.set(title=title, xlabel='Displacement direction β', ylabel='Polarization direction α',
                 xticks=range(3), yticks=range(3), xticklabels=list('xyz'), yticklabels=list('xyz'))
        figure.colorbar(image, ax=axis, shrink=.75, label='Charge / e')
    settings = native['settings']
    mesh = '×'.join(map(str, settings['mesh']))
    kmesh = '×'.join(map(str, settings['kmesh']))
    figure.suptitle(f"LiH · zero-field Berry Born charges · a = {settings['a_bohr']:.8f} bohr\n"
                   f"{settings['xc'].upper()} / {settings['pseudo']} / {settings['basis']} · FFT {mesh} / k {kmesh} · no projection", fontsize=11)
    save(figure, prefix+'born_tensors')

    datasets = [(native['finite_differences'], 'Rebuilt GradSCF SCF', 2e-5)]
    if reference is not None:
        datasets.append((reference['finite_differences'], 'Independent PySCF SCF + AO links', 5e-5))
    figure, axes = plt.subplots(1, len(datasets), figsize=(5.4*len(datasets), 4.2),
                                constrained_layout=True, sharey=True, squeeze=False)
    colors = plt.cm.tab10(np.arange(6))
    for panel, (rows, title, gate) in zip(axes.ravel(), datasets):
        for index, (atom, coordinate) in enumerate((a, i) for a in range(2) for i in range(3)):
            selected = sorted((row for row in rows if row['atom'] == atom and row['axis'] == coordinate), key=lambda row:row['step_bohr'])
            panel.loglog([row['step_bohr'] for row in selected], [row['maximum_charge_error_e'] for row in selected],
                         'o-', color=colors[index], label=f"{['Li','H'][atom]} {['x','y','z'][coordinate]}")
        panel.axhline(gate, color='.4', linestyle=':', label='Declared gate')
        panel.set(title=title, xlabel='Cartesian displacement step / bohr', ylabel=r'max $|Z^*_{\mathrm{AD}}-Z^*_{\mathrm{FD}}|$ / e')
        panel.grid(alpha=.2, which='both')
        panel.legend(fontsize=8, ncol=2)
    figure.suptitle('Six independent displacement directions · continuously anchored Berry branches', fontsize=12)
    save(figure, prefix+'born_displacement_validation')


if __name__ == '__main__':
    main()
