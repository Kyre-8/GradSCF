"""Plot the raw clamped-ion tensor and separately reconverged field checks."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('reproducibility/ad_showcase/P04_lih_dielectric')
plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def main(linear_name='gradscf', finite_name='finite_field', output_name='dielectric_validation'):
    linear = json.loads((ROOT/(linear_name+'.json')).read_text())
    finite = json.loads((ROOT/(finite_name+'.json')).read_text())
    assert linear['full_tensor'] and finite['complete'] and finite['validation_passed']
    matrix = np.asarray(linear['epsilon_infinity'])
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), constrained_layout=True)
    image = axes[0].imshow(matrix, cmap='RdBu_r', vmin=-abs(matrix).max(), vmax=abs(matrix).max())
    for i in range(3):
        for j in range(3):
            label = f'{matrix[i,j]:.6f}' if i == j else f'{matrix[i,j]:.3e}'
            axes[0].text(j, i, label, ha='center', va='center', fontsize=9,
                         color='white' if i == j else 'black')
    axes[0].set(title=r'Raw $\epsilon_\infty = I+4\pi\chi$', xlabel='Electric-field direction β',
                ylabel='Polarization direction α', xticks=range(3), yticks=range(3),
                xticklabels=list('xyz'), yticklabels=list('xyz'))
    figure.colorbar(image, ax=axes[0], shrink=.75, label='Dimensionless')
    for axis, label in enumerate('xyz'):
        rows = sorted((r for r in finite['finite_differences'] if r['axis'] == axis),
                      key=lambda r:r['step_au'])
        axes[1].loglog([r['step_au'] for r in rows], [r['maximum_error'] for r in rows],
                      'o-', label=f'Field along {label}')
    axes[1].axhline(finite['gate_chi_absolute'], color='.4', linestyle=':', label='Absolute gate')
    axes[1].set(title='Two-step finite-field reconvergence',
                xlabel=r'Field step / $E_h\,(e\,a_0)^{-1}$',
                ylabel=r'max $|\chi_{\mathrm{AD}}-\chi_{\mathrm{FD}}|$ / a.u.')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=.2, which='both')
    settings = linear['settings']
    figure.suptitle(f"LiH · clamped-ion static Berry electric response · a = {settings['a_bohr']:.8f} bohr\n"
        f"PBE / GTH-Pade / SZV · FFT{settings['mesh'][0]} / k{settings['kmesh'][0]} · raw finite model, no symmetry projection", fontsize=11)
    for extension in ('png', 'pdf'):
        figure.savefig(ROOT/(output_name+'.'+extension), dpi=180, bbox_inches='tight')
    plt.close(figure)


if __name__ == '__main__':
    main()
