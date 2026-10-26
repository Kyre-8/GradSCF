"""Plot parameter sensitivity and retain the failed coarse-grid comparisons."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants, h, c


OUTPUT = Path('reproducibility/ad_showcase/P07_periodic_xc')


def main():
    path = OUTPUT/'gradscf.json'
    native = json.loads(path.read_text())
    reference = json.loads((OUTPUT/'pyscf.json').read_text())
    assert native['passed'] and reference['passed']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference['native_sha256']
    factor = physical_constants['Hartree energy'][0]/(h*c*100)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), constrained_layout=True)
    rows = native['sensitivity_map']
    alpha = np.array([row['alpha'] for row in rows])
    eigenvalues = np.array([row['raw_dynamical_eigenvalues'] for row in rows])
    frequencies = np.sign(eigenvalues)*np.sqrt(abs(eigenvalues))*factor
    for mode in range(6):
        axes[0, 0].plot(alpha, frequencies[:, mode], 'o-',
            color='#2166ac' if mode == 5 else '#999999',
            label='Tracked stretch' if mode == 5 else ('Other raw modes' if mode == 0 else None))
    tangent = (native['selected_frequency_hartree']+
               (alpha-1)*native['frequency_alpha_derivative'])*factor
    axes[0, 0].plot(alpha, tangent, '--', color='#b2182b', label='Local AD tangent')
    axes[0, 0].axhline(0., color='black', linewidth=.7)
    axes[0, 0].set(xlabel='Dirac-exchange scale α', ylabel='Raw signed frequency (cm⁻¹)',
                   title='(a) Fixed-cell local curvature spectrum')
    axes[0, 0].legend(frameon=False, fontsize=8)

    matrix = np.array(native['hessian_alpha_derivative'])
    limit = np.max(abs(matrix))
    shown = axes[0, 1].imshow(matrix, cmap='coolwarm', vmin=-limit, vmax=limit)
    labels = ['H1 x', 'H1 y', 'H1 z', 'H2 x', 'H2 y', 'H2 z']
    axes[0, 1].set_xticks(range(6), labels, rotation=45)
    axes[0, 1].set_yticks(range(6), labels)
    axes[0, 1].set_title('(b) AD force-constant derivative ∂H/∂α')
    fig.colorbar(shown, ax=axes[0, 1], shrink=.85, label='Eh / bohr²')

    rows = native['finite_differences']
    axes[1, 0].loglog([r['parameter_step'] for r in rows],
        [r['max_error_hartree_bohr2'] for r in rows], 'o-', label='Native Hessian differences')
    rows = reference['parameter_checks']
    axes[1, 0].loglog([r['parameter_step'] for r in rows],
        [r['hessian_derivative_error'] for r in rows], 's-', label='Independent PySCF gradient FD')
    axes[1, 0].axhline(reference['gates']['hessian_alpha_derivative_hartree_bohr2'],
                      color='black', linestyle=':', label='Independent final-step gate')
    axes[1, 0].set(xlabel='Exchange-parameter step', ylabel='Maximum derivative error (Eh / bohr²)',
                   title='(c) Complete mixed-response checks')
    axes[1, 0].legend(frameon=False, fontsize=8)

    rows = json.loads((OUTPUT/'forward_mesh_diagnostics.json').read_text())
    meshes = [r['mesh'] for r in rows]+[native['mesh'][0]]
    errors = [abs(r['difference']) for r in rows]+[reference['energy_error_hartree']]
    axes[1, 1].semilogy(meshes, errors, 'o-', color='#348741')
    axes[1, 1].axhline(reference['gates']['energy_hartree'], color='black', linestyle=':')
    axes[1, 1].set(xlabel='FFT points per axis', ylabel='Native/PySCF energy difference (Eh)',
                   title='(d) Retained discretization diagnostics')
    axes[1, 1].text(.04, .07, 'FFT31 passes energy but fails curvature.\n'
                    'Derivative accuracy needs its own gate.', transform=axes[1, 1].transAxes, fontsize=9)
    for axis in (axes[0, 0], axes[1, 0], axes[1, 1]):
        axis.grid(alpha=.2)
    fig.suptitle(f"Periodic H₂, scaled Dirac exchange / Γ / FFT{native['mesh'][0]}³\n"
                 'Fixed unoptimized geometry; raw modes retained, no ASR or stable-bulk interpretation', fontsize=11)
    for extension in ('png', 'pdf'):
        fig.savefig(OUTPUT/f'h2_xc_response.{extension}')
    plt.close(fig)


if __name__ == '__main__':
    main()
