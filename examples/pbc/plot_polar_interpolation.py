"""Show reciprocal-cutoff, interpolation and independent derivative checks."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path('reproducibility/ad_showcase/P05_polar_interpolation')


def main():
    native_path = OUTPUT/'gradscf.json'
    native = json.loads(native_path.read_text())
    oracle = json.loads((OUTPUT/'phonopy.json').read_text())
    assert native['passed'] and oracle['passed']
    assert hashlib.sha256(native_path.read_bytes()).hexdigest() == oracle['native_json_sha256']
    assert hashlib.sha256((OUTPUT/'native.npz').read_bytes()).hexdigest() == oracle['native_arrays_sha256']
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.2), constrained_layout=True)
    rows = native['cutoff_records']
    for key, label, marker in [('error', 'Matrix vs cutoff 6', 'o'),
                               ('periodic_error', 'Reciprocal translation', 's'),
                               ('self_inverse_imaginary', 'Self-inverse Im part', '^')]:
        axes[0].semilogy([r['cutoff'] for r in rows],
            np.maximum([r[key] for r in rows], 1e-18), marker+'-', label=label)
    axes[0].set_xlabel('Reciprocal integer cutoff')
    axes[0].set_ylabel('Maximum error (Eh / bohr²)')
    axes[0].set_title('(a) Fixed reciprocal-sum convergence')
    axes[0].set_xticks([1, 2, 3, 4, 5])
    axes[0].legend(frameon=False, fontsize=8)

    rows = native['qmesh_records']
    axes[1].semilogy([r['qmesh'] for r in rows],
                    [r['changed_smearing_error'] for r in rows], 'o-', color='#2166ac')
    axes[1].set_xlabel('q mesh per axis')
    axes[1].set_ylabel('Off-grid matrix error (Eh / bohr²)')
    axes[1].set_title('(b) Change decomposition width')
    axes[1].set_xticks([3, 5, 7], ['3³', '5³', '7³'])
    axes[1].text(.04, .07, 'Same sampled total IFCs\nΛ: 0.35 → 0.45 bohr⁻¹',
                 transform=axes[1].transAxes, fontsize=9)

    rows = oracle['finite_differences']
    steps = np.array([r['step'] for r in rows])
    errors = np.array([r['derivative_error'] for r in rows])
    axes[2].loglog(steps, errors, 'o-', color='#b2182b', label='AD vs Phonopy FD')
    axes[2].loglog(steps, errors[0]*(steps/steps[0])**2, '--', color='#777777', label='h² guide')
    axes[2].axhline(oracle['gates']['derivative'], color='black', linestyle=':', label='Final-step gate')
    axes[2].set_xlabel('Directional step h (fractional q)')
    axes[2].set_ylabel('Derivative error (Eh / bohr²)')
    axes[2].set_title('(c) Independent q-response check')
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle('Manufactured two-atom polar IFC model; no ASR projection or material fit\n'
        'GradSCF vs pinned Phonopy 2.44 C kernel; matrix error ≤ 5.56 × 10⁻¹⁷ Eh/bohr²', fontsize=11)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUTPUT/('polar_interpolation_checks.'+suffix))
    plt.close(fig)


if __name__ == '__main__':
    main()

# Saved on jy_gpu_1 (2026-10-16): polar_interpolation_checks.png/.pdf.
# Inputs are the verified native and independent Phonopy records above.
# No material calculation, ASR correction or interpolation fit is introduced.
