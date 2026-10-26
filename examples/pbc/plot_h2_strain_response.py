"""Plot the verified local periodic H2 curvature and volume sensitivity."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants, h, c


OUTPUT = Path('reproducibility/ad_showcase/P07_periodic_strain')
data = json.loads((OUTPUT/'gradscf.json').read_text())
if not data['complete'] or not data['passed']:
    raise RuntimeError('Use a completed, validated strain calculation.')
plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 220})
fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
labels = ['H1 x', 'H1 y', 'H1 z', 'H2 x', 'H2 y', 'H2 z']
for ax, key, title in [(axes[0, 0], 'hessian_hartree_bohr2', 'Raw Cartesian curvature H'),
                        (axes[0, 1], 'strain_hessian_derivative', 'AD derivative dH/d ln V')]:
    matrix = np.asarray(data[key])
    limit = np.max(abs(matrix))
    picture = ax.imshow(matrix, cmap='coolwarm', vmin=-limit, vmax=limit)
    ax.set_xticks(range(6), labels, rotation=45)
    ax.set_yticks(range(6), labels)
    ax.set_title(title)
    fig.colorbar(picture, ax=ax, label='Eh / bohr²', shrink=.8)
rows = data['finite_differences']
steps = [row['log_volume_step'] for row in rows]
errors = [row['max_hessian_derivative_error'] for row in rows]
axes[1, 0].loglog(steps, errors, 'o-', color='#2166ac', label='Recomputed Hessian FD')
axes[1, 0].axhline(2e-7, color='black', linestyle='--', label='Declared gate')
axes[1, 0].set_xlabel('Central log-volume step')
axes[1, 0].set_ylabel('Maximum |AD − FD| (Eh / bohr²)')
axes[1, 0].set_title('Complete curvature derivative')
axes[1, 0].legend(frameon=False, fontsize=8)
axes[1, 0].grid(alpha=.2, which='both')
factor = physical_constants['Hartree energy'][0]/(h*c*100)
rows = data['frequency_finite_differences']
axes[1, 1].semilogx([row['step'] for row in rows],
    [row['error']*factor for row in rows], 'o-', color='#348741')
axes[1, 1].set_xlabel('Central log-volume step')
axes[1, 1].set_ylabel('Frequency-slope |AD − FD| (cm⁻¹)')
axes[1, 1].set_title('Isolated local stretch, full root tracking')
axes[1, 1].text(.04, .95,
    f"ω = {data['selected_frequency_hartree']*factor:.3f} cm⁻¹\n"
    f"dω/d ln V = {data['frequency_derivative_log_volume']*factor:.3f} cm⁻¹\n"
    f"Local γ = {data['local_mode_gruneisen']:.6f}", transform=axes[1, 1].transAxes,
    va='top', fontsize=9)
axes[1, 1].grid(alpha=.2)
fig.suptitle('Periodic H₂: RHF / Γ / FFT9³, 6-bohr cubic cell\n'
             'Local finite-model response; fixed fractional atoms, no ASR or thermal interpretation', fontsize=11)
for extension in ['png', 'pdf']:
    fig.savefig(OUTPUT/f'h2_strain_response.{extension}')
plt.close(fig)

# Saved on jy_gpu_1 (2026-10-16): h2_strain_response.png and .pdf.
# Isolated local mode: omega=4739.289 cm^-1, d omega/dln(V)=-3772.100 cm^-1.
# Local gamma=0.795921; the coarse model's three negative raw curvatures remain in JSON.
