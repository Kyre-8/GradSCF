"""Plot fixed-cell electronic-temperature curvature checks, without smoothing."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path('reproducibility/ad_showcase/P07_periodic_temperature')
data = json.loads((OUTPUT/'gradscf.json').read_text())
if not data['complete'] or not data['passed']:
    raise RuntimeError('Use a completed free-energy response calculation.')
matrix = np.asarray(data['temperature_derivative_bohr_minus2'])
rows = data['finite_differences']
steps = [row['step_kbt_hartree'] for row in rows]
errors = [row['max_error_bohr_minus2'] for row in rows]
plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 220})
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
limit = np.max(abs(matrix))
picture = axes[0].imshow(matrix, cmap='coolwarm', vmin=-limit, vmax=limit)
labels = ['H1 x', 'H1 y', 'H1 z', 'H2 x', 'H2 y', 'H2 z']
axes[0].set_xticks(range(6), labels, rotation=45)
axes[0].set_yticks(range(6), labels)
axes[0].set_title('dH / d(kBT), fixed geometry and cell')
fig.colorbar(picture, ax=axes[0], label='bohr⁻²', shrink=.8)
axes[1].loglog(steps, errors, 'o-', color='#2166ac', label='Recomputed free-energy Hessian FD')
axes[1].axhline(2e-6, color='black', linestyle='--', label='Declared final-step gate')
axes[1].set_xlabel('Central kBT step (Eh)')
axes[1].set_ylabel('Maximum |AD − FD| (bohr⁻²)')
axes[1].set_title('Fixed-N electronic-temperature derivative')
axes[1].grid(alpha=.2, which='both')
axes[1].legend(frameon=False, fontsize=8)
fig.suptitle('Periodic H₂, RHF / Γ / FFT9³, kBT = 0.05 Eh\n'
             'Mermin free energy; no volume expansion or nuclear anharmonicity', fontsize=11)
for extension in ['png', 'pdf']:
    fig.savefig(OUTPUT/f'h2_temperature_response.{extension}')
plt.close(fig)

# Saved on jy_gpu_1 (2026-10-16): h2_temperature_response.png and .pdf.
# Final H_kBT FD error=1.73946e-7 bohr^-2 at a 1e-4 Eh temperature step.
# Fixed-N electronic-temperature response only; no expansion or anharmonicity.
