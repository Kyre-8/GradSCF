"""P08 density/potential/response maps, energy Taylor curves and grid checks."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path('reproducibility/ad_showcase/P08_ofdft_parameter_response')


def main():
    report = json.loads((OUTPUT/'gradscf.json').read_text())
    reference = json.loads((OUTPUT/'independent.json').read_text())
    assert report['complete'] and reference['passed']
    main_case = next(case for case in report['cases'] if 'taylor_energy_coefficients' in case)
    mesh = tuple(main_case['mesh'])
    arrays = np.load(OUTPUT/f'mesh_{mesh[0]}.npz')
    independent = np.load(OUTPUT/'independent_response.npz')
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    length = report['length_bohr']

    def field(axis, values, title, label, *, signed=False):
        plane = np.asarray(values).reshape(mesh)[:, :, 0]
        # Duplicate periodic boundary samples for display only. No numerical
        # upsampling or altered grid is used by the response calculation.
        shown = np.pad(plane, ((0, 1), (0, 1)), mode='wrap').T
        coordinates = np.linspace(0., length, mesh[0]+1)
        options = dict(cmap='RdBu_r', vmin=-np.max(abs(plane)), vmax=np.max(abs(plane))) if signed else dict(cmap='viridis')
        image = axis.pcolormesh(coordinates, coordinates, shown, shading='nearest', **options)
        colorbar = fig.colorbar(image, ax=axis, shrink=.85)
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        colorbar.ax.ticklabel_format(style='sci', scilimits=(-2, 2))
        axis.set(title=title, xlabel='x / bohr', ylabel='y / bohr', xlim=(0, length), ylim=(0, length))
        axis.set_aspect('equal')

    field(axes[0, 0], arrays['density'], 'Stationary density, N = 2', 'electron bohr⁻³')
    field(axes[0, 1], arrays['external_potential'], 'Manufactured external potential', 'Hartree', signed=True)
    field(axes[0, 2], arrays['density_jacobian'][:, 0], 'External-amplitude response ∂n/∂u', 'electron bohr⁻³ Eh⁻¹', signed=True)
    field(axes[1, 0], arrays['density_jacobian'][:, 1], 'TF-weight response ∂n/∂a', 'electron bohr⁻³', signed=True)
    field(axes[1, 1], arrays['density_jacobian'][:, 2], 'vW-weight response ∂n/∂λ', 'electron bohr⁻³', signed=True)
    field(axes[1, 2], arrays['density_jacobian'][:, 0]-independent['density_jacobian'][:, 0],
          'Response difference: native − independent', 'electron bohr⁻³ Eh⁻¹', signed=True)

    path = main_case['parameter_path']
    energy0 = main_case['total_energy_hartree']
    t = np.array([row['t'] for row in path])
    energy = np.array([row['energy_hartree'] for row in path])
    coefficient1, coefficient2 = main_case['taylor_energy_coefficients']
    fine = np.linspace(-.2, .2, 161)
    axes[2, 0].plot(fine, 1000*fine*coefficient1, '--', label='Linear Taylor')
    axes[2, 0].plot(fine, 1000*(fine*coefficient1+fine**2*coefficient2), '-', label='Quadratic Taylor')
    axes[2, 0].plot(t, 1000*(energy-energy0), 'o', fillstyle='none', color='black', label='Reoptimized density')
    axes[2, 0].set(title='Joint external/KEDF parameter path', xlabel='Dimensionless path t', ylabel='E(t) − E(0) / mEh')
    axes[2, 0].legend(fontsize=8)
    for key, label, marker in [('linear_energy_hartree', 'Linear', 'o'), ('quadratic_energy_hartree', 'Quadratic', 's')]:
        steps = [.05, .1, .2]
        errors = [max(abs(row['energy_hartree']-row[key]) for row in path if np.isclose(abs(row['t']), step)) for step in steps]
        axes[2, 1].loglog(steps, errors, marker+'-', label=label)
    axes[2, 1].set(title='Taylor truncation, both path signs', xlabel='|t|', ylabel='Maximum energy error / Eh')
    axes[2, 1].legend(fontsize=8)
    sizes = [case['mesh'][0] for case in reference['cases']]
    first = axes[2, 2].semilogy(sizes, [case['continuum_energy_error'] for case in reference['cases']], 'o-', color='#0072B2', label='Energy vs continuum')
    other = axes[2, 2].twinx()
    second = other.semilogy(sizes[1:], reference['relative_curvature_grid_increments'], 's--', color='#D55E00', label='Response curvature increment')
    third = other.semilogy(sizes[1:], [max(row['relative_l2_by_parameter']) for row in reference['density_map_grid_increments']], '^:', color='#009E73', label='Max density-map L2 increment')
    axes[2, 2].set(title='Finite-grid error and response refinement', xlabel='x/y FFT grid size (z size = 3)', ylabel='Energy error / Eh')
    other.set_ylabel('Relative response increment')
    axes[2, 2].legend(first+second+third, [line.get_label() for line in first+second+third], fontsize=7, loc='upper right')
    for axis in axes[2]:
        axis.grid(alpha=.2)
    fig.suptitle('Fixed-N TF + λ vW + Hartree: explicit external KEDF parameters\n'
                 'Periodic 5-bohr box · N = 2 · z-uniform analytic model · 9×9×3 response maps, z = 0', fontsize=14)
    for extension in ['png', 'pdf']:
        fig.savefig(OUTPUT/f'density_parameter_response.{extension}', dpi=170)
    plt.close(fig)
    with (OUTPUT/'density_response_maps.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['x_bohr', 'y_bohr', 'z_bohr', 'density_e_bohr3', 'external_Eh',
                         'dn_du_e_bohr3_Eh', 'dn_da_e_bohr3', 'dn_dlambda_e_bohr3'])
        for point, density, potential, response in zip(arrays['coordinates_bohr'], arrays['density'], arrays['external_potential'], arrays['density_jacobian']):
            writer.writerow([*point, density, potential, *response])
    (OUTPUT/'plot_settings.json').write_text(json.dumps(dict(
        slice_z_bohr=0., mesh=list(mesh), normalization='Raw density/response values; fixed integrated N',
        display='Nearest cell values with duplicated periodic boundary samples; no numerical grid refinement',
        model_scope=report['spatial_scope']), indent=2)+'\n')
    print('Saved P08 density/energy/parameter-response maps and grid figure', flush=True)


if __name__ == '__main__':
    main()
