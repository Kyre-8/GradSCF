"""V06 frequency response, damping dependence and independent numerical gates."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants


OUTPUT = Path('reproducibility/ad_showcase/V06_water_dynamic_raman')


def main():
    report = json.loads((OUTPUT/'gradscf.json').read_text())
    independent = json.loads((OUTPUT/'pyscf.json').read_text())
    assert report['complete'] and independent['passed']
    data = np.load(OUTPUT/'response.npz')
    energy = data['frequencies_hartree']*physical_constants['Hartree energy in eV'][0]
    dampings = data['dampings_hartree']
    alpha = np.trace(data['polarizability'], axis1=-2, axis2=-1)/3
    activities = data['activities_angstrom4_amu']
    colors = ['#0072B2', '#D55E00', '#009E73']
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    for index, (eta, color) in enumerate(zip(dampings, colors)):
        label = f'η = {eta:g} Eh'
        axes[0, 0].plot(energy, alpha[index].real, color=color, label=label)
        axes[0, 1].plot(energy, alpha[index].imag, color=color, label=label)
        axes[1, 0].semilogy(energy, activities[index, :, 1], color=color, label=label)
    axes[0, 0].set(title='Dispersive electronic response', ylabel='Re Tr(α)/3 / bohr³')
    axes[0, 1].set(title='Absorptive electronic response', ylabel='Im Tr(α)/3 / bohr³')
    for mode, color in enumerate(colors):
        axes[0, 2].semilogy(energy, activities[1, :, mode], color=color,
                            label=f'Mode {mode+1}: {report["mode_frequencies_cm1"][mode]:.0f} cm⁻¹')
    axes[0, 2].set(title='Mode activities at η = 0.01 Eh', ylabel='Placzek activity / Å⁴ amu⁻¹')
    axes[1, 0].set(title='Symmetric stretch: damping dependence', ylabel='Placzek activity / Å⁴ amu⁻¹')
    for axis in [*axes[0], axes[1, 0]]:
        axis.set_xlabel('Incident photon energy / eV')
        axis.legend(fontsize=8)
        axis.grid(alpha=.18)
    for row, raman, color in zip(report['frequency_grid_checks'], independent['raman_frequency_grid_checks'], colors):
        integrals = row['integrated_imaginary_isotropic_alpha']
        error = [abs(integrals[1]-integrals[0])/abs(integrals[1]),
                 abs(integrals[2]-integrals[1])/abs(integrals[2])]
        axes[1, 1].semilogy(row['samples'][1:], error, 'o-', color=color, label=f'α, η={row["damping_hartree"]:g}')
        raman_area = np.asarray(raman['integrated_activities'])
        raman_error = np.max(abs(np.diff(raman_area, axis=0))/abs(raman_area[1:]), axis=1)
        axes[1, 1].semilogy(row['samples'][1:], raman_error, 's--', color=color, label=f'Raman, η={row["damping_hartree"]:g}')
    axes[1, 1].axhline(2e-4, color='gray', linestyle=':', label='Area convergence gate')
    axes[1, 1].set(title='Frequency-grid refinement', xlabel='Number of frequency samples',
                   ylabel='Relative integrated-area change')
    axes[1, 1].legend(fontsize=7, ncol=2)
    checks = independent['mode_derivative_checks']
    for mode, color in enumerate(colors, 1):
        errors = [max(row['maximum_derivative_error'] for row in checks
                       if row['mode'] == mode and row['max_cartesian_step_bohr'] == step)
                  for step in [3e-4, 1e-4]]
        axes[1, 2].loglog([3e-4, 1e-4], errors, 'o-', color=color, label=f'Mode {mode}')
    axes[1, 2].set(title='Independent displaced PySCF response', xlabel='Maximum Cartesian displacement / bohr',
                   ylabel='Max |AD − FD| / [bohr² / √mₑ]')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].invert_xaxis()
    fig.suptitle('Water RHF/6-31G*: complete TDHF space and full geometry response\n'
                 'Harmonic frequency-dependent Placzek model · phenomenological damping · no component normalization', fontsize=15)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUTPUT/f'dynamic_raman.{suffix}', dpi=170)
    plt.close(fig)
    with (OUTPUT/'frequency_response.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['damping_hartree', 'photon_energy_ev', 'isotropic_alpha_real_bohr3',
                         'isotropic_alpha_imag_bohr3', 'mode1_activity_A4_amu', 'mode2_activity_A4_amu', 'mode3_activity_A4_amu'])
        for row, eta in enumerate(dampings):
            for index, photon in enumerate(energy):
                writer.writerow([eta, photon, alpha[row, index].real, alpha[row, index].imag, *activities[row, index]])
    (OUTPUT/'plot_settings.json').write_text(json.dumps(dict(
        figure='2x3 complete-space frequency response and validation',
        frequency_units='eV', damping_units='Hartree', polarizability_units='bohr^3',
        activity_units='angstrom^4/amu', components_normalized=False,
        approximation=report['approximation']), indent=2)+'\n')
    print('Saved V06 frequency/damping/mode/validation plots', flush=True)


if __name__ == '__main__':
    main()
