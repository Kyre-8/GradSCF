"""Plot measured OFDFT costs, independent errors and the separate KS comparison."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path('reproducibility/ad_showcase/P08_ofdft_parameter_response/large_system_scaling')


def main():
    cases = [json.loads(path.read_text()) for path in OUTPUT.glob('al_[0-9].json')]
    cases = sorted((row for row in cases if row['complete']), key=lambda row: row['atom_count'])
    resources = {row['repeat']: json.loads((OUTPUT/f"native_{row['repeat']}_resources.json").read_text())
                 for row in cases}
    cases = [row for row in cases if resources[row['repeat']]['accepted']
             and resources[row['repeat']].get('gpu_index') == 6
             and resources[row['repeat']].get('hwloc_thissystem') == '0']
    assert len(cases) >= 2, 'Scaling requires at least two accepted GPU6 sizes under the same CPU/HWLOC policy'
    references = {row['repeat']: json.loads((OUTPUT/f"al_{row['repeat']}_reference.json").read_text())
                  for row in cases if (OUTPUT/f"al_{row['repeat']}_reference.json").exists()}
    references = {key: value for key, value in references.items() if value['passed']}
    device = f"GPU{resources[cases[0]['repeat']]['gpu_index']}"
    atoms = np.array([row['atom_count'] for row in cases])
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    keys = ['forward', 'jvp_including_forward', 'vjp_including_forward', 'hvp_including_forward']
    labels = ['Forward', 'Density JVP + forward', 'Density VJP + forward', 'Energy HVP + forward']
    for key, label, color in zip(keys, labels, colors):
        axes[0, 0].loglog(atoms, [row['timings'][key]['warm_median_seconds'] for row in cases],
                          'o-', label=label, color=color)
    axes[0, 0].set(title='Synchronized warm executions', xlabel='Al atoms', ylabel='Time / s')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].loglog(atoms, [row['input_preparation_seconds'] for row in cases],
                      's-', label='Potential + Ewald + grid setup', color=colors[1])
    axes[0, 1].loglog(atoms, [row['input_preparation_seconds']+
        sum(row['timings']['forward'][key] for key in ['lower_compile_seconds', 'first_execution_seconds'])
        for row in cases], 'o-', label='Setup + compile + first forward', color=colors[0])
    axes[0, 1].set(title='First-use cost includes input assembly', xlabel='Al atoms', ylabel='Time / s')
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].loglog(atoms, [resources[row['repeat']]['peak_rss_kib']/1024**2 for row in cases],
                      'o-', label='Process peak RSS', color=colors[0])
    axes[1, 0].loglog(atoms, [resources[row['repeat']]['sampled_peak_gpu_mib']/1024 for row in cases],
                      's-', label=f'{device} memory (1 s samples)', color=colors[1])
    axes[1, 0].set(title='Memory includes compilation and all AD operations',
                   xlabel='Al atoms', ylabel='Memory / GiB')
    axes[1, 0].legend(fontsize=8)
    selected = [row for row in cases if row['repeat'] in references]
    if selected:
        axis_atoms = [row['atom_count'] for row in selected]
        axes[1, 1].loglog(axis_atoms, [references[row['repeat']]['density_relative_l2_error'] for row in selected],
                          'o-', label='Density: GradSCF vs independent', color=colors[0])
        for index, style in enumerate(('s-', '^--', 'd:')):
            step = references[selected[0]['repeat']]['response_finite_differences'][index]['step_hartree']
            axes[1, 1].loglog(axis_atoms, [references[row['repeat']]['response_finite_differences'][index]['density_jvp_relative_l2_error']
                for row in selected], style, label=f'JVP FD step {step:g} Ha', color=colors[index+1])
        axes[1, 1].legend(fontsize=7)
    axes[1, 1].set(title='Same-grid independent DFTpy + NumPy WGC',
                   xlabel='Al atoms', ylabel='Relative L² error')
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.set_xticks(atoms, labels=[str(n) for n in atoms])
    fig.suptitle('Al WGC OFDFT: fixed 17³ points per primitive cell\n'
                 f'Float64 · {device} native / CPU independent reference · fixed ions; no force claim', fontsize=12)
    for extension in ('png', 'pdf'):
        fig.savefig(OUTPUT/f'scaling.{extension}', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    fd_rows = []
    for row in selected:
        differences = references[row['repeat']]['response_finite_differences']
        steps = [item['step_hartree'] for item in differences]
        axes[0].loglog(steps, [item['density_jvp_relative_l2_error'] for item in differences],
                       'o-', label=f"{row['atom_count']} atoms")
        axes[1].loglog(steps, [item['hvp_direction_error_per_atom'] for item in differences],
                       's-', label=f"{row['atom_count']} atoms")
        fd_rows.extend(dict(atom_count=row['atom_count'], **item) for item in differences)
    axes[0].set(title='Independent density-response finite differences', xlabel='Potential step / Ha', ylabel='Relative L² error')
    axes[1].set(title='Independent energy-curvature finite differences', xlabel='Potential step / Ha', ylabel='Absolute error / (Ha⁻¹ atom⁻¹)')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    for extension in ('png', 'pdf'):
        fig.savefig(OUTPUT/f'response_fd_steps.{extension}', dpi=170)
    plt.close(fig)
    (OUTPUT/'response_fd_all_steps.json').write_text(json.dumps(fd_rows, indent=2)+'\n')
    with (OUTPUT/'scaling.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['atoms', 'grid_points', 'input_seconds', 'forward_warm_s', 'jvp_full_warm_s',
            'vjp_full_warm_s', 'hvp_full_warm_s', 'peak_rss_gib', 'gpu_index', 'sampled_gpu_peak_gib'])
        for row in cases:
            memory = resources[row['repeat']]
            writer.writerow([row['atom_count'], int(np.prod(row['mesh'])), row['input_preparation_seconds'],
                *[row['timings'][key]['warm_median_seconds'] for key in keys],
                memory['peak_rss_kib']/1024**2, memory['gpu_index'], memory['sampled_peak_gpu_mib']/1024])

    ks = sorted([json.loads(path.read_text()) for path in OUTPUT.glob('al_ks_k*.json')], key=lambda row: row['kmesh'][0])
    comparison = []
    if ks and any(row['repeat'] == 1 for row in cases):
        native = next(row for row in cases if row['repeat'] == 1)
        density = np.load(OUTPUT/'al_1.npz')['density']
        previous = None
        for row in ks:
            rho = np.load(OUTPUT/f"al_ks_k{row['kmesh'][0]}.npz")['density'].ravel()
            comparison.append(dict(kmesh=row['kmesh'], basis_level=row['basis_level'], sigma_hartree=row['sigma_hartree'],
                wgc_minus_ks_extrapolated_energy_hartree=native['energy_hartree']-row['extrapolated_energy_hartree'],
                wgc_ks_density_relative_l2=float(np.linalg.norm(density-rho)/np.linalg.norm(rho)),
                ks_density_relative_increment=None if previous is None else float(np.linalg.norm(rho-previous)/np.linalg.norm(rho))))
            previous = rho
    (OUTPUT/'comparison_summary.json').write_text(json.dumps(dict(
        same_discrete_model={str(key): dict(energy_error_hartree_per_atom=value['energy_error_hartree_per_atom'],
            density_relative_l2_error=value['density_relative_l2_error']) for key, value in references.items()},
        finite_ks_comparison=comparison,
        physical_scope='KS basis, k sampling and smearing are not jointly converged; no pure KEDF error claim',
        scaling_scope='Measured finite size range; setup has explicit atom-pair and grid-atom arrays; no near-linear claim'), indent=2)+'\n')
    print('Saved scaling figure, CSV and separate same-model/finite-KS summary', flush=True)


if __name__ == '__main__':
    main()
