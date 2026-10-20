"""Plot only explicitly computed, converged direct periodic GW query points."""
from pathlib import Path
import json


def plot_query(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    directory = Path(directory)
    data = np.load(directory/'bands.npz')
    report = json.loads((directory/'native.json').read_text())
    bands = report['output_bands']
    silicon = report.get('system') == 'silicon'
    x = np.linspace(0., 1., len(data['kpoints'])) if silicon else data['fractional_kpoints'][:, 0]
    reference = data['ks_energy'][0, bands[0]]
    conversion = 27.211386245988
    ks = (data['ks_energy']-reference)*conversion
    qp = (data['qp_energy']-reference)*conversion
    valid = data['computed_mask'] & data['converged_mask']
    fig, (ax, residual) = plt.subplots(1, 2, figsize=(9.0, 3.8),
        gridspec_kw={'width_ratios': [1.45, 1.]}, constrained_layout=True)
    for position, band in enumerate(bands):
        ax.plot(x, ks[:, band], '--', color='#64748b', lw=1.5,
                label='PBE' if position == 0 else None)
        ax.plot(x, np.where(valid[:, band], qp[:, band], np.nan), 'o-', color='#047857',
                lw=1.6, ms=5, label='Direct G₀W₀' if position == 0 else None)
        failed = data['computed_mask'][:, band] & ~data['converged_mask'][:, band]
        if np.any(failed):
            ax.scatter(x[failed], ks[failed, band], marker='x', color='#b91c1c',
                       label='QP not converged (at KS level)' if position == 0 else None)
        residual.semilogy(x, np.maximum(np.abs(data['qp_residual'][:, band]), 1e-16),
            'o-' if position == 0 else 'x-',
            lw=1.3, label=f'Band {band}')
    xlabel = 'Position along generic segment' if silicon else 'Fractional k along [100]'
    ax.set(xlabel=xlabel,
           ylabel='Energy − PBE band 3 at first point (eV)' if silicon else 'Energy − PBE valence at Γ (eV)',
           title='Si: direct finite-model generic segment' if silicon else 'Small periodic H₂: direct off-mesh points')
    ax.legend(frameon=False, fontsize=9)
    residual.axhline(1e-9, color='#b91c1c', ls='--', lw=1, label='Root tolerance')
    residual.set(xlabel=xlabel, ylabel='|Dyson residual| (Hartree)',
                 title='Actual QP convergence')
    residual.legend(frameon=False, fontsize=9)
    residual.text(.03, .08, 'Display floor: 10⁻¹⁶ Ha', transform=residual.transAxes, fontsize=9)
    for axis in (ax, residual):
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(alpha=.18)
    grid, nq = report['mesh'][0], int(np.prod(report['source_kmesh']))
    title = f'GTH-SZV · FFT {grid}³ · fixed PBE W₀ on {nq} q points · full OV + VO · fc=False'
    if silicon:
        title += '\nDirect points; no interpolation, full high-symmetry path, or bulk-convergence claim'
    fig.suptitle(title, fontsize=10)
    fig.savefig(directory/'bands.png', dpi=180)
    fig.savefig(directory/'bands.pdf')
    plt.close(fig)


def _material_panel(axis, directory):
    """Show direct QPs and their matching KS reference, with failed points absent."""
    import numpy as np

    directory = Path(directory)
    report = json.loads((directory/'native.json').read_text())
    with np.load(directory/'bands.npz') as stored:
        data = {key: stored[key].copy() for key in stored.files}
    conversion = 27.211386245988
    zero = float(data['energy_zero_hartree'])
    distance = data['path_distance']
    # ASE treats consecutive special points as a display discontinuity when
    # a short segment has no interior sample. These declared paths are
    # continuous: recover their actual distances from the unwrapped points.
    if report.get('path') and 'path_fractional_kpoints' in data:
        tick_indices, previous = [], -1
        for tick in report['tick_positions']:
            matches = np.flatnonzero(np.isclose(distance, tick, atol=1e-10, rtol=0))
            matches = matches[matches > previous]
            if not len(matches):
                raise ValueError('Saved path ticks do not match the sampled path')
            previous = int(matches[0])
            tick_indices.append(previous)
        reciprocal = 2*np.pi*np.linalg.inv(np.asarray(report['lattice_bohr'])).T
        raw_k = data['path_fractional_kpoints']@reciprocal
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(raw_k, axis=0), axis=1))]
        data['path_distance'] = distance
        report['tick_positions'] = distance[tick_indices].tolist()
    valid = data['computed_mask'] & data['converged_mask']
    ks = (data['ks_energy']-zero)*conversion
    occupied = [band for band in report['output_bands'] if band < report['nocc']]
    aligned = bool(occupied) and bool(valid[:, occupied].all())
    qp_zero = float(np.max(data['qp_energy'][:, occupied])) if aligned else zero
    qp = (data['qp_energy']-qp_zero)*conversion
    for position, band in enumerate(report['output_bands']):
        axis.plot(distance, ks[:, band], color='#64748b', ls='--', lw=1.,
                  label='Matching PBE' if position == 0 else None)
        axis.plot(distance, np.where(valid[:, band], qp[:, band], np.nan),
                  'o-', color='#047857', lw=1., ms=3.,
                  label='Direct G₀W₀' if position == 0 else None)
    for tick in report['tick_positions']:
        axis.axvline(tick, color='#a0a0a0', lw=.5, zorder=0)
    axis.axhline(0., color='#a0a0a0', lw=.5, zorder=0)
    labels = ['Γ' if label == 'G' else label for label in report['tick_labels']]
    axis.set_xticks(report['tick_positions'], labels)
    requested = np.asarray(report['output_bands'])
    coverage = valid[:, requested]
    count = int(coverage.sum())
    total = coverage.size
    name = {'silicon': 'Si', 'gaas': 'GaAs', 'gan': 'GaN', 'graphene': 'Graphene'}
    axis.margins(y=.15)
    axis.set(xlim=(distance[0], distance[-1]),
        xlabel='Wavevector path',
        ylabel='Energy − each sampled VBM (eV)' if aligned else 'Energy − PBE VBM (eV)',
        title=f"{name.get(report['system'], report['system'])} · {count}/{total} converged QP values")
    note = ('Separate valence references; raw energies retained' if aligned else
            'QP valence reference incomplete; common PBE zero')
    axis.text(.02, .02, note+'\nComputed points; lines guide the eye', transform=axis.transAxes,
              fontsize=8, color='#444444')
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(loc='upper right', frameon=False, fontsize=9)
    return report, data


def plot_material(directory):
    """Plot a saved material path; this function performs no SCF/GW calculation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    directory = Path(directory)
    fig, (axis, residual) = plt.subplots(1, 2, figsize=(10.5, 4.6),
        gridspec_kw={'width_ratios': [2., 1.]}, constrained_layout=True)
    report, data = _material_panel(axis, directory)
    computed = data['computed_mask']
    for band in report['output_bands']:
        residual.semilogy(data['path_distance'],
            np.where(computed[:, band], np.maximum(abs(data['qp_residual'][:, band]), 1e-16), np.nan),
            'o-', lw=.75, ms=2.5, label=f'Band {band}')
    residual.axhline(report['root_tolerance_hartree'], color='#b91c1c', ls='--', lw=1.)
    residual.set(xlabel='Path distance', ylabel='|Dyson residual| (Ha)',
                 title='Requested QP convergence')
    residual.spines[['top', 'right']].set_visible(False)
    residual.text(.03, .03, 'Display floor: 10⁻¹⁶ Ha', transform=residual.transAxes, fontsize=8)
    fig.suptitle(f"PBE / GTH-SZV · q={tuple(report['source_kmesh'])} · "
        f"FFT={tuple(report['mesh'])} · {report['nw']} imaginary frequencies\n"
        'Full OV + VO · nonlinear subspace QPs · body only; discretization not converged', fontsize=10)
    for suffix in ('png', 'pdf'):
        fig.savefig(directory/f'bands.{suffix}', dpi=200)
    plt.close(fig)


def plot_material_gallery(directories, output):
    """Compare saved material demonstrations in separate, explicitly scoped panels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directories, output = list(directories), Path(output)
    if not directories:
        raise ValueError('Supply at least one saved material directory')
    rows = (len(directories)+1)//2
    fig, axes = plt.subplots(rows, 2, figsize=(11., 4.6*rows), squeeze=False,
                             constrained_layout=True)
    for axis, directory in zip(axes.flat, directories):
        _material_panel(axis, directory)
    for axis in list(axes.flat)[len(directories):]:
        axis.set_visible(False)
    fig.suptitle('Matching PBE and direct G₀W₀ material bands\n'
                 'Compact GTH-SZV / body-only finite models; no bulk convergence claim', fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.'+suffix), dpi=200)
    plt.close(fig)


__all__ = ['plot_query', 'plot_material', 'plot_material_gallery']
