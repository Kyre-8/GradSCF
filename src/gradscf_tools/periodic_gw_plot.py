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


__all__ = ['plot_query']
