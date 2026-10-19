"""Plot saved, finite-model GaAs GW comparisons without running GW.

``comparison.npz`` contains axis_ev, path_distance, path_ticks, path_labels,
pbe_bands and {g0w0,evgw,qsgw,scgw}_{bands,dos}. All energies share the
PBE-VBM zero and use eV; DOS uses states/cell/eV. Optional scgw_spectral
has shape (npath,nenergy). Failed values remain NaN and are never filled.
``comparison.json`` supplies nocc, display_bands (zero-based), source_kmesh,
fft_mesh, basis, fc, beta_hartree_inverse, finite_model_note,
dos_broadening_ev, energy_limits_ev (optional), and per-method gaps with
gamma_direct_ev, sampled_gap_ev, minimum_direct_ev. No gap is recomputed
from a drawing path or substituted for a missing reported gap.
scGW lines and gap bars represent selected finite-eta ReSigma Dyson roots.
Their local Z diagnostics are distinct from physical spectral weights;
imaginary-axis, particle-number and PSD checks do not establish real-axis stability.
"""
from pathlib import Path
import json
import textwrap


_METHODS = ('g0w0', 'evgw', 'qsgw', 'scgw')
_NAMES = ('G₀W₀', 'evGW', 'qsGW', 'scGW')
_COLORS = ('#2166AC', '#B7791F', '#238B7B', '#8B4A8F')


def plot_comparison(directory):
    """Write gaas_four_gw.pdf/png from actual comparison arrays.

    Four band panels use the same limits and PBE reference. The two right
    panels compare DOS and three distinct reported gap definitions. Optional
    scGW intensity is the continued spectral band map when available;
    selected real-part Dyson roots remain in the saved arrays.
    Return output paths; this function performs no electronic calculation.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np

    directory = Path(directory)
    report = json.loads((directory/'comparison.json').read_text())
    with np.load(directory/'comparison.npz', allow_pickle=False) as stored:
        data = {key: stored[key].copy() for key in stored.files}
    energy, distance = data['axis_ev'], data['path_distance']
    ticks, labels = data['path_ticks'], data['path_labels'].astype(str)
    reference = data['pbe_bands']
    if (energy.ndim != 1 or distance.ndim != 1 or len(energy) < 2 or len(distance) < 2
            or not np.isfinite(energy).all() or not np.isfinite(distance).all()
            or np.any(np.diff(energy) <= 0) or np.any(np.diff(distance) < 0)
            or distance[-1] <= distance[0] or reference.ndim != 2
            or reference.shape[0] != len(distance) or ticks.ndim != 1
            or not len(ticks) or labels.shape != ticks.shape or not np.isfinite(ticks).all()
            or np.any(np.diff(ticks) < 0) or ticks.min() < distance[0]-1e-10
            or ticks.max() > distance[-1]+1e-10):
        raise ValueError('Invalid energy/path axes, path ticks, or PBE band shape')
    bands = np.asarray(report['display_bands'], dtype=int)
    if (bands.ndim != 1 or not len(bands) or len(np.unique(bands)) != len(bands)
            or bands.min() < 0 or bands.max() >= reference.shape[1]
            or not 0 < report['nocc'] < reference.shape[1]):
        raise ValueError('Use valid zero-based display bands and occupied-band count')
    for method in _METHODS:
        if (data[method+'_bands'].shape != reference.shape
                or data[method+'_dos'].shape != energy.shape):
            raise ValueError(f'{method} band/DOS shapes disagree with the saved axes')
    limits = np.asarray(report.get('energy_limits_ev', [-15., 15.]), dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError('energy_limits_ev must contain increasing finite eV limits')
    labels = ['Γ' if label in ('G', 'Gamma', 'GAMMA', 'Γ') else label for label in labels]
    style = {'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.titlesize': 12,
             'axes.labelsize': 10, 'pdf.fonttype': 42, 'savefig.dpi': 240}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.6),
            gridspec_kw={'width_ratios': [1., 1., 1.02]})
        panel_axes = (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])
        for panel, method, name, color in zip(panel_axes, _METHODS, _NAMES, _COLORS):
            values = data[method+'_bands'][:, bands]
            spectral_map = False
            if method == 'scgw' and 'scgw_spectral' in data:
                intensity = np.asarray(data['scgw_spectral'])
                if intensity.shape != (len(distance), len(energy)):
                    raise ValueError('scgw_spectral must have shape (npath,nenergy)')
                visible = (energy >= limits[0]) & (energy <= limits[1])
                shown = intensity[:, visible]
                finite = np.isfinite(shown)
                maximum = np.max(shown[finite]) if finite.any() else 0.
                if maximum > 0:
                    image = panel.pcolormesh(distance, energy[visible],
                        np.ma.masked_invalid(shown.T/maximum), shading='auto', cmap='Purples',
                        norm=PowerNorm(.5, vmin=0., vmax=1.), rasterized=True)
                    color_axis = panel.inset_axes([.69, .13, .27, .025])
                    bar = fig.colorbar(image, cax=color_axis, orientation='horizontal', ticks=[0., 1.])
                    bar.ax.tick_params(labelsize=6, length=2)
                    bar.set_label('A / max A (shown window)', fontsize=6, labelpad=0)
                    spectral_map = True
            for column, band in enumerate(bands):
                panel.plot(distance, np.where(np.isfinite(reference[:, band]), reference[:, band], np.nan),
                           ls='--', color='#7C858D', lw=.9, zorder=2)
                if not spectral_map:
                    panel.plot(distance, np.where(np.isfinite(values[:, column]), values[:, column], np.nan),
                               color=color, lw=1.25, zorder=3)
            for tick in ticks:
                panel.axvline(tick, color='#D8D8D8', lw=.65, zorder=1)
            panel.axhline(0., color='#B0B0B0', lw=.7, zorder=1)
            panel.set(title='scGW · spectral function' if spectral_map else name,
                      xlim=(distance[0], distance[-1]), ylim=limits,
                      xlabel='Wavevector path', ylabel='Energy − PBE VBM (eV)')
            panel.set_xticks(ticks, labels)
            line_label = 'Selected ReΣ roots' if method == 'scgw' else name
            method_handle = (Patch(facecolor=color, label='Continued A(k,E)') if spectral_map else
                Line2D([], [], color=color, lw=1.25, label=line_label))
            panel.legend(handles=[Line2D([], [], color='#7C858D', ls='--', lw=.9, label='PBE'),
                method_handle], frameon=False, fontsize=8,
                loc='upper right')
            missing = int((~np.isfinite(values)).sum())
            if missing:
                label = 'selected roots uncertified' if spectral_map else 'values unavailable'
                panel.text(.02, .03 if spectral_map else .94, f'{missing}/{values.size} {label}',
                           transform=panel.transAxes, va='bottom' if spectral_map else 'top', fontsize=8)
        dos = axes[0, 2]
        for method, name, color, linestyle in zip(_METHODS, _NAMES, _COLORS, ('-', '--', '-.', ':')):
            values = data[method+'_dos']
            dos.plot(np.where(np.isfinite(values), values, np.nan), energy,
                     color=color, ls=linestyle, lw=1.45, label=name)
        dos.axhline(0., color='#B0B0B0', lw=.7)
        dos.set(title='Density of states', ylim=limits, xlabel='DOS (states / cell / eV)',
                ylabel='Energy − PBE VBM (eV)')
        dos.legend(frameon=False, fontsize=8, ncol=2)
        unavailable = [name for method, name in zip(_METHODS, _NAMES)
                       if not np.isfinite(data[method+'_dos']).any()]
        if unavailable:
            dos.text(.02, .02, 'DOS unavailable: '+', '.join(unavailable),
                     transform=dos.transAxes, fontsize=7.5)
        gap_axis = axes[1, 2]
        metrics = (('gamma_direct_ev', 'Γ direct', -.23, None, 1.),
                   ('sampled_gap_ev', 'Sampled fundamental', 0., '//', 1.),
                   ('minimum_direct_ev', 'Sampled minimum direct', .23, '..', .4))
        for row, (method, color) in enumerate(zip(_METHODS, _COLORS)):
            gaps = report['gaps'].get(method, {})
            for key, _, offset, hatch, alpha in metrics:
                value = gaps.get(key)
                if value is None or not np.isfinite(float(value)):
                    gap_axis.text(0., row+offset, 'n/a', va='center', fontsize=7)
                    continue
                gap_axis.barh(row+offset, float(value), height=.2, edgecolor=color,
                    facecolor='white' if hatch == '//' else color, hatch=hatch,
                    alpha=alpha, linewidth=.9)
                gap_axis.annotate(f'{float(value):.2f}', (float(value), row+offset),
                    xytext=(4 if float(value) >= 0 else -4, 0), textcoords='offset points',
                    va='center', ha='left' if float(value) >= 0 else 'right', fontsize=7)
        gap_axis.set_yticks(range(4), (*_NAMES[:3], 'scGW†'))
        gap_axis.set_ylim(3.6, -1.35)
        gap_axis.axvline(0., color='#B0B0B0', lw=.7)
        gap_axis.set(title='Distinct gap definitions', xlabel='Gap (eV)')
        gap_axis.margins(x=.25)
        gap_axis.legend(handles=[Patch(facecolor='#666666' if hatch != '//' else 'white',
            edgecolor='#555555', hatch=hatch, alpha=alpha, label=label)
            for _, label, _, hatch, alpha in metrics], frameon=False, fontsize=7.5, loc='upper right')
        for panel in axes.flat:
            panel.spines[['top', 'right']].set_visible(False)
            panel.tick_params(direction='out', length=3, width=.7)
            panel.grid(axis='y', color='#E4E4E4', linewidth=.5, alpha=.6)
            panel.set_axisbelow(True)
        fig.suptitle('GaAs · four GW schemes in one finite source model', fontsize=15, y=.965)
        metadata = (f"{report['basis']} · source k mesh {tuple(report['source_kmesh'])} · "
                    f"FFT {tuple(report['fft_mesh'])} · common PBE-VBM energy zero")
        fig.text(.5, .922, metadata, ha='center', fontsize=9.5, color='#444444')
        note = ('Interpolated operators in a finite SZV/source-k model; no spin–orbit coupling; '
                'no bulk-convergence claim. ' + report['finite_model_note'])
        controls = (f"fc={report['fc']}; scGW β={report['beta_hartree_inverse']:g} Ha⁻¹; "
                    f"DOS broadening={report['dos_broadening_ev']:g} eV. "
                    'Missing/nonfinite values are omitted; gap bars use reported definitions.')
        scgw_note = ('scGW: continued spectral density; † gap bars use selected finite-η ReΣ roots. '
            'Principal-peak identity and real-axis stability are not certified. '
            'Spectral colors use square-root normalization; local Z is not physical spectral weight.')
        fig.text(.07, .065, textwrap.fill(note, 160)+'\n'+textwrap.fill(controls, 160)+'\n'+textwrap.fill(scgw_note, 160),
                 ha='left', va='bottom', fontsize=8, linespacing=1.5, color='#444444')
        fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.87, wspace=.34, hspace=.33)
        outputs = {suffix: directory/f'gaas_four_gw.{suffix}' for suffix in ('pdf', 'png')}
        for output in outputs.values():
            fig.savefig(output, facecolor='white', bbox_inches='tight')
        plt.close(fig)
    return outputs


__all__ = ['plot_comparison']
