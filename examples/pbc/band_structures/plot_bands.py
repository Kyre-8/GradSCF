"""Plot saved material bands, preserving each case's validation and model scope.

``plot_saved()`` reads existing JSON/NPZ only; it never starts an SCF or GW run.
The older ``plot`` and ``compare_methods`` entry points require passed references.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants


OUTPUT = Path('reproducibility/ad_showcase/PBC_material_bands')
HARTREE_TO_EV = physical_constants['Hartree energy in eV'][0]


def plot(case='silicon_pbe_pilot'):
    folder = OUTPUT/case
    path = folder/'native.json'
    native = json.loads(path.read_text())
    reference = json.loads((folder/'pyscf.json').read_text())
    assert native['passed'] and reference['passed']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference['native_sha256']
    with np.load(folder/'bands.npz') as stored:
        energy, distance = stored['energies_hartree'], stored['path_distance']
        zero = float(stored['energy_zero_hartree'])
    with np.load(folder/'pyscf_bands.npz') as stored:
        oracle = stored['energies_hartree']
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True,
                              gridspec_kw={'width_ratios': [2.3, 1.]})
    for band in range(energy.shape[1]):
        axes[0].plot(distance, (energy[:, band]-zero)*HARTREE_TO_EV,
                     color='#2166ac', linewidth=1.1, label='GradSCF' if band == 0 else None)
        axes[0].plot(distance, (oracle[:, band]-zero)*HARTREE_TO_EV,
                     color='#d95f02', linestyle='--', linewidth=.65,
                     label='Independent PySCF' if band == 0 else None)
    labels = ['Γ' if label == 'G' else label for label in native['tick_labels']]
    for axis in axes:
        for tick in native['tick_positions']:
            axis.axvline(tick, color='#aaaaaa', linewidth=.5)
        axis.set_xticks(native['tick_positions'], labels)
        axis.set_xlim(distance[0], distance[-1])
        axis.set_xlabel('Wavevector path')
    axes[0].axhline(0., color='#555555', linewidth=.6)
    axes[0].set(ylim=(-12, 10), ylabel='KS/GKS energy relative to declared zero (eV)',
                title='Native and independent bands')
    axes[0].legend(frameon=False, fontsize=9)
    frontier = np.asarray(reference['frontier_indices'])
    discrepancy = np.max(abs(energy[:, frontier]-oracle[:, frontier]), axis=1)*HARTREE_TO_EV*1000
    axes[1].plot(distance, discrepancy, color='#348741')
    axes[1].axhline(reference['gates']['frontier_bands_hartree']*HARTREE_TO_EV*1000,
                    color='black', linestyle=':', label='Numerical gate')
    axes[1].set(ylabel='Maximum frontier-band difference (meV)',
                title='Same-input numerical comparison')
    axes[1].legend(frameon=False, fontsize=8)
    options = native['settings']
    subtitle = (f"{native['material']} / {native['method'].upper()} / {options['basis']}\n"
        f"k={tuple(options['kmesh'])}, FFT={tuple(options['mesh'])}; "
        f"sampled path gap {native['path_gap_ev']:.3f} eV; discretization not yet converged")
    if native['physical_dimension'] < 3:
        subtitle += (f"\n{native['electrostatic_boundary']}, cell height {options['vacuum']:g} Å; "
                     'vacuum convergence not established')
    fig.suptitle(subtitle, fontsize=10)
    for suffix in ('png', 'pdf'):
        fig.savefig(folder/f'bands.{suffix}')
    plt.close(fig)


def compare_methods(material='silicon', methods=('lda', 'pbe', 'r2scan'), *, cases=None,
                    output_name=None):
    """Compare validated cases at the same geometry, basis, grid and k path."""
    records = []
    for method in methods:
        folder = OUTPUT/(f'{material}_{method}_pilot' if cases is None else cases[method])
        path = folder/'native.json'
        native = json.loads(path.read_text())
        reference = json.loads((folder/'pyscf.json').read_text())
        assert native['passed'] and reference['passed']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference['native_sha256']
        with np.load(folder/'bands.npz') as stored:
            data = {key: stored[key].copy() for key in stored.files}
        if records:
            previous = records[0][0]
            assert native['settings'] == previous['settings']
            assert native['lattice_angstrom'] == previous['lattice_angstrom']
            assert native['coordinates_angstrom'] == previous['coordinates_angstrom']
            assert native.get('electrostatic_dimension', 3) == previous.get('electrostatic_dimension', 3)
            np.testing.assert_array_equal(data['fractional_kpoints'], records[0][1]['fractional_kpoints'])
        records.append((native, data))
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(1, len(methods), figsize=(4*len(methods), 4.9),
        sharey=True, squeeze=False, constrained_layout=True)
    colors = ('#2166ac', '#b35806', '#348741', '#762a83')
    for index, (native, data) in enumerate(records):
        axis = axes[0, index]
        energy = (data['energies_hartree']-float(data['energy_zero_hartree']))*HARTREE_TO_EV
        axis.plot(data['path_distance'], energy, color=colors[index % len(colors)], linewidth=.9)
        for tick in native['tick_positions']:
            axis.axvline(tick, color='#aaaaaa', linewidth=.5)
        axis.axhline(0., color='#666666', linewidth=.6)
        axis.set_xticks(native['tick_positions'], ['Γ' if x == 'G' else x for x in native['tick_labels']])
        axis.set(xlim=(data['path_distance'][0], data['path_distance'][-1]), ylim=(-12, 10),
            xlabel='Wavevector path', title=f"{native['method'].upper()}\nSampled path gap {native['path_gap_ev']:.3f} eV")
    axes[0, 0].set_ylabel('Band energy relative to each declared reference (eV)')
    native = records[0][0]
    options = native['settings']
    fig.suptitle(f"{material}: fixed structure / {options['basis']} / {options['pseudo']}\n"
        f"k={tuple(options['kmesh'])}, FFT={tuple(options['mesh'])}; numerical cross-checks passed, discretization not converged",
        fontsize=10)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUTPUT/f'{output_name or material+"_methods"}.{suffix}')
    plt.close(fig)


COLORS = ('#0072b2', '#d55e00', '#009e73', '#cc79a7')
STYLES = ('-', '--', '-.', ':')
GROUPS = {
    'silicon': [('silicon_lda_pilot', 'silicon_pbe_pilot', 'silicon_r2scan_pilot'),
                ('silicon_pbe_compact', 'silicon_pbe0_compact')],
    'gaas': [('gaas_lda_pilot', 'gaas_pbe_pilot', 'gaas_r2scan_cpu16')],
    'gan': [('gan_lda_cpu16', 'gan_pbe_memory_probe'), ('gan_r2scan_compact',)],
    'graphene': [('graphene_lda_pilot', 'graphene_pbe_pilot', 'graphene_r2scan_pilot'),
                 ('graphene_pbe_slab12', 'graphene_pbe_slab16', 'graphene_pbe_slab20')],
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _saved_case(folder):
    native_path = folder/'native.json'
    native = json.loads(native_path.read_text())
    row = dict(case=folder.name, material=native.get('material'),
        method=native.get('method'), complete=native.get('complete', False),
        native_passed=native.get('passed', False), settings=native.get('settings'),
        exxdiv=native.get('exxdiv', 'not recorded'),
        energy_alignment=native.get('energy_alignment'),
        hashes={'native.json': _sha(native_path)}, reference_status='not available')
    reference = None
    if (folder/'pyscf.json').exists():
        reference = json.loads((folder/'pyscf.json').read_text())
        if reference['native_sha256'] != row['hashes']['native.json']:
            raise ValueError(f'{folder.name}: reference belongs to a different native record')
        row['reference_status'] = 'passed' if reference.get('passed') else 'mismatch'
        row['hashes']['pyscf.json'] = _sha(folder/'pyscf.json')
        row['reference_errors_hartree'] = {key: value for key, value in reference.items()
            if key.endswith('error_hartree')}
    arrays = None
    if row['complete'] and (folder/'bands.npz').exists():
        with np.load(folder/'bands.npz') as stored:
            arrays = {key: stored[key].copy() for key in stored.files}
        if not np.isfinite(arrays['energies_hartree']).all():
            raise ValueError(f'{folder.name}: nonfinite saved bands')
        row['hashes']['bands.npz'] = _sha(folder/'bands.npz')
        row['path_points'], row['bands'] = arrays['energies_hartree'].shape
    row['plotted'] = arrays is not None
    return native, arrays, row, reference


def _band_panel(axis, records, names, title, *, slab=False, mismatch=False):
    """Overlay native curves; energy zeros stay local to their declared case."""
    first = records[names[0]][0]
    for index, name in enumerate(names):
        native, data, row, _ = records[name]
        label = {'r2scan': 'r²SCAN'}.get(native['method'], native['method'].upper())
        if slab:
            label += f" · {native['settings']['vacuum']:g} Å / FFT z={native['settings']['mesh'][2]}"
        if 'compact' in name and native['material'] == 'silicon':
            label += f" · exxdiv={native['exxdiv']}"
        if row['reference_status'] == 'not available':
            label += ' [no independent check]'
        elif row['reference_status'] == 'mismatch':
            label += ' [native; reference mismatch]'
        energy = (data['energies_hartree']-float(data['energy_zero_hartree']))*HARTREE_TO_EV
        lines = axis.plot(data['path_distance'], energy, color=COLORS[index],
                          linestyle=STYLES[index], linewidth=1.05)
        lines[0].set_label(label)
    data = records[names[0]][1]
    axis.set_xticks(first['tick_positions'], ['Γ' if x == 'G' else x for x in first['tick_labels']])
    for tick in first['tick_positions']:
        axis.axvline(tick, color='#a0a0a0', linewidth=.45, zorder=0)
    axis.axhline(0., color='#555555', linewidth=.65, zorder=0)
    axis.set(xlim=(data['path_distance'][0], data['path_distance'][-1]), ylim=(-12, 10),
             xlabel='Wavevector path', ylabel='Energy − each saved reference (eV)', title=title)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(loc='upper center', bbox_to_anchor=(.5, -.14), frameon=False, fontsize=9)
    if mismatch:
        axis.text(.5, .97, 'REFERENCE MISMATCH / UNVALIDATED', transform=axis.transAxes,
            ha='center', va='top', color='#a51c30', weight='bold', fontsize=11,
            bbox=dict(facecolor='white', edgecolor='#a51c30', alpha=.95, pad=5))
        errors = records[names[0]][3]
        axis.text(.03, .03,
            f"Native only; independent PySCF gate failed\n"
            f"ΔE = {errors['total_energy_error_hartree']:.3f} Ha; "
            f"max frontier Δε = {errors['frontier_band_error_hartree']*HARTREE_TO_EV*1000:.1f} meV",
            transform=axis.transAxes, fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.92))


def _gw_panel(axis, folder, *, query=False):
    """Scatter actual requested QPs; rejected candidates never become energies."""
    report_file, array_file = ('native.json', 'bands.npz') if query else ('results.json', 'levels.npz')
    report = json.loads((folder/report_file).read_text())
    with np.load(folder/array_file) as stored:
        data = {key: stored[key].copy() for key in stored.files}
    ks = data['ks_energy' if query else 'ks_hartree']
    qp = data['qp_energy' if query else 'qp_hartree']
    computed = data['computed_mask' if query else 'qp_computed_mask']
    converged = data['converged_mask']
    valid = computed & converged
    frac = data['fractional_kpoints']
    zero = float(ks[:, 3].max())
    displayed = np.concatenate((ks[:, (3, 4)].ravel(), qp[:, (3, 4)][valid[:, (3, 4)]]))
    displayed = (displayed-zero)*HARTREE_TO_EV
    padding = max(1., float(np.ptp(displayed))*.15)
    for position, band in enumerate((3, 4)):
        x = np.arange(len(ks)) + (-.09 if position == 0 else .09)
        axis.scatter(x, (ks[:, band]-zero)*HARTREE_TO_EV, color='#64748b',
            marker='_' if position == 0 else '+', s=90,
            label='PBE KS (bands 3, 4)' if position == 0 else None)
        axis.scatter(x[valid[:, band]], (qp[valid[:, band], band]-zero)*HARTREE_TO_EV,
            facecolors=COLORS[0] if position == 0 else 'none', edgecolors=COLORS[0], s=42,
            label='Accepted direct G₀W₀' if position == 0 else None)
        for point in np.flatnonzero(computed[:, band] & ~converged[:, band]):
            axis.annotate(f'band {band}: QP failed\ncandidate omitted',
                (x[point], (ks[point, band]-zero)*HARTREE_TO_EV), xytext=(0, 12),
                textcoords='offset points', ha='center', va='bottom', color='#a51c30', fontsize=9)
    labels = ['(' + ', '.join(f'{round(float(x), 3) + 0.:g}' for x in point) + ')' for point in frac]
    axis.set_xticks(np.arange(len(ks)), labels, fontsize=8)
    axis.set(xlim=(-.45, len(ks)-.55),
        ylim=(float(displayed.min())-padding, float(displayed.max())+padding),
        xlabel='Direct fractional k points (no interpolation)',
        ylabel='Energy − maximum saved KS band 3 (eV)',
        title=('(d) Generic queries: 6/6 QPs accepted\nfull OV+VO · fc=False · nw=16' if query else
               '(c) Original mesh: 3/4 QPs accepted\nlegacy folded OV · fc=True · nw=8'))
    axis.axhline(0., color='#aaaaaa', linewidth=.6)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(loc='upper center', bbox_to_anchor=(.5, -.17), frameon=False, fontsize=9)
    return dict(path=str(folder), requested=int(computed.sum()), accepted=int(valid.sum()),
        passed=report['passed'], plotted_mask=valid.tolist(),
        independent_material_gw_benchmark=False,
        hashes={report_file: _sha(folder/report_file), array_file: _sha(folder/array_file)})


def plot_saved():
    """Render every completed saved material case, without electronic calculations.

    The 17-case inventory is checked for coverage. Incomplete cases remain in the
    manifest. Independently checked means same discrete model agreement only.
    """
    destination = OUTPUT/'saved_bands'
    destination.mkdir(parents=True, exist_ok=True)
    records = {path.parent.name: _saved_case(path.parent) for path in sorted(OUTPUT.glob('*/native.json'))}
    expected = {name for groups in GROUPS.values() for group in groups for name in group}
    actual = {name for name, (_, data, _, _) in records.items() if data is not None}
    if actual != expected:
        raise ValueError(f'Update saved-case coverage: extra={actual-expected}, missing={expected-actual}')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'pdf.fonttype': 42,
                         'savefig.dpi': 200, 'font.family': 'DejaVu Sans'})
    figures = {}

    def save(fig, name, cases, caption):
        fig.text(.5, -.015, caption, ha='center', va='top', fontsize=9)
        for suffix in ('png', 'pdf'):
            fig.savefig(destination/f'{name}.{suffix}', bbox_inches='tight')
        plt.close(fig)
        figures[name] = list(cases)

    fixed = 'GTH-DZVP-MOLOPT-SR / GTH-PBE · no spin–orbit coupling'
    caveat = 'Saved finite models; basis, k mesh, FFT grid and vacuum convergence are not established.'
    zeros = 'Each semiconductor uses its saved occupied-band maximum; graphene uses its saved chemical potential.'
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10), constrained_layout=True)
    _band_panel(axes[0, 0], records, GROUPS['silicon'][0],
                '(a) Matched semilocal models\nk=(2,2,2) · FFT31³ · 121 path points')
    _band_panel(axes[0, 1], records, GROUPS['silicon'][1],
                '(b) Compact PBE / PBE0 models\nk=(2,1,1) · FFT21³ · 61 path points')
    gw = [_gw_panel(axes[1, 0], OUTPUT.parent/'PBC_silicon_gw'),
          _gw_panel(axes[1, 1], OUTPUT.parent/'periodic_gw_query'/'silicon', query=True)]
    fig.suptitle('Silicon · all five saved KS/GKS cases and both direct GW data sets\n'
                 + fixed + ' (upper panels); GW uses GTH-SZV (lower panels)', fontsize=13)
    save(fig, 'silicon_all', sum(GROUPS['silicon'], ()),
         'Upper panels: all five independent same-input band checks passed. Lower panels: QP solver certificates only; no independent Si GW benchmark.\n'
         'GW: PBE W₀, source k=(2,1,1), FFT21³, Ewald exchange, η=0.001 Ha. The two GW prescriptions are distinct.\n'+caveat)

    fig, ax = plt.subplots(figsize=(9, 5.8), constrained_layout=True)
    _band_panel(ax, records, GROUPS['gaas'][0], 'Matched LDA / PBE / r²SCAN models\nk=(2,2,2) · FFT41³ · 121 path points')
    fig.suptitle('GaAs · all three saved cases\n'+fixed, fontsize=13)
    save(fig, 'gaas_all', GROUPS['gaas'][0], 'All three independent same-input band checks passed.\n'+caveat)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.9), constrained_layout=True)
    _band_panel(axes[0], records, GROUPS['gan'][0], '(a) LDA / PBE\nk=(2,2,2) · FFT(41,41,65) · 121 points')
    _band_panel(axes[1], records, GROUPS['gan'][1], '(b) Compact r²SCAN · exxdiv=ewald\nk=(2,1,1) · FFT(21,21,33) · 61 points', mismatch=True)
    fig.suptitle('GaN · all three saved native cases\n'+fixed, fontsize=13)
    save(fig, 'gan_all', sum(GROUPS['gan'], ()), 'PBE passed its independent check; LDA has no saved independent check. Compact r²SCAN failed its independent check.\n'
         'Panels differ in k mesh, FFT grid and path sampling: their differences are not a controlled functional comparison.\n'+caveat)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), constrained_layout=True)
    _band_panel(axes[0], records, GROUPS['graphene'][0], '(a) 3D repeated slabs · 20 Å cell height\nk=(3,3,1) · FFT(25,25,161) · 121 points')
    _band_panel(axes[1], records, GROUPS['graphene'][1], '(b) 2D Coulomb cutoff · PBE\nk=(3,3,1) · height / z-grid vary · 121 points', slab=True)
    fig.suptitle('Graphene · all six saved cases\n'+fixed, fontsize=13)
    save(fig, 'graphene_all', sum(GROUPS['graphene'], ()), 'Left: all three independent checks passed. Right: 20 Å passed; 12 / 16 Å have no saved independent check.\n'
         'Each curve is aligned to its own SCF chemical potential (electronic temperature 0.0002 Ha).\n'+caveat)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0), constrained_layout=True)
    titles = ('Si · k=(2,2,2), FFT31³', 'GaAs · k=(2,2,2), FFT41³',
              'GaN · k=(2,2,2), FFT(41,41,65)', 'Graphene · 3D, 20 Å, k=(3,3,1), FFT(25,25,161)')
    overview = []
    for axis, material, title in zip(axes.flat, GROUPS, titles):
        group = GROUPS[material][0]
        _band_panel(axis, records, group, title)
        overview.extend(group)
    fig.suptitle('Saved material band structures · semilocal overview\n'+fixed, fontsize=13)
    save(fig, 'materials_overview', overview,
         'Shown: matched-setting groups within each material. Per-material figures also include compact hybrids, failed GaN r²SCAN, 2D slabs and direct Si GW points.\n'
         + zeros+' Display window: −12 to 10 eV; saved files retain every band.\n'+caveat)
    manifest = dict(physical_calculations_started=0, material_complete_cases=len(actual),
        material_records=len(records), material_cases=[row for _, _, row, _ in records.values()],
        figures=figures, gw=gw, energy_window_ev=[-12, 10],
        alignment=zeros, interpolation='None beyond straight segments joining saved KS/GKS path samples; GW uses unconnected markers.',
        historical_exxdiv='Unrecorded for the semilocal pilots; explicit values are retained where present.',
        plot_source_sha256=_sha(Path(__file__)))
    (destination/'plot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(complete_cases=len(actual), figures=list(figures), gw=gw), indent=2))
    return manifest


if __name__ == '__main__':
    plot_saved()
