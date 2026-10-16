"""P01/P02: plot saved, independently checked LiH forces and cubic EOS."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants
from scipy.interpolate import CubicHermiteSpline


OUTPUT = Path('reproducibility/ad_showcase/P01_P02_lih_periodic')
BASELINE = 'pbe_szv_k2_fft31_repaired'
GROUPS = {
    'FFT mesh (SZV, k2³)': [('31³', BASELINE), ('41³', 'pbe_szv_k2_fft41'), ('51³', 'pbe_szv_k2_fft51')],
    'k mesh (SZV, FFT31³)': [('Γ', 'pbe_szv_k1_fft31'), ('2³', BASELINE),
                              ('3³', 'pbe_szv_k3_fft31'), ('4³', 'pbe_szv_k4_fft31'),
                              ('5³', 'pbe_szv_k5_fft31'), ('6³', 'pbe_szv_k6_fft31'),
                              ('8³', 'pbe_szv_k8_fft31')],
    'Basis (FFT41³, k2³)': [('SZV', 'pbe_szv_k2_fft41'), ('SR-DZVP', 'pbe_molopt_sr_k2_fft41')],
}
NATIVE_ONLY = {'pbe_szv_k5_fft31', 'pbe_szv_k6_fft31', 'pbe_szv_k8_fft31'}
EH_TO_GPA = physical_constants['Hartree energy'][0]/physical_constants['Bohr radius'][0]**3/1e9
# Fixed before the final k8 calculation. These are finite-model tolerances,
# not complete-basis or material-prediction accuracy claims.
CONVERGENCE_GATES = {
    'fft': dict(energy_hartree=1e-7, force_hartree_per_bohr=1e-7, stress_hartree_per_bohr3=1e-7),
    'k': dict(energy_hartree=1e-4, force_hartree_per_bohr=2e-6, stress_hartree_per_bohr3=2e-6),
}


def read_case(name):
    path = OUTPUT/(name+'.json')
    reference_path = OUTPUT/('pyscf_'+name+'.json')
    native = json.loads(path.read_text())
    assert native['complete']
    if name == 'pbe_szv_k8_fft31':
        budget = json.loads((OUTPUT/'k8_resource_budget.json').read_text())
        assert budget['accepted'] and budget['peak_rss_kib'] <= budget['budget_kib']
    if name in NATIVE_ONLY:
        return native, None, dict(native=hashlib.sha256(path.read_bytes()).hexdigest(), independent=None)
    reference = json.loads(reference_path.read_text())
    assert reference['complete'] and reference['passed']
    assert reference['native_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    return native, reference['reference'], dict(native=hashlib.sha256(path.read_bytes()).hexdigest(),
        independent=hashlib.sha256(reference_path.read_bytes()).hexdigest())


def differences(first, second):
    return dict(energy_hartree=abs(first['energy_hartree']-second['energy_hartree']),
        force_hartree_per_bohr=float(np.max(abs(np.asarray(first['forces_hartree_per_bohr'])-second['forces_hartree_per_bohr']))),
        stress_hartree_per_bohr3=float(np.max(abs(np.asarray(first['stress_hartree_per_bohr3'])-second['stress_hartree_per_bohr3']))))


def main():
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 220})
    names = list(dict.fromkeys(name for rows in GROUPS.values() for _, name in rows))
    names.extend(['hf_szv_k1_fft31', 'pbe_molopt_sr_k2_fft31'])
    cases = {name: read_case(name) for name in names}
    baseline = cases[BASELINE][0]
    fingerprints = {name: values[2] for name, values in cases.items()}
    convergence = {}
    for kind, first, last in [('fft', 'pbe_szv_k2_fft41', 'pbe_szv_k2_fft51'),
                              ('k', 'pbe_szv_k6_fft31', 'pbe_szv_k8_fft31')]:
        delta = differences(cases[first][0], cases[last][0])
        convergence[kind] = dict(first=first, last=last, increments=delta,
            gates=CONVERGENCE_GATES[kind],
            passed=all(delta[key] < limit for key, limit in CONVERGENCE_GATES[kind].items()))
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.8), constrained_layout=True)
    for column, (title, rows) in enumerate(GROUPS.items()):
        abscissa = np.arange(len(rows))
        for independent in [False, True]:
            selected = [index for index, (_, name) in enumerate(rows)
                        if cases[name][int(independent)] is not None]
            states = [cases[rows[index][1]][int(independent)] for index in selected]
            energy = [1000*(value['energy_hartree']-baseline['energy_hartree']) for value in states]
            force = [1e6*value['forces_hartree_per_bohr'][1][0] for value in states]
            stress_key = 'symmetric_public_stress_hartree_per_bohr3' if independent else 'stress_hartree_per_bohr3'
            pressure = [-EH_TO_GPA*np.trace(value[stress_key])/3 for value in states]
            for row, values in enumerate([energy, force, pressure]):
                axes[row, column].plot(abscissa[selected], values, marker='o' if independent else 'x',
                    color='#c65b22' if independent else '#246b8e', markerfacecolor='none',
                    linestyle='none' if independent else '-', linewidth=1.2,
                    label='PySCF, checked settings' if independent else 'GradSCF AD')
                axes[row, column].set_xticks(abscissa, [label for label, _ in rows])
                axes[row, column].grid(alpha=.2)
        axes[0, column].set_title(title)
    for column, kind, pair in [(0, 'fft', '41→51'), (1, 'k', '6→8')]:
        axes[2, column].set_xlabel(pair+(': within declared gates' if convergence[kind]['passed']
                                       else ': outside declared gates'))
    axes[0, 0].set_ylabel('E − common baseline (mEh)')
    axes[1, 0].set_ylabel('H force, x component (µEh/bohr)')
    axes[2, 0].set_ylabel('Pressure (GPa)')
    axes[0, 0].legend(frameon=False)
    fig.suptitle('Displaced LiH: separate discretization and basis sensitivities\n'
                 'PBE / GTH-Pade; a = 7.6 bohr; fixed integer occupations', fontsize=13)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUTPUT/f'lih_convergence.{suffix}')
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6), constrained_layout=True)
    quantities = [('energy_hartree', '|ΔE| (Eh)'),
                  ('force_hartree_per_bohr', 'max |ΔF| (Eh/bohr)'),
                  ('stress_hartree_per_bohr3', 'max |Δσ| (Eh/bohr³)')]
    for row, (kind, entries) in enumerate([('fft', list(GROUPS.values())[0]),
                                          ('k', list(GROUPS.values())[1])]):
        pairs = list(zip(entries[:-1], entries[1:]))
        increments = [differences(cases[first[1]][0], cases[last[1]][0])
                      for first, last in pairs]
        labels = [first[0]+'→'+last[0] for first, last in pairs]
        for column, (key, unit) in enumerate(quantities):
            ax = axes[row, column]
            ax.semilogy(range(len(pairs)), [delta[key] for delta in increments],
                        'o-', color='#246b8e', linewidth=1.2)
            ax.axhline(CONVERGENCE_GATES[kind][key], color='#c65b22',
                       linestyle='--', label='Declared last-increment gate')
            ax.set_xticks(range(len(pairs)), labels, rotation=30)
            ax.set_ylabel(unit); ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(['Energy', 'All force components', 'Full stress tensor'][column])
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].set_xlabel('FFT mesh; fixed SZV / k2³')
    axes[1, 0].set_xlabel('k mesh; fixed SZV / FFT31³')
    fig.suptitle('Successive increments resolve the final finite-model checks\n'
                 'The basis and jointly refined FFT/k limits remain separate questions', fontsize=12)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUTPUT/f'lih_convergence_increments.{suffix}')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    labels = ['Li x', 'Li y', 'Li z', 'H x', 'H y', 'H z']
    x = np.arange(6)
    for index, (name, label, color) in enumerate([
        ('pbe_szv_k1_fft31', 'PBE', '#246b8e'), ('hf_szv_k1_fft31', 'HF', '#c65b22')]):
        native, reference, _ = cases[name]
        shift = (index-.5)*.32
        axes[0].bar(x+shift, 1e3*np.asarray(native['forces_hartree_per_bohr']).reshape(-1),
                    width=.3, color=color, alpha=.8, label=label+' AD')
        axes[0].plot(x+shift, 1e3*np.asarray(reference['forces_hartree_per_bohr']).reshape(-1),
                     'ko', markersize=3, markerfacecolor='none',
                     label='PySCF' if index == 0 else None)
    axes[0].set_xticks(x, labels, rotation=30)
    axes[0].set_ylabel('Force (mEh/bohr)')
    axes[0].set_title('Matched Γ / SZV / FFT31³ model')
    axes[0].axhline(0, color='.5', linewidth=.7); axes[0].legend(frameon=False)
    fft_rows = list(GROUPS.values())[0]
    net = [np.linalg.norm(cases[name][0]['raw_net_force_hartree_per_bohr']) for _, name in fft_rows]
    axes[1].semilogy([31, 41, 51], net, 'o-', color='#246b8e')
    axes[1].set_xlabel('FFT points per lattice direction')
    axes[1].set_xticks([31, 41, 51])
    axes[1].set_ylabel('Raw |Σ force| (Eh/bohr)')
    axes[1].set_title('Translation error retained; no force projection')
    axes[1].grid(alpha=.2)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUTPUT/f'lih_forces.{suffix}')
    plt.close(fig)

    native_path, reference_path = OUTPUT/'lattice_optimization.json', OUTPUT/'pyscf_lattice_optimization.json'
    native, reference = json.loads(native_path.read_text()), json.loads(reference_path.read_text())
    assert native['complete'] and native['passed'] and reference['complete'] and reference['passed']
    assert reference['native_sha256'] == hashlib.sha256(native_path.read_bytes()).hexdigest()
    points = sorted(native['points'], key=lambda row: row['a_bohr'])
    independent = sorted(reference['points'], key=lambda row: row['settings']['a_bohr'])
    a = np.array([row['a_bohr'] for row in points]); minimum = native['minimum']
    # A visual guide uses the actual sampled energies and their AD slopes.
    # No interpolated point enters the minimum, curvature or convergence gates.
    energy_curve = CubicHermiteSpline(a, [row['energy_hartree'] for row in points],
                                     [row['derivative_hartree_per_bohr'] for row in points])
    dense_a = np.linspace(a.min(), a.max(), 401)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(dense_a, 1e3*(energy_curve(dense_a)-minimum['energy_hartree']),
                 '-', color='#246b8e', linewidth=1.2, label='Hermite guide')
    axes[0].plot(a, [1e3*(row['energy_hartree']-minimum['energy_hartree']) for row in points],
                 'x', color='#246b8e', label='GradSCF')
    axes[0].plot([row['settings']['a_bohr'] for row in independent],
                 [1e3*(row['energy_hartree']-minimum['energy_hartree']) for row in independent],
                 'o', markerfacecolor='none', color='#c65b22', label='PySCF')
    axes[0].set_ylabel('E − native cubic minimum (mEh)'); axes[0].legend(frameon=False)
    axes[1].plot(dense_a, -4*energy_curve(dense_a, 1)/(3*dense_a**2)*EH_TO_GPA,
                 '-', color='#246b8e', linewidth=1.2)
    axes[1].plot(a, [row['pressure_hartree_per_bohr3']*EH_TO_GPA for row in points],
                 'x', color='#246b8e')
    axes[1].axhline(0, color='.5', linewidth=.7); axes[1].set_ylabel('AD pressure (GPa)')
    for ax in axes:
        ax.axvline(minimum['a_bohr'], color='#246b8e', linestyle=':', linewidth=1)
        ax.set_xlabel('Cubic lattice parameter a (bohr)'); ax.grid(alpha=.2)
    fig.suptitle('LiH cubic minimum: SZV / FFT31³ / k2³\n'
                 'Fixed fractional sites; finite-model EOS, not a full crystal-stability test', fontsize=12)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUTPUT/f'lih_equation_of_state.{suffix}')
    plt.close(fig)

    rows = []
    for name in names:
        state, _, _ = cases[name]
        row = dict(case=name, energy_hartree=state['energy_hartree'],
            pressure_hartree_per_bohr3=state['pressure_hartree_per_bohr3'],
            gap_hartree=state['sampled_indirect_gap_hartree'], nao=state['nao'],
            peak_rss_kib=state['peak_rss_kib'], elapsed_seconds=state['elapsed_seconds'])
        row.update({f'force_{atom}_{axis}_hartree_per_bohr': state['forces_hartree_per_bohr'][i][j]
                    for i, atom in enumerate(['Li', 'H']) for j, axis in enumerate('xyz')})
        rows.append(row)
    with (OUTPUT/'convergence.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    increments = {}
    for label, entries in GROUPS.items():
        increments[label] = [dict(first=first[1], second=second[1],
            **differences(cases[first[1]][0], cases[second[1]][0])) for first, second in zip(entries[:-1], entries[1:])]
    increments['SR-DZVP FFT sensitivity (k2³)'] = [dict(
        first='pbe_molopt_sr_k2_fft31', second='pbe_molopt_sr_k2_fft41',
        **differences(cases['pbe_molopt_sr_k2_fft31'][0], cases['pbe_molopt_sr_k2_fft41'][0]))]
    summary = dict(complete=True, passed=all(item['passed'] for item in convergence.values()),
        input_sha256=fingerprints, successive_increments=increments,
        convergence=convergence,
        independent_checked_cases=[name for name in names if name not in NATIVE_ONLY],
        native_only_discretization_cases=sorted(NATIVE_ONLY),
        eos_input_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        eos_independent_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        eos_curve='Cubic Hermite visual guide through native energies and AD slopes; not used for numerical gates.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Matched-code agreement and finite discretization/basis sensitivity are distinct. No complete-basis or experimental-material accuracy is inferred.')
    (OUTPUT/'plot_validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    assert summary['passed'], 'One or more declared finite-model convergence gates failed.'
    print('LiH figures and source-checked tables complete', flush=True)


if __name__ == '__main__':
    main()

# Recorded on jy_gpu_1, CPU float64 (2026-10-16); see P01_P02_lih_periodic.
# LiH figures and source-checked tables complete: four PNG/PDF figure pairs.
# FFT41->51 and k6->8 last-increment gates passed on their separate axes.
# SR-DZVP changes the displaced H force sign; basis convergence is not established.
# Plots preserve raw force sums, failed basis candidates and native-only k points.
