"""Plot raw Si mode tensors and relative polarized weights, without projection.

The chosen Lorentzian width is a display resolution. It is not a computed
phonon lifetime. The spectral weights are |e_s^T (d alpha_cell/dQ) e_i|^2;
frequency-dependent photon/thermal factors and absolute cross sections are
not included. Negative raw modes remain visible in the frequency table.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path('reproducibility/ad_showcase/P06_silicon_raman')


def plot_sampling(names=('gamma_fft15', 'k2_fft15', 'k3_fft15'), *, parameter='kmesh'):
    """Vary either the k mesh or FFT mesh, holding every other setting fixed."""
    if parameter not in ('kmesh', 'mesh'):
        raise ValueError("Choose parameter='kmesh' or parameter='mesh'.")
    records = [json.loads((OUTPUT/(name+'_probe.json')).read_text()) for name in names]
    reference = {key: value for key, value in records[0]['settings'].items() if key != parameter}
    for record in records:
        assert record['complete'] and record['probe_passed'] and record['nonzero_xy']
        assert {key: value for key, value in record['settings'].items() if key != parameter} == reference
    mesh = np.array([record['settings'][parameter][0] for record in records])
    k_sampling = parameter == 'kmesh'
    output_name = 'silicon_k_sampling' if k_sampling else 'silicon_fft_sampling'
    axis_label = (f"Full k mesh; FFT{reference['mesh'][0]} held fixed" if k_sampling
                  else f"FFT mesh; k={reference['kmesh'][0]}³ held fixed")
    columns = np.array([record['dchi_column_dQ'] for record in records])
    ratio = abs(columns[:, 1:]/columns[:, :1])
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), constrained_layout=True)
    axes[0].plot(mesh, columns[:, 0], 'o-', color='#2166ac')
    axes[0].set_ylabel(r'Allowed $\partial\chi_{xy}/\partial Q$ ($a_0^{-1}m_e^{-1/2}$)')
    for index, label, color in [(0, 'yy / xy', '#b2182b'), (1, 'zy / xy', '#348741')]:
        axes[1].semilogy(mesh, ratio[:, index], 'o-', label=label, color=color)
    axes[1].set_ylabel('Absolute forbidden / allowed derivative')
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(mesh, ['Γ' if k_sampling and value == 1 else f'{value}³' for value in mesh])
        ax.set_xlabel(axis_label)
        ax.grid(alpha=.2)
    fig.suptitle('Si PBE / GTH-PBE / SZV: raw single-factor sampling sensitivity\n'
                 'No point-group projection; derivative checks do not establish sampling convergence', fontsize=11)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUTPUT/(output_name+'.'+suffix), dpi=220)
    plt.close(fig)
    with (OUTPUT/(output_name+'.csv')).open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['kmesh_side' if k_sampling else 'fft_mesh_side', 'dchi_xy_dQ', 'dchi_yy_dQ', 'dchi_zy_dQ',
                         'abs_yy_over_xy', 'abs_zy_over_xy'])
        for n, column, leakage in zip(mesh, columns, ratio):
            writer.writerow([n, *column, *leakage])
    record = dict(complete=True, fixed_settings=reference, sampled_parameter=parameter,
        raw_forbidden_over_allowed=ratio.tolist(),
        input_sha256={name: hashlib.sha256((OUTPUT/(name+'_probe.json')).read_bytes()).hexdigest()
                      for name in names},
        interpretation='Sampling sensitivity of raw finite-model derivatives; no extrapolation or symmetry projection.')
    record['kmesh_sides' if k_sampling else 'fft_mesh_sides'] = mesh.tolist()
    (OUTPUT/(output_name+'.json')).write_text(json.dumps(record, indent=2)+'\n')
    return ratio


def main(name='gamma_fft15'):
    tensor_path, phonon_path = (OUTPUT/(name+suffix+'.json') for suffix in ('_tensors', '_phonons'))
    report, phonons = (json.loads(path.read_text()) for path in (tensor_path, phonon_path))
    if not report['complete'] or not report['reciprocity_passed']:
        raise RuntimeError('Complete the raw Raman tensors and reciprocity check first.')
    if hashlib.sha256(phonon_path.read_bytes()).hexdigest() != report['phonon_sha256']:
        raise RuntimeError('The matched phonon record changed.')
    tensors = np.asarray(report['mode_dalpha_dQ'])
    frequencies = np.asarray(report['signed_frequencies_cm1'])
    modes = np.asarray(phonons['mass_weighted_modes'])
    kside = report['settings']['kmesh'][0]
    klabel = 'Γ' if kside == 1 else f'{kside}³'
    model_label = f"Si PBE/GTH-SZV, k = {klabel}, FFT = {report['settings']['mesh'][0]}³"
    probe = json.loads((OUTPUT/(name+'_probe.json')).read_text())
    cartesian = np.asarray(report['cartesian_mass_scaled_dchi'])
    recovered = (cartesian[2, :, 1]-cartesian[5, :, 1])/np.sqrt(2.)
    normalization_error = float(np.max(abs(recovered-probe['dchi_column_dQ'])))
    if normalization_error >= 1e-8:
        raise RuntimeError('The Cartesian tensor does not recover the checked normalized optical column.')
    np.testing.assert_allclose(modes.T@modes, np.eye(6), atol=1e-10, rtol=0)
    optical_basis = np.vstack([np.eye(3), -np.eye(3)])/np.sqrt(2.)
    optical_fraction = np.sum(abs(optical_basis.T@modes)**2, axis=0)
    selected = np.flatnonzero(optical_fraction > .99)
    if len(selected) != 3 or np.any(frequencies[selected] <= 0.):
        raise RuntimeError('The raw finite-model modes do not form a stable isolated optical triplet.')
    optical_tensors = tensors[selected]
    diagonal_fraction = float(np.linalg.norm(np.diagonal(optical_tensors, axis1=-2, axis2=-1))
                              /np.linalg.norm(optical_tensors))
    offdiagonal = np.sqrt(2.)*np.stack([optical_tensors[:,1,2], optical_tensors[:,2,0],
                                       optical_tensors[:,0,1]], axis=-1)
    singular_values = np.linalg.svd(offdiagonal, compute_uv=False)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.6), constrained_layout=True)
    maximum = float(np.max(abs(tensors[selected])))
    for ax, mode in zip(axes, selected):
        image = ax.imshow(tensors[mode], cmap='coolwarm', vmin=-maximum, vmax=maximum)
        ax.set_xticks(range(3), ['x', 'y', 'z']); ax.set_yticks(range(3), ['x', 'y', 'z'])
        ax.set_xlabel('Incident field'); ax.set_ylabel('Scattered polarization')
        ax.set_title(f'Mode {mode+1}: {frequencies[mode]:.2f} cm⁻¹')
        for i in range(3):
            for j in range(3):
                color = 'white' if abs(tensors[mode,i,j]) > .55*maximum else 'black'
                ax.text(j, i, f'{tensors[mode,i,j]:.3g}', ha='center', va='center', fontsize=9, color=color)
    fig.colorbar(image, ax=axes, label=r'$\partial\alpha_{\rm cell}/\partial Q$ ($a_0^2m_e^{-1/2}$)', shrink=.8)
    fig.suptitle(f'{model_label}: raw finite-model Raman tensors\n'
                 f'Optical diagonal/total norm = {diagonal_fraction:.3%}; no symmetry projection', fontsize=11)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUTPUT/(name+'_raw_tensors.'+suffix))
    plt.close(fig)

    angle = np.linspace(0., 2*np.pi, 361)
    incident = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=-1)
    crossed = np.stack([-np.sin(angle), np.cos(angle), np.zeros_like(angle)], axis=-1)
    weights = []
    for scattered in (incident, crossed):
        amplitude = np.einsum('ai,mij,aj->ma', scattered, tensors[selected], incident)
        weights.append(np.sum(abs(amplitude)**2, axis=0))
    # A single common scale preserves the parallel/crossed comparison.
    scale = max(float(np.max(weights)), 1e-30)
    width = 5.  # Imposed half width, in cm^-1.
    grid = np.linspace(max(0., min(frequencies[selected])-80), max(frequencies[selected])+80, 1601)
    selected_tensors = tensors[selected]
    stick_weights = np.array([abs(selected_tensors[:,0,0])**2,
                             abs(selected_tensors[:,1,0])**2])
    lines = width/(np.pi*((grid[None,:]-frequencies[selected,None])**2+width**2))
    spectra = stick_weights@lines
    spectrum_scale = max(float(np.max(spectra)), 1e-30)
    fig = plt.figure(figsize=(10, 4.4), constrained_layout=True)
    polar = fig.add_subplot(1, 2, 1, projection='polar')
    spectral = fig.add_subplot(1, 2, 2)
    for values, label, color in zip(weights, ['Parallel', 'Crossed'], ['#2166ac', '#b2182b']):
        polar.plot(angle, values/scale, label=label, color=color)
    polar.set_title('Optical-subspace sum; eᵢ in the xy plane', pad=20)
    polar.legend(loc='lower center', bbox_to_anchor=(.5, -.2), ncol=2, frameon=False)
    for values, label, color in zip(spectra, ['xx', 'yx'], ['#2166ac', '#b2182b']):
        spectral.plot(grid, values/spectrum_scale, label=label, color=color)
    spectral.set_xlabel('Wavenumber (cm⁻¹)')
    spectral.set_ylabel('Relative squared polarizability-derivative weight')
    spectral.set_title('eᵢ = x; imposed HWHM = 5 cm⁻¹')
    spectral.legend(frameon=False); spectral.grid(alpha=.2)
    fig.suptitle(f'{model_label}: finite-model polarization response', fontsize=11)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUTPUT/(name+'_polarized_response.'+suffix))
    plt.close(fig)

    with (OUTPUT/(name+'_mode_weights.csv')).open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['mode', 'signed_frequency_cm-1', 'optical_fraction',
                         'xx_weight_bohr4_per_me', 'yx_weight_bohr4_per_me'])
        for mode in range(6):
            writer.writerow([mode+1, frequencies[mode], optical_fraction[mode],
                             abs(tensors[mode,0,0])**2, abs(tensors[mode,1,0])**2])
    validation = dict(complete=True, tensor_sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        phonon_sha256=report['phonon_sha256'], optical_modes=(selected+1).tolist(),
        optical_fractions=optical_fraction.tolist(), raw_negative_mode_count=int(np.sum(frequencies < 0)),
        normalized_probe_recovery_error=normalization_error,
        optical_diagonal_norm_fraction=diagonal_fraction,
        optical_offdiagonal_singular_values=singular_values.tolist(),
        angular_common_scale=scale, spectral_common_scale=spectrum_scale,
        display_halfwidth_cm1=width, interpretation=__doc__)
    (OUTPUT/(name+'_plot_validation.json')).write_text(json.dumps(validation, indent=2)+'\n')
    print('Saved raw tensor and polarized-response figures:', name, flush=True)


if __name__ == '__main__':
    main()
