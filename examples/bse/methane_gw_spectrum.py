"""CH4: compare G0W0, evGW0 and evGW followed by static full singlet BSE.

Native GradSCF throughout; no PySCF calculation and no command-line interface.
Run from the repository root with PYTHONPATH=src JAX_PLATFORMS=cpu.
All G/W sums retain the full orbital space. The explicitly chosen QP window
contains all occupied and 29 virtual orbitals, preserving complete degenerate
shells. Higher virtual energies remain at RHF values in the evGW iterations.
This is a windowed eigenvalue-self-consistency comparison in one diffuse basis,
not a basis-converged gas-phase spectrum or an experimental reproduction.
"""
from pathlib import Path
import json
import platform
import time

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf import bse, gto, gw, scf
from gradscf.tools.spectra import HARTREE_TO_EV

output = Path('outputs/methane_gw_bse')
output.mkdir(parents=True, exist_ok=True)
methods = ('g0w0', 'evgw0', 'evgw')
basis, auxbasis = 'aug-cc-pvdz', 'weigend'
nvirtual = 29
nw = 200
eta_gw = 1e-5                       # Hartree, real-frequency W broadening
eta_ev = .15                        # eV, spectral half-width (FWHM ~ 0.30 eV)
energy_ev = np.linspace(8., 24., 3201)

mol = gto.M(
    atom='''
    C   0.000   0.000   0.000
    H   0.629   0.629   0.629
    H   0.629  -0.629  -0.629
    H  -0.629   0.629  -0.629
    H  -0.629  -0.629   0.629
    ''',
    basis=basis, unit='Angstrom', cart=True,
)
start = time.perf_counter()
mf = scf.RHF(mol, conv_tol=1e-12).density_fit(auxbasis).run()
assert mf.converged
scf_seconds = time.perf_counter() - start
nocc = int(np.count_nonzero(mf.mo_occ))
nmo = len(mf.mo_energy)
occupied = tuple(range(nocc))
virtual = tuple(range(nocc, nocc + nvirtual))
qp_orbitals = occupied + virtual
assert virtual[-1] < nmo
print(f'RHF: {float(mf.e_tot):.12f} Ha; {nmo} MOs; {scf_seconds:.2f} s', flush=True)

report = {
    'geometry_angstrom': str(mol.atom), 'basis': basis, 'auxbasis': auxbasis,
    'cartesian': True, 'scf_energy_ha': float(mf.e_tot), 'scf_conv_tol': 1e-12,
    'scf_seconds': scf_seconds, 'nocc': nocc, 'nmo': nmo,
    'qp_orbitals': qp_orbitals, 'optical_occupied': occupied,
    'optical_virtual': virtual, 'g_orbitals': list(range(nmo)),
    'screening_occupied': occupied, 'screening_virtual': list(range(nocc, nmo)),
    'nw': nw, 'eta_gw_ha': eta_gw, 'spectral_hwhm_ev': eta_ev,
    'bse': 'static full singlet, dense, all roots in the optical window',
    'bse_conv_tol_ha': 1e-9, 'evgw_conv_tol_ha': 1e-8,
    'evgw_damping': .3, 'evgw_max_cycle': 80, 'qp_solver': 'secant',
    'backend': jax.default_backend(), 'jax': jax.__version__,
    'python': platform.python_version(), 'platform': platform.platform(),
    'device': str(jax.devices()[0]), 'float64': bool(jax.config.x64_enabled),
    'energy_ev': energy_ev.tolist(), 'methods': {},
}

for method in methods:
    start = time.perf_counter()
    mygw = gw.GW(
        mf, method=method, nw=nw, eta=eta_gw,
        max_cycle=80, conv_tol=1e-8, damp=.3,
        g_orbitals=range(nmo), screening_occupied=occupied,
        screening_virtual=range(nocc, nmo),
    ).run(orbs=qp_orbitals)
    assert mygw.converged
    qp_residual = float(np.max(np.abs(mygw.result.qp_residual)))
    weights = np.asarray(mygw.result.qp_weight)[list(qp_orbitals)]
    assert np.all(np.isfinite(weights))
    gw_seconds = time.perf_counter() - start
    print(f'{method}: GW {gw_seconds:.2f} s; residual {qp_residual:.3e} Ha; '
          f'min Z {weights.min():.6f}', flush=True)

    # W0 for G0W0/evGW0; the updated W for evGW is inherited from mygw.
    # Every root in this optical space is retained in the polarizability.
    response = bse.BSE(
        mygw, occupied=occupied, virtual=virtual, tda=False,
        singlet=True, solver='dense', nroots=len(occupied) * len(virtual),
        max_dense=320, conv_tol=1e-9,
    ).run()
    print(f'{method}: BSE max residual '
          f'{float(np.max(response.result.residual_norms)):.3e} Ha; '
          f'stable {bool(np.all(response.result.stable))}; '
          f'converged {bool(response.converged.all())}', flush=True)
    assert response.converged.all() and np.all(response.result.stable)
    excitation_ev = np.asarray(response.e) * HARTREE_TO_EV
    strengths = np.asarray(response.oscillator_strength())
    sigma = np.asarray(response.absorption_cross_section(
        energy_ev / HARTREE_TO_EV, eta=eta_ev / HARTREE_TO_EV, unit='Mb',
    ))
    assert np.all(np.isfinite(sigma)) and np.all(sigma >= 0)
    elapsed = time.perf_counter() - start
    bright = np.flatnonzero(strengths > 1e-5)
    print(f'{method}: first bright root {excitation_ev[bright[0]]:.6f} eV; '
          f'total {elapsed:.2f} s', flush=True)
    report['methods'][method] = {
        'qp_energy_ha': np.asarray(mygw.mo_energy).tolist(),
        'screening_energy_ha': np.asarray(mygw.result.screening_energy).tolist(),
        'qp_weight_selected': weights.tolist(),
        'qp_residual_max_ha': qp_residual, 'gw_seconds': gw_seconds,
        'bse_residual_max_ha': float(np.max(response.result.residual_norms)),
        'bse_stability_margins_ha': np.asarray(response.result.stability_margins).tolist(),
        'excitation_ev': excitation_ev.tolist(), 'oscillator_strength': strengths.tolist(),
        'cross_section_mb': sigma.tolist(), 'elapsed_seconds': elapsed,
    }
    (output / 'spectrum.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    np.savetxt(output / f'{method}_transitions.csv', np.column_stack((excitation_ev, strengths)),
               delimiter=',', header='energy_eV,oscillator_strength', comments='')

    # A fixed-MO snapshot permits independent optical-window checks without
    # rerunning GW. It retains the actual QP coverage and screening provenance.
    data = mygw.get_bse_inputs()
    dipole_mo = np.einsum('pi,xpq,qj->xij', mf.mo_coeff, data['dipole_ao'], mf.mo_coeff)
    np.savez_compressed(
        output / f'{method}_reference.npz', qp_energy=mygw.mo_energy,
        screening_energy=mygw.result.screening_energy, mo_factors=data['mo_factors'],
        dipole_mo=dipole_mo, qp_computed_mask=mygw.result.qp_computed_mask,
        qp_converged_mask=mygw.result.converged_mask,
    )

np.savetxt(output / 'absorption.csv', np.column_stack(
    [energy_ev] + [report['methods'][m]['cross_section_mb'] for m in methods]),
    delimiter=',', header='energy_eV,G0W0_Mb,evGW0_Mb,evGW_Mb', comments='')
print(f'Saved {output}; plot with examples/bse/plot_methane_gw_spectrum.py', flush=True)

# Executed with JAX 0.8.1, CPU float64 (Apple M4 Pro):
# RHF energy: -40.199597689343 Ha
# Method    First bright band/eV   Minimum selected Z   Maximum QP residual/Ha
# G0W0          10.857565               0.886308               6.624e-11
# evGW0         10.833348               0.842398               5.165e-09
# evGW          10.808672               0.835719               3.228e-09
# All 145 full-BSE roots in each optical window are stable and converged;
# maximum physical BSE residual across the three runs: 1.802e-13 Ha.
# First bright bands are threefold degenerate. Their summed strengths are
# 0.34629912 (G0W0), 0.34571233 (evGW0), 0.34283750 (evGW).
# The all-61-MO evGW0 probe did NOT converge; the above is the stated QP window.
