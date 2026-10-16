"""CH4 GW/BSE with the existing finite-eta RPA resolvent implementation.

Run with PYTHONPATH=src JAX_PLATFORMS=cpu. No CLI or PySCF calculation.
This explicitly selects the public low-level G0W0 W representation; it does
not fix or validate the original GW facade's observed batched-LAPACK failure.
The CD grid, eta, QP roots and orbital windows match methane_gw_spectrum.py.
Independent MolGW comparison and plotting are separate steps.
"""
from pathlib import Path
import hashlib
import json
import platform
from time import perf_counter

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
from gradscf import bse, gto, gw, scf
from gradscf.tools.spectra import HARTREE_TO_EV

output = Path('outputs/methane_resolvent_spectrum')
output.mkdir(parents=True, exist_ok=True)
geometry = '''C 0.000 0.000 0.000
H 0.629 0.629 0.629
H 0.629 -0.629 -0.629
H -0.629 0.629 -0.629
H -0.629 -0.629 0.629'''
started = perf_counter()
mf = scf.RHF(gto.M(atom=geometry, unit='Angstrom', basis='aug-cc-pvdz', cart=True),
             conv_tol=1e-12).density_fit('weigend').run()
assert mf.converged
reference, state = mf.to_reference(), mf.scf_result
nocc, nmo = int(np.count_nonzero(mf.mo_occ)), len(mf.mo_energy)
occupied, virtual = tuple(range(nocc)), tuple(range(nocc, nocc+29))
qp_orbitals = occupied+virtual
coefficients = np.asarray(mf.mo_coeff)
mo_factors = np.einsum('Ppq,pi,qj->Pij', reference.df_factors, coefficients, coefficients)
dipole_mo = np.einsum('pi,xpq,qj->xij', coefficients, reference.dipole_integrals, coefficients)
common = dict(mo_energy=mf.mo_energy, mo_coeff=mf.mo_coeff, nocc=nocc,
    df_factors=reference.df_factors, fock_matrix=state.fock_matrix,
    hcore_matrix=state.hcore_matrix, density_matrix=state.density_matrix,
    nw=200, eta=1e-5, orbs=qp_orbitals, g_orbitals=range(nmo),
    screening_occupied=occupied, screening_virtual=range(nocc,nmo))
report = dict(complete=False, geometry_angstrom=geometry, basis='aug-cc-pvdz', auxbasis='weigend',
    cartesian=True, nocc=nocc, nmo=nmo, scf_energy_ha=float(mf.e_tot), scf_conv_tol=1e-12,
    scf_seconds=perf_counter()-started, nw=200, eta_gw_ha=1e-5, qp_solver='secant',
    qp_orbitals=qp_orbitals, g_orbitals=list(range(nmo)), optical_occupied=occupied,
    optical_virtual=virtual, screening_occupied=occupied, screening_virtual=list(range(nocc,nmo)),
    evgw_max_cycle=80, evgw_conv_tol_ha=1e-8, evgw_damping=.3, spectral_hwhm_ev=.15,
    implementation_branch='Existing public finite-eta resolvent; same discrete CD equation. Original facade runtime failure remains separately archived.',
    scope='Fixed-geometry windowed static singlet BSE; not a basis/window-converged spectrum or experimental validation.',
    hostname=platform.node(), python=platform.python_version(), jax=jax.__version__,
    backend=jax.default_backend(), float64=bool(jax.config.x64_enabled), methods={})

for method in ('g0w0','evgw0','evgw'):
    started = perf_counter()
    if method == 'g0w0':
        result = gw.g0w0_cd_restricted(**common, resolvent_expansion=True, qp_solver='secant')
    else:
        result = gw.evgw_cd_restricted(**common, update_w=method == 'evgw',
                                      max_iter=80, tol=1e-8, damping=.3)
    assert result.converged
    gw_seconds = perf_counter()-started
    source = bse.BSEReference.from_gw_result(result, nocc=nocc, mo_factors=mo_factors,
                                             dipole_mo=dipole_mo)
    response = bse.BSE(source, occupied=occupied, virtual=virtual, singlet=True,
        tda=False, solver='dense', nroots=145, max_dense=320, conv_tol=1e-9).run()
    assert response.converged.all() and np.all(response.result.stable)
    energies, strengths = np.asarray(response.e)*HARTREE_TO_EV, np.asarray(response.oscillator_strength())
    weights = np.asarray(result.qp_weight)[list(qp_orbitals)]
    assert np.isfinite(weights).all()
    report['methods'][method] = dict(qp_energy_ha=np.asarray(result.mo_energy).tolist(),
        screening_energy_ha=np.asarray(result.screening_energy).tolist(), qp_weight_selected=weights.tolist(),
        qp_residual_max_ha=float(np.max(abs(result.qp_residual))), gw_seconds=gw_seconds,
        bse_residual_max_ha=float(np.max(response.result.residual_norms)),
        excitation_ev=energies.tolist(), oscillator_strength=strengths.tolist(),
        elapsed_seconds=perf_counter()-started)
    np.savez_compressed(output/f'{method}_reference.npz', qp_energy=result.mo_energy,
        screening_energy=result.screening_energy, mo_factors=mo_factors, dipole_mo=dipole_mo,
        qp_computed_mask=result.qp_computed_mask, qp_converged_mask=result.converged_mask)
    (output/'spectrum.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(method, 'first bright / eV', energies[np.flatnonzero(strengths>1e-5)[0]],
          'QP residual / Eh', report['methods'][method]['qp_residual_max_ha'], flush=True)

report['complete'] = True
report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
source_root, source_hash = Path(gw.__file__).resolve().parents[2], hashlib.sha256()
for path in sorted(source_root.rglob('*.py')):
    source_hash.update(path.relative_to(source_root.parent).as_posix().encode())
    source_hash.update(path.read_bytes())
report['source_python_sha256'] = source_hash.hexdigest()
(output/'spectrum.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print('Saved QP/145-root results and explicit BSE snapshots to', output, flush=True)
