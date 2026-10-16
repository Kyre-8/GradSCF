"""Clamped-ion static dielectric response of the same small LiH Born model.

The perturbation is the macroscopic Berry electric enthalpy -V E.P. The
zero-field density response comes from the shared KS implicit equations.
FFT31/k2/SZV is a finite-model verification, not a converged material result.
"""
import hashlib
import json
import platform
import resource
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)
from gradscf.dfpt.electric import electric_problem
from gradscf.pbc.polarization import make_plan, polarization
from lih_born_charges import mean_field, SETTINGS, LINEAR


OUTPUT = Path('reproducibility/ad_showcase/P04_lih_dielectric')


def run(name='gradscf', *, axes=(0, 1, 2), settings=None):
    started = perf_counter()
    settings = dict(SETTINGS if settings is None else settings)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT/(name+'.json')
    destination.write_text('{"complete": false}\n')
    mf = mean_field(settings)
    if not mf.converged:
        raise RuntimeError('The zero-field LiH reference did not converge.')
    plan = make_plan(np.asarray(mf.kpts)@np.asarray(mf.cell.lattice).T/(2*np.pi))
    checked = polarization(mf.inputs, mf.result, plan, nocc=mf.cell.nelec)
    print('Reference', float(mf.e_tot), 'gap', float(checked.minimum_gap), flush=True)

    problem = electric_problem(mf)
    respond = jax.jit(lambda direction:problem.response(tangent=direction, config=LINEAR))
    records, densities = [], []
    for axis in axes:
        before = perf_counter()
        response = respond(jnp.eye(3)[axis])
        jax.block_until_ready(response)
        row = dict(axis=axis, chi_column=np.asarray(response.values['polarization']).tolist(),
                   residual=float(response.residual_norm), converged=bool(response.converged),
                   seconds=perf_counter()-before)
        records.append(row)
        densities.append(np.asarray(response.density))
        print('Field response', row, flush=True)
        if not row['converged'] or not np.isfinite(row['chi_column']).all():
            raise RuntimeError('An electric response failed its true linear residual.')

    full = tuple(axes) == (0, 1, 2)
    chi, epsilon, diagnostics = None, None, {}
    if full:
        chi = np.stack([row['chi_column'] for row in records], axis=1)
        epsilon = np.eye(3)+4*np.pi*chi
        diagnostics = dict(raw_maximum_asymmetry=float(abs(chi-chi.T).max()),
            raw_maximum_cubic_deviation=float(abs(chi-np.eye(3)*np.trace(chi)/3).max()),
            minimum_symmetric_chi_eigenvalue=float(np.linalg.eigvalsh(.5*(chi+chi.T)).min()))
        if diagnostics['raw_maximum_asymmetry'] > 1e-8:
            raise RuntimeError('The unprojected susceptibility violates reciprocity.')
    paths = ['src/gradscf/dfpt/electric.py', 'src/gradscf/pbc/polarization.py',
             'src/gradscf/dfpt/ks.py', 'src/gradscf/pbc/_kpoint.py',
             'examples/pbc/lih_born_charges.py']
    report = dict(complete=True, full_tensor=full, settings=settings,
        energy_hartree=float(mf.e_tot), reference_gap_hartree=float(checked.minimum_gap),
        reference_minimum_link=float(jnp.min(checked.minimum_link_singular_value)),
        reference_phase_anchor=[np.asarray(p).tolist() for p in checked.phases],
        volume_bohr3=float(mf.cell.volume), directions=records,
        coordinates_bohr=np.asarray(mf.cell.coords).tolist(),
        lattice_bohr=np.asarray(mf.cell.lattice).tolist(),
        fractional_kpoints=(np.asarray(mf.kpts)@np.asarray(mf.cell.lattice).T/(2*np.pi)).tolist(),
        occupations=list(mf.cell.nelec),
        pseudopotential_charges=np.asarray(mf.cell.charges).tolist(),
        susceptibility=None if chi is None else chi.tolist(),
        epsilon_infinity=None if epsilon is None else epsilon.tolist(),
        definition='chi[alpha,beta]=dP_alpha/dE_beta, epsilon_inf=I+4*pi*chi; fixed nuclei/cell',
        field_units='Eh/(e bohr)', polarization_units='e/bohr^2',
        elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__,
        dtype='float64', **diagnostics,
        scope='Declared finite FFT/k/basis model. Electronic static clamped-ion response, no ionic relaxation, no symmetry projection or material convergence claim.')
    destination.write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(OUTPUT/(name+'_response.npz'), density_derivatives=densities)
    print('Completed', report['elapsed_seconds'], 's; peak RSS', report['peak_rss_kib'], 'KiB', flush=True)
    return report


if __name__ == '__main__':
    run()

# Verified on jy_gpu_1, CPU float64, JAX 0.10.0 (2026-10-16):
# chi diagonal = 0.04487636014408; raw off-diagonal = 0.000237280363338.
# epsilon_inf diagonal = 1.563932973394; off-diagonal = 0.002981752985.
# Max linear residual 2.03e-15; raw chi asymmetry 1.35e-17.
# Full workflow 118.84 s, peak RSS 5,646,932 KiB, including compilation.
# Separate two-step finite-field check: all 12 endpoints passed.
# Max chi FD error 2.5344e-8 at h=1e-4; 2.2810e-9 at h=3e-5 a.u.
# Fixed FFT31/k2/SZV model; no symmetry projection or convergence claim.
# Matching P03 at FFT71 (same a/k/basis): epsilon_inf diagonal 1.563925410506,
# off-diagonal 0.00298153041610; full response 206.19 s / 18,960,976 KiB.
# Twelve FFT71 finite-field endpoints passed (separate checker), max chi FD
# error 2.53445e-8; 490.18 s / 45,789,376 KiB, within the 64 GiB RSS cap.
# FFT refinement alone does not establish basis or k/string convergence.
