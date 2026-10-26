"""Raw Gamma phonons at the validated finite-model LiH cubic minimum.

PBE / GTH-Pade / GTH-SZV, four explicit electrons, k=2^3, FFT=31^3.
The scalar lattice minimum does not imply full crystal stability. No ASR
projection or absolute-value replacement is applied to the signed frequencies.
Run from the repository root with the root and src on PYTHONPATH.
"""
import hashlib
import json
import platform
import resource
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import physical_constants, h, c

jax.config.update('jax_enable_x64', True)

from gradscf import dfpt
from gradscf.pbc.phonon import Phonons
from gradscf.solvers import LinearSolverConfig
from examples.pbc.lih_forces_stress import SETTINGS as FORCE_SETTINGS
from examples.pbc.lih_forces_stress import diagnostics, make_mean_field


OUTPUT = Path('reproducibility/ad_showcase/P03_lih_gamma_phonons')
SETTINGS = dict(FORCE_SETTINGS, a_bohr=7.780032276700287,
                displacement_bohr=[[0., 0., 0.], [0., 0., 0.]])
# NIST relative atomic masses, 7Li and 1H, in unified atomic mass units.
MASSES_AMU = np.array([7.0160034366, 1.00782503223])
MASS_SOURCES = ['https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=Li',
                'https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=H']
LINEAR = LinearSolverConfig(method='direct', rtol=1e-10, atol=1e-12, max_dense=256)
TO_CM1 = physical_constants['Hartree energy'][0]/(h*c*100)


def _record(mean_field, settings):
    """Keep the actual periodic model and runtime source with each result."""
    source = hashlib.sha256()
    for path in sorted(Path('src').rglob('*.py')):
        source.update(path.as_posix().encode()); source.update(path.read_bytes())
    return dict(settings=settings, **diagnostics(mean_field),
        energy_hartree=float(mean_field.e_tot),
        coordinates_bohr=np.asarray(mean_field.cell.coords).tolist(),
        lattice_bohr=np.asarray(mean_field.cell.lattice).tolist(),
        fractional_kpoints=(np.asarray(mean_field.kpts)@np.asarray(mean_field.cell.lattice).T/(2*np.pi)).tolist(),
        basis_data=mean_field.cell.basis_spec, pseudopotential_data=mean_field.cell.pseudopotentials,
        masses_amu=MASSES_AMU.tolist(), mass_sources=MASS_SOURCES,
        source_python_sha256=source.hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).with_name('lih_forces_stress.py').read_bytes()).hexdigest(),
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__, dtype='float64',
        scope='Gamma local curvatures of the specified finite periodic model; no ASR or stability assumption.')


def probe(name='gamma_probe'):
    """One force-constant action before allocating the full six-column result."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/(name+'.json'); path.write_text('{"complete": false}\n')
    started = perf_counter()
    mean_field = make_mean_field(SETTINGS).run()
    record = _record(mean_field, SETTINGS)
    problem = dfpt.atomic_problem(mean_field)
    direction = jnp.zeros_like(mean_field.cell.coords).at[1, 0].set(1.)
    respond = jax.jit(lambda u: problem.response(tangent=u, config=LINEAR))
    reference_seconds = perf_counter()-started
    before = perf_counter()
    result = respond(direction)
    column = -np.asarray(result.values['forces'])
    cold_seconds = perf_counter()-before
    assert bool(result.converged) and np.isfinite(column).all()
    before = perf_counter()
    warm = -np.asarray(respond(direction).values['forces'])
    warm_seconds = perf_counter()-before
    np.testing.assert_allclose(warm, column, atol=1e-12, rtol=0)
    record.update(complete=True, passed=True, response_dimension=int(problem.solution.size),
        direction=np.asarray(direction).tolist(), force_constant_action_hartree_per_bohr2=column.tolist(),
        linear_residual_norm=float(result.residual_norm), reference_seconds=reference_seconds,
        cold_action_seconds=cold_seconds, warm_action_seconds=warm_seconds,
        elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    path.write_text(json.dumps(record, indent=2)+'\n')
    print('Gamma action probe complete:', record['response_dimension'], 'variables;',
          record['elapsed_seconds'], 's;', record['peak_rss_kib'], 'KiB', flush=True)
    return record


def run(name='pbe_szv_k2_fft31', settings=None, *, max_grid=30_000,
        max_pair_elements=20_000_000, linear=LINEAR):
    """Existing DFPT/phonon drivers, followed by raw diagnostics and storage."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/(name+'.json'); path.write_text('{"complete": false}\n')
    started = perf_counter()
    settings = dict(SETTINGS if settings is None else settings)
    mean_field = make_mean_field(settings).run()
    record = _record(mean_field, settings)
    # The declared primitive workspace is 31^3 * 3^2 * 8^2 = 17,159,616
    # AO-pair elements. These ceilings are checks, not memory preallocations.
    phonons = Phonons(mean_field, masses=MASSES_AMU, max_grid=max_grid,
                       max_pair_elements=max_pair_elements)
    before = perf_counter()
    result = phonons.kernel(asr=False, config=linear, method='primitive')
    phonon_seconds = perf_counter()-before
    phi = np.asarray(result.force_constants)
    matrix, modes = np.asarray(result.dynamical_matrix), np.asarray(result.modes)
    frequencies = np.asarray(result.frequencies)
    forces = np.asarray(phonons.reference_forces)
    translations = np.tile(np.eye(3), (2, 1))
    mass_ratio = physical_constants['atomic mass constant'][0]/physical_constants['electron mass'][0]
    mass = np.repeat(MASSES_AMU*mass_ratio, 3)
    np.testing.assert_allclose(matrix, .5*(phi+phi.conj().T)/np.sqrt(mass[:, None]*mass[None, :]), atol=1e-13, rtol=0)
    np.testing.assert_allclose(modes.conj().T@modes, np.eye(6), atol=1e-10, rtol=0)
    eigenvalues = np.sign(frequencies)*frequencies**2
    eigen_residual = float(np.linalg.norm(matrix@modes-modes*eigenvalues))
    assert bool(result.converged) and np.isfinite(phi).all() and eigen_residual < 1e-10
    assert np.max(abs(forces)) < 3e-6
    record.update(complete=True, passed=True,
        force_constants_real_hartree_per_bohr2=phi.real.tolist(),
        force_constants_imag_hartree_per_bohr2=phi.imag.tolist(),
        frequencies_signed_hartree=frequencies.tolist(), frequencies_signed_cm1=(frequencies*TO_CM1).tolist(),
        reference_forces_hartree_per_bohr=forces.tolist(),
        reference_net_force_hartree_per_bohr=forces.sum(axis=0).tolist(),
        reference_torque_hartree=np.cross(np.asarray(mean_field.cell.coords), forces).sum(axis=0).tolist(),
        raw_translation_action_hartree_per_bohr2=(phi@translations).real.tolist(),
        raw_asr_residual_hartree_per_bohr2=float(result.asr_residual),
        hermiticity_error_hartree_per_bohr2=float(result.hermiticity_error),
        linear_residual_norm=float(result.residual_norm), eigen_residual_hartree2=eigen_residual,
        atomic_mass_to_electron_mass=mass_ratio, hartree_to_wavenumber=TO_CM1,
        asr_projected=False, phonon_seconds_includes_compilation=phonon_seconds,
        workspace_limits=dict(max_grid=max_grid, max_pair_elements=max_pair_elements),
        linear_solver=asdict(linear),
        rotation_convention='Atom-only rotation at fixed lattice is not a periodic symmetry; no molecular rotation sum rule is imposed.',
        elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    np.savez_compressed(OUTPUT/(name+'.npz'), force_constants=phi, dynamical_matrix=matrix,
        modes=modes, frequencies_signed_hartree=frequencies, masses_amu=MASSES_AMU,
        density=np.asarray(mean_field.make_rdm1()))
    path.write_text(json.dumps(record, indent=2)+'\n')
    print('Raw Gamma frequencies (cm^-1):', frequencies*TO_CM1, flush=True)
    print('Raw ASR residual (Eh/bohr^2):', record['raw_asr_residual_hartree_per_bohr2'], flush=True)
    return record


if __name__ == '__main__':
    run()

# Measured on jy_gpu_1, CPU float64, 2026-10-16.
# run('pbe_szv_k2_fft71_gmres', dict(SETTINGS, mesh=[71]*3),
#     max_grid=360000, max_pair_elements=210000000,
#     linear=replace(LINEAR, method='gmres', restart=40))
# Raw Gamma frequencies (cm^-1): [2.31253279e-01 2.31253343e-01 2.31253388e-01 4.41410297e+02
#  4.41410297e+02 4.41410297e+02]
# Raw ASR residual (Eh/bohr^2): 3.596611608262572e-08
