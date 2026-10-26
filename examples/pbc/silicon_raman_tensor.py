"""Raw Si Raman tensors and matched Gamma phonons in one finite model.

Run the single-column AD/gauge/finite-difference checks before this expansion.
The electronic and vibrational models use exactly the same cell, basis, k
mesh and FFT mesh. No acoustic or crystal-symmetry projection is applied.
The resulting mode-resolved polarization weights are not absolute cross
sections or a basis/k/FFT-converged experimental spectrum.
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
from scipy.constants import physical_constants, h, c

from gradscf.pbc.phonon import Phonons
from silicon_raman import (OUTPUT, SETTINGS, MASSES_AMU, AMU_TO_ME, LINEAR,
                           mean_field, mixed_problem)


def provenance():
    files = ['examples/pbc/silicon_raman.py',
             'examples/pbc/silicon_raman_tensor.py',
             'src/gradscf/dfpt/electric.py',
             'src/gradscf/pbc/polarization.py',
             'src/gradscf/pbc/phonon.py']
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}


def save(path, values):
    """Keep each completed stage if a later bounded job is interrupted."""
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(values, indent=2)+'\n')
    temporary.replace(path)


def execution_record(started):
    return dict(elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hostname=platform.node(), python=platform.python_version(),
        jax=jax.__version__, backend=jax.default_backend(), dtype='float64')


def require_checks(name, settings):
    suffixes = ('_ad_checks', '_gauge', '_fd_0.0003', '_fd_0.0001')
    for suffix in suffixes:
        result = json.loads((OUTPUT/(name+suffix+'.json')).read_text())
        if not (result['complete'] and result['validation_passed']
                and result['settings'] == settings):
            raise RuntimeError('Complete the matched AD, gauge and two-step FD checks first.')


def run_phonons(name='gamma_fft15', settings=None):
    settings = SETTINGS if settings is None else settings
    require_checks(name, settings)
    started = perf_counter()
    mf = mean_field(settings)
    assert mf.converged
    phonons = Phonons(mf, masses=MASSES_AMU, max_grid=100_000,
                      max_pair_elements=200_000_000)
    result = phonons.kernel(method='primitive', config=LINEAR, asr=False)
    jax.block_until_ready(result)
    phi = np.asarray(result.force_constants)
    modes = np.asarray(result.modes)
    # Gamma real displacement vectors are needed for the mixed derivative.
    # A complex result is diagnosed, never silently replaced by its real part.
    imaginary_error = float(max(np.max(abs(phi.imag)), np.max(abs(modes.imag))))
    if imaginary_error > 1e-8:
        raise RuntimeError('Gamma force constants or modes are not real.')
    conversion = physical_constants['Hartree energy'][0]/(h*c*100)
    frequencies = np.asarray(result.frequencies)
    values = dict(complete=True, settings=settings, source_sha256=provenance(),
        force_constants_eh_bohr2=phi.real.tolist(),
        mass_weighted_modes=modes.real.tolist(), frequencies_hartree=frequencies.tolist(),
        signed_frequencies_cm1=(frequencies*conversion).tolist(),
        raw_asr_residual=float(result.asr_residual),
        raw_hermiticity_error=float(result.hermiticity_error),
        imaginary_error=imaginary_error, response_residual=float(result.residual_norm),
        reference_forces_eh_bohr=np.asarray(phonons.reference_forces).tolist(),
        energy_hartree=float(mf.e_tot), volume_bohr3=float(mf.cell.volume),
        masses_amu=MASSES_AMU.tolist(), amu_to_me=AMU_TO_ME,
        **execution_record(started),
        scope='Raw Gamma Hessian of the declared fixed-geometry finite model; no ASR projection.')
    save(OUTPUT/(name+'_phonons.json'), values)
    print('Raw Gamma frequencies (cm^-1)', values['signed_frequencies_cm1'], flush=True)
    print('Raw ASR', values['raw_asr_residual'], 'response residual', values['response_residual'], flush=True)
    return values


def run_tensors(name='gamma_fft15', settings=None, *, resume=True):
    """Differentiate all six Cartesian coordinates, then contract with modes.

    Each Cartesian direction is mass scaled so that its coordinate Q has
    units sqrt(me)*bohr. A mode tensor is the linear combination specified by
    its mass-weighted unit eigenvector; mixing inside a degenerate subspace
    changes individual tensors but preserves the summed polarization weight.
    Set resume=False for a fresh calculation after recomputing the phonons;
    a resumed calculation checks the exact saved model and source hashes.
    """
    settings = SETTINGS if settings is None else settings
    require_checks(name, settings)
    started = perf_counter()
    target = OUTPUT/(name+'_tensors.json')
    phonons = json.loads((OUTPUT/(name+'_phonons.json')).read_text())
    if not phonons['complete'] or phonons['settings'] != settings:
        raise RuntimeError('Compute matched phonons before the Raman tensors.')
    mf = mean_field(settings)
    assert mf.converged
    if abs(float(mf.e_tot)-phonons['energy_hartree']) > 1e-10:
        raise RuntimeError('The Raman and phonon reference states differ.')
    hashes = provenance()
    report = dict(complete=False, settings=settings, source_sha256=hashes,
        phonon_sha256=hashlib.sha256((OUTPUT/(name+'_phonons.json')).read_bytes()).hexdigest(),
        energy_hartree=float(mf.e_tot), volume_bohr3=float(mf.cell.volume), columns={})
    if resume and target.exists():
        previous = json.loads(target.read_text())
        for key in ('settings', 'source_sha256', 'phonon_sha256'):
            if previous[key] != report[key]:
                raise RuntimeError('A cached Raman column belongs to a different source/model.')
        report = previous
    save(target, report)
    _, _, respond = mixed_problem(mf)
    inverse_mass = 1/np.repeat(np.sqrt(MASSES_AMU*AMU_TO_ME), 3)
    directions = (np.eye(6)*inverse_mass).reshape(6, 2, 3)
    for coordinate, displacement in enumerate(directions):
        for field in range(3):
            key = f'{coordinate},{field}'
            if key in report['columns']:
                continue
            before = perf_counter()
            result = respond(jnp.asarray(displacement), jnp.eye(3)[field])
            jax.block_until_ready(result)
            column = {key: np.asarray(value).tolist() for key, value in result.items()}
            residuals = [column[key] for key in
                         ('reference_residual', 'field_residual', 'mixed_residual')]
            if (not np.isfinite(column['derivative']).all()
                    or not np.isfinite(residuals).all() or max(residuals) >= 1e-10):
                raise RuntimeError('A Raman column failed its true response residual.')
            column['seconds'] = perf_counter()-before
            report['columns'][key] = column
            save(target, report)
            print('Cartesian/field', key, 'derivative', column['derivative'], flush=True)
    derivatives = np.array([[report['columns'][f'{a},{b}']['derivative']
                             for b in range(3)] for a in range(6)]).swapaxes(-1, -2)
    modes = np.asarray(phonons['mass_weighted_modes'])
    tensors = np.einsum('aij,av->vij', derivatives, modes)
    # Static response follows a scalar enthalpy. Reciprocity is tested on the
    # raw tensor; the output is not symmetrized to improve the reported error.
    reciprocity = float(np.max(abs(tensors-tensors.swapaxes(-1, -2))))
    report.update(complete=True, cartesian_mass_scaled_dchi=derivatives.tolist(),
        mode_dchi_dQ=tensors.tolist(), mode_dalpha_dQ=(float(mf.cell.volume)*tensors).tolist(),
        reciprocity_error=reciprocity, reciprocity_passed=reciprocity < 1e-8,
        signed_frequencies_cm1=phonons['signed_frequencies_cm1'],
        **execution_record(started),
        scope='Raw mode tensors of the matched finite model. Crystal selection rules and convergence are reported separately.')
    save(target, report)
    assert report['reciprocity_passed'], 'Static Raman tensor reciprocity failed.'
    return report


if __name__ == '__main__':
    run_phonons()
    run_tensors(resume=False)
