"""Raw LiH force response at q=(0,0,1/2), with an independent two-cell gate.

The fixed P02 geometry, PBE/GTH-Pade/SZV and k2^3 sampling match the Gamma
example. A bounded FFT31 one-column probe precedes a full matrix or finer grid.
q is fractional reciprocal; phases contain lattice translations, not atom
positions. This self-inverse point does not test general complex q/-q pairs.
"""
import hashlib
import json
import resource
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf.dfpt.force_constants import primitive_force_response
from gradscf.pbc.momentum import momentum_transfer, wrap_to_kmesh
from examples.pbc.lih_forces_stress import make_mean_field
from examples.pbc.lih_gamma_phonons import LINEAR, SETTINGS, _record


OUTPUT = Path('reproducibility/ad_showcase/P03_lih_zone_boundary')
Q = np.array([0., 0., .5])
RESPONSE = replace(LINEAR, method='gmres', restart=40, maxiter=100)
GATES = dict(linear_residual_norm=1e-9, imaginary_hartree_per_bohr2=1e-8,
             hermiticity_hartree_per_bohr2=1e-8)


def run(name='pbe_szv_k2_fft31_qhalf_probe', *, settings=None, probe=True,
        max_grid=30000, max_pair_elements=20000000):
    """Use the existing primitive response; retain raw columns before gates."""
    started = perf_counter()
    settings = dict(SETTINGS if settings is None else settings)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/(name+'.json')
    if path.exists():
        raise FileExistsError('Choose a new case name; retain earlier probe results.')
    path.write_text('{"complete": false}\n')
    mean_field = make_mean_field(settings).run()
    record = _record(mean_field, settings)
    grid = int(np.prod(settings['mesh']))
    pairs = grid*record['nao']**2*len(mean_field.kpts)**2
    if grid > max_grid or pairs > max_pair_elements:
        raise ValueError('The declared primitive response workspace is too small.')
    canonical = wrap_to_kmesh(Q)
    transfer = momentum_transfer(record['fractional_kpoints'], canonical)
    # Match Phonons' canonical representative. +1/2 and -1/2 are the same
    # point here, so evaluating the wrapped pair twice is not an extra test.
    forces, respond = primitive_force_response(mean_field, canonical, RESPONSE)
    directions = np.eye(6).reshape(6, 2, 3)
    if probe:
        directions = directions[3:4]  # Unit H x displacement in the +q cell.
    columns, residuals, solve_flags, elapsed = [], [], [], []
    before = perf_counter()
    for direction in directions:
        tick = perf_counter()
        column, residual, converged = respond(jnp.asarray(direction))
        columns.append(np.asarray(column))
        residuals.append(float(residual)); solve_flags.append(bool(converged))
        elapsed.append(perf_counter()-tick)
    matrix = np.stack(columns, axis=1)
    response_seconds = perf_counter()-before
    errors = dict(linear_residual_norm=max(residuals),
                  imaginary_hartree_per_bohr2=float(np.max(abs(matrix.imag))))
    if not probe:
        errors['hermiticity_hartree_per_bohr2'] = float(np.max(abs(matrix-matrix.conj().T)))
    passed = all(solve_flags) and np.isfinite(matrix).all() and all(
        errors[key] < GATES[key] for key in errors)
    record.update(complete=True, passed=bool(passed), probe=probe,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gamma_helper_sha256=hashlib.sha256(Path(__file__).with_name('lih_gamma_phonons.py').read_bytes()).hexdigest(),
        scope='Raw self-inverse q=(0,0,1/2) response only; no general conjugacy, interpolation or material convergence claim.',
        q_requested=Q.tolist(), q_evaluated=canonical.tolist(),
        k_partners=transfer.partners.tolist(), reciprocal_shifts=transfer.shifts.tolist(),
        response_dimension=4*len(mean_field.kpts)*record['nao']**2,
        directions=directions.tolist(), force_constant_actions_real_hartree_per_bohr2=matrix.real.tolist(),
        force_constant_actions_imag_hartree_per_bohr2=matrix.imag.tolist(),
        reference_forces_hartree_per_bohr=np.asarray(forces).tolist(),
        linear_solver=asdict(RESPONSE), linear_residuals=residuals, solve_converged=solve_flags,
        errors=errors, gates={key:GATES[key] for key in errors}, asr_projected=False,
        workspace_limits=dict(max_grid=max_grid, max_pair_elements=max_pair_elements),
        workspace_grid_points=grid, workspace_pair_elements=pairs,
        response_seconds_includes_compilation=response_seconds, action_seconds=elapsed,
        elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    np.savez_compressed(OUTPUT/(name+'.npz'), actions=matrix, directions=directions,
                        density=np.asarray(mean_field.make_rdm1()))
    path.write_text(json.dumps(record, indent=2)+'\n')
    print('LiH q-half response:', bool(passed), errors, flush=True)
    print('Total seconds / peak KiB:', record['elapsed_seconds'], record['peak_rss_kib'], flush=True)
    assert passed, errors
    return record


if __name__ == '__main__':
    run()

# Measured FFT31 H x probe on jy_gpu_1, CPU float64, 2026-10-16:
# LiH q-half response: True {'linear_residual_norm': 6.913479860987118e-13,
#                          'imaginary_hartree_per_bohr2': 6.552686363223365e-10}
# Total seconds / peak KiB: 130.91669790699962 8302656
# This one-column result does not establish full Hermiticity or a dispersion.
