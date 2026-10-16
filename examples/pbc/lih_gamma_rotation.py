"""A joint cell/atom rotation checks covariance of the raw Gamma action.

Rotating atoms alone at fixed lattice is not a periodic rigid-motion symmetry.
Here Cartesian atoms, row lattice and therefore absolute k/grid vectors rotate
together, while fractional atoms and k points stay fixed.
"""
import hashlib
import json
import resource
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import dfpt
from examples.pbc.lih_forces_stress import make_mean_field, diagnostics
from examples.pbc.lih_gamma_phonons import OUTPUT, LINEAR


def run(name='pbe_szv_k2_fft31'):
    started = perf_counter()
    native_path = OUTPUT/(name+'.json')
    native = json.loads(native_path.read_text())
    assert native['complete'] and native['passed'] and not native['asr_projected']
    axis = np.array([1., 2., 3.]); axis /= np.linalg.norm(axis)
    angle = .23
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    rotation = np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)
    np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-14, rtol=0)
    coordinates, lattice = np.asarray(native['coordinates_bohr']), np.asarray(native['lattice_bohr'])
    direction = np.array([[.3, -.2, .1], [-.4, .15, .2]])
    direction /= np.linalg.norm(direction)
    mean_field = make_mean_field(native['settings'], coordinates=coordinates@rotation.T,
                                 lattice=lattice@rotation.T).run()
    state = diagnostics(mean_field)
    fractional = np.asarray(mean_field.kpts)@np.asarray(mean_field.cell.lattice).T/(2*np.pi)
    np.testing.assert_allclose(fractional, native['fractional_kpoints'], atol=2e-14, rtol=0)
    problem = dfpt.atomic_problem(mean_field)
    respond = jax.jit(lambda u: problem.response(tangent=u, config=LINEAR))
    result = respond(jnp.asarray(direction@rotation.T))
    action = -np.asarray(result.values['forces'])
    phi = np.asarray(native['force_constants_real_hartree_per_bohr2'])
    expected = (phi@direction.ravel()).reshape(2, 3)@rotation.T
    errors = dict(energy_hartree=abs(float(mean_field.e_tot)-native['energy_hartree']),
                  force_constant_action_hartree_per_bohr2=float(np.max(abs(action-expected))))
    gates = dict(energy_hartree=1e-9, force_constant_action_hartree_per_bohr2=2e-8)
    passed = bool(result.converged) and all(errors[key] < gates[key] for key in errors)
    report = dict(complete=True, passed=passed, **state,
        native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rotation=rotation.tolist(), original_direction=direction.tolist(),
        rotated_action_hartree_per_bohr2=action.tolist(), expected_action_hartree_per_bohr2=expected.tolist(),
        errors=errors, gates=gates, linear_residual_norm=float(result.residual_norm),
        elapsed_seconds=perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Joint cell/atoms/k/grid rotation covariance; no atom-only rotation sum rule or ASR projection.')
    (OUTPUT/'rotation_covariance.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Joint rotation covariance:', passed, errors, flush=True)
    assert passed, errors
    return report


if __name__ == '__main__':
    run()

# Measured on jy_gpu_1, CPU float64, 2026-10-16.
# run()
# Joint rotation covariance: True {'energy_hartree': 0.0, 'force_constant_action_hartree_per_bohr2': 2.4875934645507414e-15}
