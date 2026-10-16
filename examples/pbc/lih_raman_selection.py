"""An inversion-forbidden Raman column in a small rocksalt LiH model.

This uses the P03/P04 lattice, pseudopotentials and isotope masses, but an
explicitly separate FFT15 grid. No FFT71 phonon frequency is assigned to this
model. The optical displacement is normalized in electron-mass atomic units.
The Si example supplies the same electric-enthalpy mixed-response assembly;
all SCF, geometry response and linear solves remain GradSCF calculations.
"""
import hashlib
import json
import resource
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.dfpt.electric import electric_problem
from lih_forces_stress import SETTINGS as FORCE_SETTINGS, make_mean_field, diagnostics
from silicon_raman import AMU_TO_ME, LINEAR, mixed_problem


OUTPUT = Path('reproducibility/ad_showcase/P06_silicon_raman')
SETTINGS = dict(FORCE_SETTINGS, a_bohr=7.780032276700287, mesh=[15]*3, kmesh=[2]*3,
                displacement_bohr=[[0., 0., 0.], [0., 0., 0.]])
MASSES_AMU = np.array([7.0160034366, 1.00782503223])  # 7Li and 1H, as in P03/P04.


def run():
    started = perf_counter()
    mf = make_mean_field(SETTINGS).run()
    reference = diagnostics(mf)
    # In mass-weighted coordinates, this optical vector is perpendicular to
    # the rigid translation (sqrt(M_Li), sqrt(M_H)) and has unit Euclidean norm.
    vector = np.zeros((2, 3))
    vector[:, 2] = np.array([np.sqrt(MASSES_AMU[1]), -np.sqrt(MASSES_AMU[0])])/np.sqrt(sum(MASSES_AMU))
    displacement = vector/np.sqrt(MASSES_AMU[:, None]*AMU_TO_ME)
    field = jnp.array([0., 1., 0.])
    _, _, respond = mixed_problem(mf)
    result = respond(jnp.asarray(displacement), field)
    jax.block_until_ready(result)
    observed = {key: np.asarray(value).tolist() for key, value in result.items()}
    residuals = [observed[key] for key in
                 ('reference_residual', 'field_residual', 'mixed_residual')]
    ad_error = float(np.max(abs(np.asarray(observed['derivative']))))
    print('LiH inversion-forbidden mixed column', observed, flush=True)
    assert np.isfinite(residuals).all() and max(residuals) < 1e-10
    assert ad_error < 1e-8, 'Raw inversion-forbidden Raman derivative is too large.'

    # Reconverge the two inversion-related structures independently. This is
    # a symmetry/FD check of chi, never the source of the reported AD slope.
    maximum_step = 1e-4
    qstep = maximum_step/float(np.max(abs(displacement)))
    endpoints = []
    for sign in (1., -1.):
        coordinates = np.asarray(mf.cell.coords)+sign*qstep*displacement
        displaced = make_mean_field(SETTINGS, coordinates=coordinates,
                                    lattice=np.asarray(mf.cell.lattice)).run()
        diagnostics(displaced)
        response = electric_problem(displaced).response(tangent=field, config=LINEAR)
        jax.block_until_ready(response)
        assert response.converged and float(response.residual_norm) < 1e-10
        endpoints.append(np.asarray(response.values['polarization']))
    finite = (endpoints[0]-endpoints[1])/(2*qstep)
    fd_error = float(np.max(abs(finite-np.asarray(observed['derivative']))))
    passed = fd_error < 2e-8
    record = dict(complete=True, passed=passed, settings=SETTINGS, **reference,
        masses_amu=MASSES_AMU.tolist(), displacement_per_Q=displacement.tolist(),
        mass_norm=float(np.sum(MASSES_AMU[:, None]*AMU_TO_ME*displacement**2)),
        center_of_mass_derivative=np.sum(MASSES_AMU[:, None]*displacement, axis=0).tolist(),
        response=observed, raw_forbidden_maximum=ad_error,
        maximum_cartesian_step_bohr=maximum_step, qstep_sqrt_me_bohr=qstep,
        reconverged_chi_columns=np.asarray(endpoints).tolist(),
        finite_difference=finite.tolist(), maximum_fd_error=fd_error,
        elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ['examples/pbc/lih_raman_selection.py', 'examples/pbc/silicon_raman.py',
             'src/gradscf/dfpt/electric.py', 'src/gradscf/pbc/polarization.py']},
        scope='Inversion selection rule for this FFT15 finite model; no FFT71 frequency or material-convergence claim.')
    (OUTPUT/'lih_k2_fft15_selection.json').write_text(json.dumps(record, indent=2)+'\n')
    assert passed, 'Independent inversion-related chi responses disagree with the AD zero.'
    print('LiH raw forbidden maximum', ad_error, 'FD error', fd_error, flush=True)
    return record


if __name__ == '__main__':
    run()
