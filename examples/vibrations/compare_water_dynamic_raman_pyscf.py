"""Independent PySCF public A/B and SciPy complex resolvent for V06.

No GradSCF physics is imported. Every displaced point rebuilds and converges
PySCF HF, AO dipoles, full A/B and the complex three-component response.
"""
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, scf
from pyscf.tdscf.rhf import get_ab
from scipy.linalg import solve
from scipy.integrate import simpson


OUTPUT = Path('reproducibility/ad_showcase/V06_water_dynamic_raman')


def build_response(data, coordinates):
    molecule = gto.M(atom=list(zip(data['symbols'], coordinates)), basis=data['basis'],
                     unit='Bohr', cart=True, verbose=0)
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol, mean_field.conv_tol_grad = 1e-13, 1e-10
    mean_field.kernel()
    assert mean_field.converged
    a, b = get_ab(mean_field)
    nocc, nvir = a.shape[:2]
    dimension = nocc*nvir
    a, b = a.reshape(dimension, dimension), b.reshape(dimension, dimension)
    matrix = np.block([[a, b], [-b, -a]])
    c = mean_field.mo_coeff
    dipole = np.einsum('pi,xpq,qa->xia', c[:, :nocc], molecule.intor('int1e_r', comp=3), c[:, nocc:]).reshape(3, dimension).T
    rhs = np.concatenate([dipole, -dipole])

    def response(frequency, damping):
        shifted = matrix-(frequency+1j*damping)*np.eye(2*dimension)
        solution = solve(shifted, rhs, assume_a='gen')
        residual = float(np.linalg.norm(shifted@solution-rhs))
        assert residual < 1e-9
        return 2*dipole.T@(solution[:dimension]+solution[dimension:])

    return response


def main():
    started = perf_counter()
    native_path = OUTPUT/'gradscf.json'
    data = json.loads(native_path.read_text())
    assert data['complete']
    native = np.load(OUTPUT/'response.npz')
    for name in native.files:
        assert np.isfinite(native[name]).all(), name
    coordinates = np.asarray(data['coordinates_bohr'])
    response = build_response(data, coordinates)
    expected = np.array([[response(frequency, damping) for frequency in native['frequencies_hartree']]
                         for damping in native['dampings_hartree']])
    np.testing.assert_allclose(native['polarizability'], expected, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(native['static_polarizability'], response(0., 0.), atol=3e-8, rtol=3e-8)
    # The eta->0 limit is meaningful away from electronic poles. The pole
    # region instead retains explicit damping dependence, not a claimed
    # converged lifetime or a finite zero-damping resonance height.
    nonresonant_index = int(np.argmin(abs(native['frequencies_hartree']-.11)))
    nonresonant_frequency = float(native['frequencies_hartree'][nonresonant_index])
    zero_damping = response(nonresonant_frequency, 0.)
    damping_errors = [float(np.max(abs(curve[nonresonant_index]-zero_damping))) for curve in native['polarizability']]
    assert damping_errors[0] > damping_errors[1] > damping_errors[2]
    assert damping_errors[-1] < .3*damping_errors[0]
    raman_grid_checks = []
    frequencies = native['frequencies_hartree']
    for eta, curves in zip(native['dampings_hartree'], native['activities_angstrom4_amu']):
        areas = np.array([simpson(curves[::stride], x=frequencies[::stride], axis=0) for stride in [4, 2, 1]])
        relative = abs(areas[-1]-areas[-2])/abs(areas[-1])
        assert np.max(relative) < 2e-4
        raman_grid_checks.append(dict(damping_hartree=float(eta),
            samples=[len(frequencies[::stride]) for stride in [4, 2, 1]],
            integrated_activities=areas.tolist(), last_relative_changes=relative.tolist()))
    checks = []
    modes = np.asarray(data['mode_displacements'])
    for mode, direction in enumerate(modes, 1):
        for maximum_step in [3e-4, 1e-4]:
            step = maximum_step/np.max(abs(direction))
            plus, minus = build_response(data, coordinates+step*direction), build_response(data, coordinates-step*direction)
            selected = [row for row in data['geometry_fd_checks'] if row['mode'] == mode
                        and row['max_cartesian_step_bohr'] == maximum_step]
            for row in selected:
                frequency, damping = row['frequency_hartree'], row['damping_hartree']
                finite = (plus(frequency, damping)-minus(frequency, damping))/(2*step)
                derivative = np.asarray(row['derivative_real'])+1j*np.asarray(row['derivative_imag'])
                error = float(np.max(abs(derivative-finite)))
                checks.append(dict(mode=mode, max_cartesian_step_bohr=maximum_step,
                    frequency_hartree=frequency, damping_hartree=damping, maximum_derivative_error=error,
                    finite_difference_real=finite.real.tolist(), finite_difference_imag=finite.imag.tolist()))
                if maximum_step == 1e-4:
                    np.testing.assert_allclose(derivative, finite, atol=3e-5, rtol=3e-4)
    record = dict(passed=True, native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        native_npz_sha256=hashlib.sha256((OUTPUT/'response.npz').read_bytes()).hexdigest(),
        source='PySCF public get_ab plus SciPy complex solve, complete occupied/virtual space',
        maximum_polarizability_error=float(np.max(abs(native['polarizability']-expected))),
        static_polarizability_error=float(np.max(abs(native['static_polarizability']-response(0., 0.)))),
        nonresonant_damping_limit=dict(frequency_hartree=nonresonant_frequency,
            damping_hartree=native['dampings_hartree'].tolist(),
            maximum_errors_vs_independent_zero_damping=damping_errors),
        raman_frequency_grid_checks=raman_grid_checks,
        mode_derivative_checks=checks, elapsed_seconds=perf_counter()-started,
        pyscf_version=pyscf.__version__, python=platform.python_version(), hostname=platform.node(),
        gates=dict(polarizability_atol=2e-7, polarizability_rtol=2e-7,
                   mode_derivative_atol=3e-5, mode_derivative_rtol=3e-4))
    (OUTPUT/'pyscf.json').write_text(json.dumps(record, indent=2)+'\n')
    print('V06 independent PySCF full-grid and displaced response gates PASS', flush=True)
    print('maximum alpha and finest mode errors:', record['maximum_polarizability_error'],
          max(row['maximum_derivative_error'] for row in checks if row['max_cartesian_step_bohr'] == 1e-4), flush=True)


if __name__ == '__main__':
    main()
