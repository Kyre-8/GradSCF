"""Periodic H2: free-energy curvatures versus electronic temperature.

Differentiate the fixed-N Mermin free energy at positive kBT. Cell, geometry,
basis and FFT grid stay fixed. This isolates electronic occupation response;
it does not include expansion, nuclear temperature or anharmonic vibrations.
The coarse, unoptimized H2 model is a numerical check, not stable bulk phonons.
"""
from pathlib import Path
from time import perf_counter
import json
import resource
import hashlib
import platform

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)
from gradscf.pbc import gto, scf
from gradscf.pbc.scf import run_kpoint_scf
from gradscf.integrals.periodic.fft import build_inputs
from gradscf.scf import SCFDifferentiationConfig
from gradscf.dfpt import atomic_problem
from gradscf.solvers import LinearSolverConfig, EigenSolverConfig, solve_hermitian
from scipy.constants import physical_constants



def run():
    started = perf_counter()
    cell = gto.M(atom='H .4 .5 .6; H 1.8 .8 .9', a=np.eye(3)*6.,
                 basis='gth-szv', pseudo='gth-pade', unit='Bohr', mesh=(9,)*3)
    kbt = .05  # Hartree; electronic occupations only, fixed spin populations.
    differentiation = SCFDifferentiationConfig(mode='implicit', tolerance=1e-11,
                                              max_iter=100, restart=20)
    controls = dict(conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-11,
                    max_cycle=150, damping=.1)
    mf = scf.RHF(cell, temperature=kbt, differentiation=differentiation, **controls).run()
    assert mf.converged

    def state(temperature, displacement):
        inputs = build_inputs(cell, parameters=cell.parameters_at(cell.coords+displacement.reshape(2, 3)))
        return run_kpoint_scf(inputs, mesh=cell.mesh, nelec=cell.nelec, xc='hf', exxdiv='ewald',
                             temperature=temperature, differentiation=differentiation, **controls)

    def free_energy(temperature, displacement):
        return state(temperature, displacement).free_energy

    zero = jnp.zeros(6)
    hessian = jax.jit(jax.jacfwd(jax.grad(free_energy, argnums=1), argnums=1))
    base = np.asarray(hessian(jnp.array(kbt), zero))
    problem = atomic_problem(mf)
    linear = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-12, max_dense=64)
    respond = jax.jit(lambda u:problem.response(tangent=u.reshape(2, 3), config=linear))
    columns = []
    for direction in jnp.eye(6):
        result = respond(direction)
        assert result.converged
        columns.append(-np.asarray(result.values['forces']).ravel())
    dfpt_error = float(np.max(abs(base-np.stack(columns, axis=1))))
    assert dfpt_error < 2e-8, dfpt_error
    derivative = np.asarray(jax.jit(lambda t:jax.jvp(lambda u:hessian(u, zero),
                                                    (t,), (jnp.ones_like(t),))[1])(jnp.array(kbt)))
    mass_ratio = physical_constants['atomic mass constant'][0] / physical_constants['electron mass'][0]
    mass = np.repeat(np.full(2, 1.00782503223)*mass_ratio, 3)
    mass_scale = np.sqrt(mass[:, None]*mass[None, :])
    roots = solve_hermitian(jnp.asarray(base/mass_scale), config=EigenSolverConfig(
        method='dense', nroots=6, atol=1e-12))
    assert np.all(roots.converged)
    values, modes = np.asarray(roots.values), np.asarray(roots.vectors)
    assert values[-1] > 0 and values[-1]-values[-2] > 1e-5*values[-1]
    frequency_slope = float(modes[:, -1] @ (derivative/mass_scale) @ modes[:, -1])/(2*np.sqrt(values[-1]))
    records = []
    for step in (1e-3, 3e-4, 1e-4):
        plus = np.asarray(hessian(jnp.array(kbt+step), zero))
        minus = np.asarray(hessian(jnp.array(kbt-step), zero))
        finite = (plus-minus)/(2*step)
        # Independent numerical diagonalization of the reconverged Hessians.
        plus_values, plus_modes = np.linalg.eigh(plus/mass_scale)
        minus_values, minus_modes = np.linalg.eigh(minus/mass_scale)
        assert abs(plus_modes[:, -1] @ modes[:, -1]) > .99
        assert abs(minus_modes[:, -1] @ modes[:, -1]) > .99
        finite_frequency = (np.sqrt(plus_values[-1])-np.sqrt(minus_values[-1]))/(2*step)
        records.append(dict(step_kbt_hartree=step, max_error_bohr_minus2=float(np.max(abs(derivative-finite))),
                            frequency_slope_error=abs(finite_frequency-frequency_slope)))
        print(records[-1], flush=True)
    assert records[-1]['max_error_bohr_minus2'] < 2e-6, records
    assert records[-1]['frequency_slope_error'] < 2e-8, records
    free_slope = float(jax.jit(jax.grad(lambda t:free_energy(t, zero)))(jnp.array(kbt)))
    np.testing.assert_allclose(free_slope, -float(mf.entropy), atol=1e-10, rtol=0)
    np.testing.assert_allclose(float(mf.result.free_energy),
                              float(mf.result.total_energy-kbt*mf.result.entropy), atol=1e-12)
    np.testing.assert_allclose(np.sum(mf.result.mo_occ_spin, axis=(-1, -2)), cell.nelec, atol=1e-10)
    output = Path('reproducibility/ad_showcase/P07_periodic_temperature')
    output.mkdir(parents=True, exist_ok=True)
    data = dict(complete=True, passed=True, kbt_hartree=kbt, dfpt_error=dfpt_error,
        coordinates_bohr=np.asarray(cell.coords).tolist(), lattice_bohr=np.asarray(cell.lattice).tolist(),
        basis_data=cell.basis_spec, pseudopotential_data=cell.pseudopotentials, mesh=list(cell.mesh),
        controls=controls, explicit_electrons=cell.nelectron, hostname=platform.node(), jax=jax.__version__,
        raw_dynamical_eigenvalues=values.tolist(), selected_frequency_hartree=float(np.sqrt(values[-1])),
        selected_frequency_derivative_kbt=frequency_slope, free_energy_derivative_kbt=free_slope,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        free_energy_hartree=float(mf.e_free), total_energy_hartree=float(mf.e_tot),
        entropy_over_kb=float(mf.entropy), occupations=np.asarray(mf.result.mo_occ_spin).tolist(),
        hessian_hartree_bohr2=base.tolist(), temperature_derivative_bohr_minus2=derivative.tolist(),
        finite_differences=records, elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Fixed-cell/fixed-geometry periodic H2 Gamma/FFT9 finite model; fixed-N Mermin free energy. Not thermal expansion, nuclear anharmonicity or stable material phonons.')
    (output/'gradscf.json').write_text(json.dumps(data, indent=2)+'\n')
    print('PASS',data['elapsed_seconds'],data['peak_rss_kib'],flush=True)

    return data


if __name__ == "__main__":
    run()

# Verified on jy_gpu_1, CPU float64/JAX 0.10.0 (2026-10-16), kBT=0.05 Eh:
# Free-energy Hessian vs DFPT: max error 2.04e-13 Eh/bohr^2.
# H_kBT FD errors at 1e-3/3e-4/1e-4 Eh: 1.739e-5 / 1.566e-6 / 1.739e-7 bohr^-2.
# F_kBT = -0.000229542437073; -S/kB = -0.000229542440544.
# Isolated local omega_kBT = -9.88168908582e-5; final root FD error 3.91e-9.
# Workflow 95.54 s, peak RSS 2,519,812 KiB. Three raw unstable curvatures remain.
# Electronic-temperature response at fixed geometry/cell, not thermal expansion.
