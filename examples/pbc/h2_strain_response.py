"""Periodic H2: differentiate a raw harmonic curvature with log cell volume.

A small Gamma/FFT9 RHF model validates the numerical chain. Fractional atoms
and k points stay fixed under strain; Cartesian phonon displacements do not
scale with the cell. This is not a converged bulk crystal or a temperature
shift. The energy Hessian is independently assembled with the existing DFPT
force-response driver before its strain derivative is used.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import resource

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)
from gradscf.pbc import gto, scf
from gradscf.pbc.scf import run_gamma_scf
from gradscf.integrals.periodic.fft import build_inputs
from gradscf.scf import SCFDifferentiationConfig
from gradscf.dfpt import atomic_problem
from gradscf.solvers import LinearSolverConfig, EigenSolverConfig, solve_hermitian
from scipy.constants import physical_constants


def run():
    started = perf_counter()
    output = Path('reproducibility/ad_showcase/P07_periodic_strain')
    output.mkdir(parents=True, exist_ok=True)
    # Cell defaults are GTH-SZV/GTH-Pade; the exact assets are saved below.
    cell = gto.M(atom='H .4 .5 .6; H 1.8 .8 .9', a=np.eye(3)*6., unit='Bohr', mesh=(9,)*3)
    config = SCFDifferentiationConfig(mode='implicit', tolerance=1e-11, max_iter=100, restart=20)
    controls = dict(conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-11,
                    max_cycle=150, damping=.1)
    mf = scf.RHF(cell, differentiation=config, **controls).run()
    assert mf.converged

    def state(log_volume, displacement):
        scale = jnp.exp(log_volume/3)
        lattice = scale*cell.lattice
        coordinates = scale*cell.coords+displacement.reshape(cell.coords.shape)
        inputs = build_inputs(cell, parameters=cell.parameters_at(coordinates), lattice=lattice)
        return run_gamma_scf(inputs, nelec=cell.nelec, xc='hf', exxdiv='ewald',
                             differentiation=config, mesh=cell.mesh, **controls)

    def energy(log_volume, displacement):
        return state(log_volume, displacement).free_energy

    zero = jnp.zeros(cell.coords.size)
    second = jax.jit(jax.jacfwd(jax.grad(energy, argnums=1), argnums=1))
    hessian = np.asarray(second(jnp.array(0.), zero))
    print('Energy Hessian complete', perf_counter()-started, np.linalg.norm(hessian), flush=True)
    problem = atomic_problem(mf)
    linear = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-12, max_dense=64)
    respond = jax.jit(lambda u:problem.response(tangent=u.reshape(cell.coords.shape), config=linear))
    columns = []
    for direction in jnp.eye(len(zero)):
        result = respond(direction)
        assert result.converged
        columns.append(-np.asarray(result.values['forces']).reshape(-1))
    dfpt = np.stack(columns, axis=1)
    error = float(np.max(abs(hessian-dfpt)))
    assert error < 2e-8, error
    strain = jax.jit(lambda s:jax.jvp(lambda t:second(t, zero), (s,), (jnp.ones_like(s),))[1])
    derivative = np.asarray(strain(jnp.array(0.)))
    print('Strain derivative complete', perf_counter()-started, np.linalg.norm(derivative), flush=True)
    records = []
    for step in (1e-3, 3e-4, 1e-4):
        plus, minus = np.asarray(second(jnp.array(step), zero)), np.asarray(second(jnp.array(-step), zero))
        finite = (plus-minus)/(2*step)
        err = float(np.max(abs(derivative-finite)))
        records.append(dict(log_volume_step=step, max_hessian_derivative_error=err))
        print(records[-1], flush=True)
    assert records[-1]['max_hessian_derivative_error'] < 2e-7, records
    mass_ratio = physical_constants['atomic mass constant'][0] / physical_constants['electron mass'][0]
    masses = np.repeat(np.full(2, 1.00782503223) * mass_ratio, 3)
    dynamical = hessian / np.sqrt(masses[:, None]*masses[None, :])
    derivative_matrix = derivative / np.sqrt(masses[:, None]*masses[None, :])
    modes = solve_hermitian(jnp.asarray(dynamical), config=EigenSolverConfig(
        method='dense', nroots=6, atol=1e-12))
    assert np.all(modes.converged)
    values, vectors = np.asarray(modes.values), np.asarray(modes.vectors)
    # The isolated highest root is the local stretch. Soft or negative raw
    # curvatures of the other modes remain in the output, without ASR.
    gap = values[-1]-values[-2]
    assert values[-1] > 0 and gap > 1e-5*values[-1]
    slope = float(vectors[:, -1] @ derivative_matrix @ vectors[:, -1])
    frequency = np.sqrt(values[-1])
    frequency_slope = slope/(2*frequency)
    gruneisen = -slope/(2*values[-1])
    root_errors = []
    for step in (1e-3, 3e-4, 1e-4):
        displaced_roots = []
        for sign in (1., -1.):
            matrix = np.asarray(second(jnp.array(sign*step), zero))/np.sqrt(masses[:, None]*masses[None, :])
            local = solve_hermitian(jnp.asarray(matrix), config=EigenSolverConfig(
                method='dense', nroots=6, atol=1e-12))
            assert np.all(local.converged)
            assert abs(np.asarray(local.vectors)[:, -1] @ vectors[:, -1]) > .99
            displaced_roots.append(np.sqrt(float(local.values[-1])))
        finite = (displaced_roots[0]-displaced_roots[1])/(2*step)
        root_errors.append(dict(step=step, frequency_slope=finite, error=abs(finite-frequency_slope)))
    assert root_errors[-1]['error'] < 2e-9
    data = dict(complete=True, passed=True, geometry_bohr=np.asarray(cell.coords).tolist(),
        lattice_bohr=np.asarray(cell.lattice).tolist(), mesh=list(cell.mesh),
        basis=cell.basis_spec, pseudo=cell.pseudopotentials, energy_hartree=float(mf.e_tot),
        hessian_hartree_bohr2=hessian.tolist(), dfpt_hessian=dfpt.tolist(),
        dfpt_error=error, strain_hessian_derivative=derivative.tolist(), finite_differences=records,
        dynamical_eigenvalues=values.tolist(), selected_mode=5, selected_mode_gap=gap,
        selected_frequency_hartree=float(frequency), frequency_derivative_log_volume=frequency_slope,
        local_mode_gruneisen=gruneisen, frequency_finite_differences=root_errors,
        elapsed_seconds=perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Periodic H2 fixed finite Gamma/FFT9 model; log-volume derivative of raw Cartesian curvature. No temperature or physical material convergence claim.')
    (output/'gradscf.json').write_text(json.dumps(data, indent=2)+'\n')
    print('PASS',data['elapsed_seconds'],data['peak_rss_kib'], 'local gamma', gruneisen, flush=True)
    return data


if __name__ == "__main__":
    run()

# Verified on jy_gpu_1, CPU float64, JAX 0.10.0 (2026-10-16):
# Energy Hessian vs existing DFPT: max error 1.19e-15 Eh/bohr^2.
# dH/dln(V) FD errors: 1.06e-7, 9.51e-9, 1.06e-9 at steps 1e-3/3e-4/1e-4.
# Selected stretch omega = 0.0215937881945 Eh; d omega/dln(V) = -0.0171869514531 Eh.
# Local mode gamma = 0.795921090747; final frequency-slope FD error 9.49e-13 Eh.
# Workflow 61.97 s, peak RSS 2,369,960 KiB. The raw Hessian has 3 negative roots.
# Fixed coarse Gamma/FFT9 model and unoptimized geometry, not a stable crystal.
