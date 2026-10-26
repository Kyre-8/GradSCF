"""Periodic H2: harmonic-curvature response to a local exchange parameter.

The external functional is alpha times Dirac exchange. Its current density,
potential and response kernel use the normal Functional/SCF/DFPT interface.
This isolates XC-parameter response at fixed cell and geometry; the small
Gamma/FFT41 model is not a stable bulk material or an XC accuracy benchmark.
"""
import hashlib
import json
from pathlib import Path
import platform
import resource
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import physical_constants

jax.config.update('jax_enable_x64', True)
from gradscf import dft, dfpt
from gradscf.pbc import gto, dft as periodic_dft
from gradscf.pbc.scf import run_kpoint_scf
from gradscf.integrals.periodic.fft import build_inputs
from gradscf.scf import SCFDifferentiationConfig
from gradscf.solvers import EigenSolverConfig, LinearSolverConfig, solve_hermitian


OUTPUT = Path('reproducibility/ad_showcase/P07_periodic_xc')
COORDINATES = np.array([[.4, .5, .6], [1.8, .8, .9]])
LATTICE = np.eye(3)*6.
CONTROLS = dict(conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-11,
                max_cycle=150, damping=.1)
DIFFERENTIATION = SCFDifferentiationConfig(mode='implicit', tolerance=1e-11,
                                          max_iter=100, restart=20)


def exchange_energy(parameters, inputs):
    density, weights = inputs
    # The reported density minimum verifies that this numerical floor is inactive.
    rho = jnp.maximum(density, 1e-20)
    return -parameters['alpha']*.75*(3/jnp.pi)**(1/3)*jnp.dot(weights, rho**(4/3))


FUNCTIONAL = dft.Functional(lambda state:(state.rho, state.weights), exchange_energy)


def run(mesh=41, output=OUTPUT):
    started = perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cell = gto.M(atom=list(zip(['H', 'H'], COORDINATES)), a=LATTICE, unit='Bohr',
                 basis='gth-szv', pseudo='gth-pade', mesh=(mesh,)*3, precision=1e-10)
    alpha = jnp.array(1.)
    mf = periodic_dft.RKS(cell, functional=FUNCTIONAL, params={'alpha':alpha},
        exxdiv=None, differentiation=DIFFERENTIATION, **CONTROLS).run()
    assert mf.converged

    def state(scale, displacement):
        data = build_inputs(cell, parameters=cell.parameters_at(cell.coords+displacement.reshape(2, 3)))
        return run_kpoint_scf(data, mesh=cell.mesh, nelec=cell.nelec, xc=None,
            functional=FUNCTIONAL, params={'alpha':scale}, exxdiv=None,
            differentiation=DIFFERENTIATION, **CONTROLS)

    def energy(scale, displacement):
        return state(scale, displacement).total_energy

    zero = jnp.zeros(6)
    hessian = jax.jit(jax.jacfwd(jax.grad(energy, argnums=1), argnums=1))
    before = perf_counter()
    base = np.asarray(hessian(alpha, zero))
    hessian_seconds = perf_counter()-before
    problem = dfpt.atomic_problem(mf)
    linear = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-12, max_dense=64)
    respond = jax.jit(lambda u:problem.response(tangent=u.reshape(2, 3), config=linear))
    columns, residuals = [], []
    for direction in jnp.eye(6):
        result = respond(direction)
        assert result.converged
        columns.append(-np.asarray(result.values['forces']).ravel())
        residuals.append(float(result.residual_norm))
    dfpt_error = float(np.max(abs(base-np.stack(columns, axis=1))))
    assert dfpt_error < 2e-8
    before = perf_counter()
    sensitivity = jax.jit(lambda a:jax.jvp(lambda x:hessian(x, zero),
                                           (a,), (jnp.ones_like(a),))[1])
    derivative = np.asarray(sensitivity(alpha))
    derivative_seconds = perf_counter()-before
    before = perf_counter()
    np.asarray(sensitivity(alpha))
    warm_seconds = perf_counter()-before
    finite = []
    for step in (1e-2, 3e-3, 1e-3):
        plus, minus = np.asarray(hessian(alpha+step, zero)), np.asarray(hessian(alpha-step, zero))
        error = float(np.max(abs(derivative-(plus-minus)/(2*step))))
        finite.append(dict(parameter_step=step, max_error_hartree_bohr2=error))
        print('Parameter FD:', finite[-1], flush=True)
    assert finite[-1]['max_error_hartree_bohr2'] < 2e-7
    assert np.linalg.norm(derivative) > 1e-5

    mass_ratio = physical_constants['atomic mass constant'][0]/physical_constants['electron mass'][0]
    masses = np.repeat(np.full(2, 1.00782503223)*mass_ratio, 3)
    mass_scale = np.sqrt(masses[:, None]*masses[None, :])
    modes = solve_hermitian(jnp.asarray(base/mass_scale),
        config=EigenSolverConfig(method='dense', nroots=6, atol=1e-12))
    assert np.all(modes.converged)
    values, vectors = np.asarray(modes.values), np.asarray(modes.vectors)
    assert values[-1] > 0 and values[-1]-values[-2] > 1e-5*values[-1]
    frequency = np.sqrt(values[-1])
    slope = float(vectors[:, -1]@(derivative/mass_scale)@vectors[:, -1])/(2*frequency)
    samples = []
    for scale in (.8, .9, 1., 1.1, 1.2):
        current = state(jnp.array(scale), zero)
        assert current.converged
        matrix = np.asarray(hessian(jnp.array(scale), zero))
        roots, frame = np.linalg.eigh(matrix/mass_scale)
        overlap = abs(frame[:, -1]@vectors[:, -1])
        assert roots[-1] > 0 and overlap > .99
        samples.append(dict(alpha=scale, energy_hartree=float(current.total_energy),
            frequency_hartree=float(np.sqrt(roots[-1])), mode_overlap=float(overlap),
            raw_dynamical_eigenvalues=roots.tolist()))
    rho, _ = FUNCTIONAL.inputs_for_density(mf.inputs, mf.result.density_spin)
    assert float(jnp.min(rho)) > 1e-18
    record = dict(complete=True, passed=True, coordinates_bohr=COORDINATES.tolist(),
        lattice_bohr=LATTICE.tolist(), basis='gth-szv', pseudo='gth-pade', mesh=list(cell.mesh),
        precision=cell.precision, alpha=1., xc='alpha * spin-unpolarized Dirac exchange; no correlation',
        total_energy_hartree=float(mf.e_tot), controls=CONTROLS,
        density_minimum=float(jnp.min(rho)), dfpt_error=dfpt_error, dfpt_residuals=residuals,
        hessian_hartree_bohr2=base.tolist(), hessian_alpha_derivative=derivative.tolist(),
        finite_differences=finite, raw_dynamical_eigenvalues=values.tolist(),
        selected_mode=vectors[:, -1].tolist(), mass_scale=mass_scale.tolist(),
        selected_frequency_hartree=float(frequency), frequency_alpha_derivative=slope,
        sensitivity_map=samples, elapsed_seconds=perf_counter()-started,
        timings_seconds=dict(hessian_first=hessian_seconds, derivative_first=derivative_seconds,
                            derivative_warm=warm_seconds),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hostname=platform.node(), jax=jax.__version__, backend=jax.default_backend(), dtype='float64',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Relaxed XC-parameter derivative of local periodic curvature; fixed coarse model, no stable bulk or XC accuracy claim.')
    (output/'gradscf.json').write_text(json.dumps(record, indent=2)+'\n')
    print('XC parameter gates passed:', dfpt_error, finite[-1], 'seconds', record['elapsed_seconds'], flush=True)
    return record


if __name__ == '__main__':
    run()

# c20 CPU / JAX 0.10.0 / float64, FFT41^3, 2026-10-16:
# E = -1.1043121496946025 Eh; complete-energy Hessian vs DFPT: 3.66e-15 Eh/bohr^2.
# dH/dalpha vs rebuilt Hessian differences (h=1e-2,3e-3,1e-3):
# 1.43e-14, 5.58e-14, 1.73e-13 Eh/bohr^2.
# First Hessian / first parameter derivative / warm derivative: 17.17 / 20.88 / 4.28 s.
# Full example: 113.46 s; process peak RSS: 6,890,432 KiB.
# Unoptimized two-AO H2 cell: negative raw modes remain; this is a response test.
