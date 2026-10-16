"""Al WGC: fixed-resolution supercells and nontrivial density AD scaling.

Edit REPEAT below; run each size in a fresh process on jy_gpu_1. No CLI.
The potential directions include both fixed physical and supercell wavelengths.
This measures the declared implementation, including its quadratic input setup.
It does not establish KEDF physical accuracy or nuclear-coordinate derivatives.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)
from gradscf import dfpt, ofdft, scf

try:
    from .atlas_inputs import crystal, gaussian_free_inputs, pz_lda_energy, provenance
except ImportError:
    from atlas_inputs import crystal, gaussian_free_inputs, pz_lda_energy, provenance


REPEAT = 1                         # staged sequence: 1, 3, then 5 and 7 if feasible
OUTPUT = Path('reproducibility/ad_showcase/P08_ofdft_parameter_response/large_system_scaling')
EULER_TOLERANCE = 1e-9              # density-weighted Euler RMS, Hartree
SEED = 1729
DIRECTION = np.array([.6, .8])     # unit Euclidean norm, potential units Hartree
KINETIC = ofdft.KineticFunctional('wgc')


def crystal_probes(repeat, base_mesh=(17, 17, 17)):
    """Al primitive repeat with exactly fixed voxel and unit-RMS directions."""
    if repeat < 1 or repeat % 2 != 1:
        raise ValueError('Use a positive odd repeat to preserve the odd FFT mesh.')
    atoms, _ = crystal('Al', spacing=.18)
    atoms = atoms.repeat((repeat,)*3)
    mesh = tuple(repeat*n for n in base_mesh)
    fractions = np.stack(np.meshgrid(*[np.arange(n)/n for n in mesh], indexing='ij'), axis=-1).reshape(-1, 3)
    phases = np.random.default_rng(SEED).uniform(-np.pi, np.pi, 4)
    p = (np.cos(2*np.pi*repeat*fractions[:, 0]+phases[0])
         +.35*np.cos(2*np.pi*(fractions[:, 0]+fractions[:, 1])+phases[1]))
    q = (np.sin(2*np.pi*repeat*fractions[:, 2]+phases[2])
         +.35*np.sin(2*np.pi*(fractions[:, 1]+fractions[:, 2])+phases[3]))
    probes = np.stack((p, q), axis=1)
    probes -= probes.mean(axis=0)
    probes /= np.sqrt(np.mean(probes**2, axis=0))
    return atoms, mesh, probes


def make_inputs(repeat, base_mesh=(17, 17, 17)):
    atoms, mesh, probes = crystal_probes(repeat, base_mesh)
    return gaussian_free_inputs(atoms, mesh), probes


def configuration(data):
    # The dimensionless sphere constraint is checked separately in stationarity.
    return ofdft.OFDFTConfig(xc=None, maxiter=1000,
        tolerance=2*float(data.nelectron)*EULER_TOLERANCE,
        differentiation=scf.SCFDifferentiationConfig(tolerance=1e-10,
            max_iter=100, restart=40, regularization=0.))


def solver(data, probes):
    config = configuration(data)
    probes = jnp.asarray(probes)
    def solve(theta):
        moved = replace(data, external_potential=data.external_potential+probes@theta)
        return ofdft.run_ofdft(moved, kinetic=KINETIC, xc_energy_fn=pz_lda_energy,
                              config=config)
    return solve


def stationarity(data, result):
    problem = dfpt.ofdft_problem(data, result, kinetic=KINETIC,
        xc_energy_fn=pz_lda_energy, config=configuration(data))
    residual = problem.residual(problem.solution, problem.inputs)
    raw = residual.at[:-1].multiply(data.nelectron)
    return dict(euler_rms_hartree=float(jnp.linalg.norm(residual[:-1])/2),
                sphere_constraint=float(result.electron_number/data.nelectron-1),
                response_chart_constraint=float(residual[-1]),
                full_kkt_residual=float(jnp.linalg.norm(raw)),
                normalized_response_residual=float(jnp.linalg.norm(residual)),
                forward_iterations=int(result.iterations))


def timed(function, *args, repetitions=3):
    """Compile separately, then synchronize the complete output at every call."""
    started = perf_counter()
    executable = jax.jit(function).lower(*args).compile()
    compile_seconds = perf_counter()-started
    started = perf_counter()
    value = jax.block_until_ready(executable(*args))
    first_seconds = perf_counter()-started
    samples = []
    valid_forward = not hasattr(value, 'converged') or bool(value.converged)
    for _ in range(repetitions if valid_forward else 0):
        started = perf_counter()
        value = jax.block_until_ready(executable(*args))
        samples.append(perf_counter()-started)
    record = dict(lower_compile_seconds=compile_seconds, first_execution_seconds=first_seconds,
                  warm_seconds=samples, warm_median_seconds=float(np.median(samples)) if samples else None)
    print('TIMING', json.dumps(record), flush=True)
    return value, record


def source_manifest():
    files = sorted(Path('src/gradscf').rglob('*.py'))+sorted(Path('examples/ofdft').glob('*.py'))
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return dict(source_sha256=digest, files=hashes)


def calculate(repeat=REPEAT):
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT/f'al_{repeat}.json'
    manifest = source_manifest()
    (OUTPUT/f'al_{repeat}_source.json').write_text(json.dumps(manifest, indent=2)+'\n')
    report = dict(complete=False, repeat=repeat, atom_count=repeat**3, mesh=[17*repeat]*3,
        model='WGC99 second-order + Hartree + OEPP + PZ-LDA', forces_validated=False,
        source_sha256=manifest['source_sha256'], hostname=platform.node(),
        python=platform.python_version(), jax=jax.__version__, backend=jax.default_backend(),
        device_kinds=[device.device_kind for device in jax.devices()], dtype='float64',
        environment={key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS',
            'JAX_ENABLE_X64', 'XLA_PYTHON_CLIENT_PREALLOCATE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        diagnostics=None, density_min=None, density_max=None,
        native_euler_tolerance_hartree=EULER_TOLERANCE,
        adjoint_linear_residual=None,
        adjoint_diagnostic='Unavailable from public result; implicit solver rejects unsuccessful solves with NaNs')
    target.write_text(json.dumps(report, indent=2)+'\n')
    setup_start = perf_counter()
    data, probes = make_inputs(repeat)
    jax.block_until_ready(data)
    setup_seconds = perf_counter()-setup_start
    solve = solver(data, probes)
    theta = jnp.zeros(2)
    direction = jnp.asarray(DIRECTION)
    ground, forward_time = timed(solve, theta)
    diagnostics = stationarity(data, ground)
    report.update(stage='forward passed; response pending' if ground.converged else 'forward failed', diagnostics=diagnostics,
        energy_hartree=float(ground.total_energy), input_preparation_seconds=setup_seconds,
        density_min=float(ground.density.min()), density_max=float(ground.density.max()),
        forward_timing=forward_time)
    target.write_text(json.dumps(report, indent=2)+'\n')
    assert ground.converged, diagnostics
    assert diagnostics['euler_rms_hartree'] <= 1.01*EULER_TOLERANCE
    assert abs(diagnostics['sphere_constraint']) < 2e-12
    print('GROUND', repeat, float(ground.total_energy), diagnostics, flush=True)

    density = lambda value: solve(value).density
    energy = lambda value: solve(value).total_energy
    response, jvp_time = timed(lambda t, v: jax.jvp(density, (t,), (v,))[1], theta, direction)
    cotangent = data.weights*jnp.asarray(probes[:, 0]+.3*probes[:, 1])/data.nelectron
    # Preparation includes the primal solve and residual capture. Pullback timing
    # alone is never reported as the cost of a complete reverse calculation.
    prep_start = perf_counter()
    primal, pullback = jax.vjp(jax.jit(density), theta)
    jax.block_until_ready((primal, pullback))
    prep_seconds = perf_counter()-prep_start
    reverse, pullback_time = timed(lambda c: pullback(c)[0], cotangent)
    full_reverse, vjp_time = timed(jax.grad(lambda t: jnp.dot(cotangent, density(t))), theta)
    hvp, hvp_time = timed(lambda t, v: jax.jvp(jax.grad(energy), (t,), (v,))[1], theta, direction)
    assert all(np.isfinite(value).all() for value in (response, reverse, full_reverse, hvp))
    np.testing.assert_allclose(reverse, full_reverse, atol=2e-9, rtol=2e-7)
    adjoint_error = float(abs(reverse@direction-cotangent@response))
    assert adjoint_error < 2e-9
    response_rms = float(jnp.sqrt(jnp.dot(data.weights, response**2)/jnp.sum(data.weights)))
    assert response_rms > 1e-5, 'The direction must produce a nontrivial density response'
    number_response = float(jnp.dot(data.weights, response))
    assert abs(number_response)/float(data.nelectron) < 1e-9

    finite = []
    for step in (1e-3, 3e-4, 1e-4):
        plus, minus = jax.block_until_ready((jax.jit(solve)(theta+step*direction),
                                           jax.jit(solve)(theta-step*direction)))
        assert plus.converged and minus.converged
        numerical = (plus.density-minus.density)/(2*step)
        curvature = (plus.total_energy-2*ground.total_energy+minus.total_energy)/step**2
        relative = float(jnp.linalg.norm(numerical-response)/jnp.linalg.norm(response))
        finite.append(dict(step_hartree=step, density_jvp_relative_l2_error=relative,
            energy_curvature_per_atom=float(curvature)/repeat**3,
            energy_curvature_error_per_atom=float(abs(curvature-direction@hvp))/repeat**3))
    assert min(row['density_jvp_relative_l2_error'] for row in finite) < 2e-4
    assert min(row['energy_curvature_error_per_atom'] for row in finite) < 2e-5
    np.savez_compressed(OUTPUT/f'al_{repeat}.npz', density=np.asarray(ground.density),
        density_jvp=np.asarray(response), probes=probes, density_cotangent=np.asarray(cotangent),
        potential=np.asarray(data.external_potential), lattice=np.asarray(data.lattice),
        weights=np.asarray(data.weights), vjp=np.asarray(reverse), hvp=np.asarray(hvp))
    report.update(complete=True, stage='complete', source_sha256=manifest['source_sha256'],
        input_preparation_seconds=setup_seconds, total_elapsed_seconds=perf_counter()-started,
        energy_hartree=float(ground.total_energy), energy_hartree_per_atom=float(ground.total_energy)/repeat**3,
        energy_components={name: float(value) for name, value in zip(ground.components._fields, ground.components)},
        electron_number=float(ground.electron_number), diagnostics=diagnostics,
        density_response_rms=response_rms, relative_electron_number_response=number_response/float(data.nelectron),
        adjoint_identity_error=adjoint_error, vjp=np.asarray(reverse).tolist(), hvp=np.asarray(hvp).tolist(),
        directional_energy_curvature=float(direction@hvp), finite_differences=finite,
        timings=dict(forward=forward_time, jvp_including_forward=jvp_time,
            vjp_preparation_seconds=prep_seconds, vjp_pullback_only=pullback_time,
            vjp_including_forward=vjp_time, hvp_including_forward=hvp_time),
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__,
        backend=jax.default_backend(), devices=[str(device) for device in jax.devices()], dtype='float64',
        device_kinds=[device.device_kind for device in jax.devices()],
        environment={key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS',
            'JAX_ENABLE_X64', 'XLA_PYTHON_CLIENT_PREALLOCATE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        probe_seed=SEED, probe_normalization='each potential column zero mean, unit RMS; path norm 1',
        direction=DIRECTION.tolist(), pseudopotential=provenance('Al'),
        scope='Fixed-ion periodic Al; no force/stress validation and no claim of near-linear total cost')
    target.write_text(json.dumps(report, indent=2)+'\n')
    print('COMPLETE', repeat, 'seconds', report['total_elapsed_seconds'], flush=True)
    return report


if __name__ == '__main__':
    calculate()
