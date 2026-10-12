"""Profile full implicit NNAO backward and the production single-system epoch.

Run each thread configuration in a fresh process using the current native backend.
Warm timings exclude compilation. Training uses the existing accepted-state
Adam loop; profile-only passes do not update parameters or optimizer moments.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from gradscf.model.nnao.training import native as training


def split_scf(engine, context, args):
    """Keep primal iteration tapes out of a separately timed implicit adjoint.

    Use the production primal/stability routines, and the identical density
    fixed point and energy expression. Agreement with the production fused
    value_and_grad is mandatory before accepting any measurements.
    """
    import jax
    import jax.numpy as jnp
    from gradscf.model.nnao.training import objective as experiment
    from gradscf.scf.stability import stabilize_df_rhf
    from gradscf.solvers.nonlinear import ImplicitFixedPointConfig, implicit_fixed_point_solution
    cfg = experiment.RKSConfig(xc_spec='hf', jk_backend='df', max_cycle=150,
        conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9)
    nelectron = context['layout'].nelectron

    def solve(inputs):
        s, h, factors = inputs
        n = s.shape[0]
        kwargs = dict(overlap=s, hcore=h, eri=None, df_factors=factors,
            nelectron=nelectron, nuclear_repulsion=context['enuc'],
            ao=jnp.zeros((0, n)), ao_deriv1=jnp.zeros((4, 0, n)), grid_weights=jnp.zeros(0))
        result, _, _ = experiment._run_scf_with_rescue(
            experiment.run_rks_from_integrals_traceable, kwargs, cfg,
            rescue_level_shift=args.scf_rescue_level_shift)
        result, stability = stabilize_df_rhf(result, **kwargs, config=cfg)
        return result.density_matrix, result.converged & stability['stable']

    def energy(inputs, density, converged):
        s, h, factors = inputs
        occ = jnp.zeros(s.shape[0], dtype=s.dtype).at[:nelectron//2].set(2.)
        def fixed_point(d, physical):
            overlap, core, df = physical
            j, k = experiment.build_jk_from_df(df, d)
            _, c = experiment._diagonalize_fock(core+j-.5*k,
                experiment._orthogonalizer(overlap, cfg.orthogonalization_eps))
            return experiment._build_density_from_occ(c, occ)
        d = implicit_fixed_point_solution(inputs, solution=density,
            fixed_point=fixed_point,
            config=ImplicitFixedPointConfig(tolerance=args.implicit_tolerance,
                                           max_iter=100, restart=40),
            converged=converged, require_converged=True)
        j, k = experiment.build_jk_from_df(factors, d)
        return jnp.sum(d*h)+.5*jnp.sum(d*j)-.25*jnp.sum(d*k)+context['enuc']
    return jax.jit(solve), jax.jit(energy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--structure-id', default='S22/19a')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--platform', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--implicit-tolerance', type=float, default=1e-8)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--exponent-learning-rate', type=float, default=1e-3)
    parser.add_argument('--log-bound', type=float, default=.25)
    parser.add_argument('--scf-rescue-level-shift', type=float, default=.2)
    parser.add_argument('--auxbasis', default='def2-universal-jkfit')
    parser.add_argument('--cpu-jit-cache-size', type=int, default=8)
    args = parser.parse_args()
    if min(args.epochs, args.repeats) < 1:
        parser.error('epochs and repeats must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu' if args.platform == 'cuda' else 'cpu'
    os.environ['JAX_ENABLE_X64'] = '1'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import jax
    import jax.numpy as jnp
    from gradscf.integrals import _native
    from gradscf.tools.jax_runtime import configure_jax_persistent_cache
    jax.config.update('jax_enable_x64', True)
    pilot = training.base.pilot
    rows = pilot.read_structures(args.structures, ids=[args.structure_id])
    theta, beta, parent = training.read_initial_parameters(args.checkpoint, log_bound=args.log_bound)
    initial = np.concatenate((theta, beta))
    library = Path(_native.__file__).parent/'lib'/('libgradscf_integrals.dylib'
        if platform.system() == 'Darwin' else 'libgradscf_integrals.so')
    native_hash = pilot.file_hash(library)
    configure_jax_persistent_cache(cache_dir=str(args.output_dir/'compilation-cache'/native_hash[:16]))
    started = time.perf_counter()
    engine = training.NativeJointEngine(rows, args, parent['identity'], None)
    context = engine.contexts[args.structure_id]
    metadata = dict(structure=rows[0], checkpoint_sha256=pilot.file_hash(args.checkpoint),
        native_sha256=native_hash, driver_sha256=pilot.file_hash(__file__),
        scientific_source_sha256=pilot.scientific_source_hash(training.ROOT),
        imported_sources=pilot.verify_source_imports(training.ROOT),
        host=platform.node(), slurm_job=os.environ.get('SLURM_JOB_ID'),
        jax_version=jax.__version__, device=str(engine.device), device_kind=engine.device.device_kind,
        native_threads=os.environ.get('GRADSCF_NATIVE_THREADS', '1'), dtype='float64',
        nao=context['layout'].topology.nao,
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        setup_seconds=time.perf_counter()-started)
    pilot.atomic_json(args.output_dir/'metadata.json', metadata)
    solve_density, implicit_energy = split_scf(engine, context, args)

    def profile():
        tick = time.perf_counter()
        t = jax.device_put(theta, engine.device)
        outputs = engine.predict(t, context['graph'])
        with jax.default_device(engine.cpu):
            outputs_cpu, beta_cpu = jax.device_put((outputs, beta), engine.cpu)
            physical = context['physical'](outputs_cpu, beta_cpu)
            jax.block_until_ready(physical)
        inputs = jax.device_put(physical, engine.device)
        density, converged = solve_density(inputs)
        value, pullback = jax.vjp(lambda x: implicit_energy(x, density, converged), inputs)
        jax.block_until_ready((value, converged))
        if not bool(converged):
            raise RuntimeError('Profile primal is not a converged stable SCF root')
        forward_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        cotangent = pullback(jnp.ones_like(value))[0]
        jax.block_until_ready(cotangent)
        implicit_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        with jax.default_device(engine.cpu):
            output_cot, beta_cot = context['pullback'](
                outputs_cpu, beta_cpu, jax.device_put(cotangent, engine.cpu))
            jax.block_until_ready((output_cot, beta_cot))
        native_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        theta_cot = engine.backward(t, context['graph'], jax.device_put(output_cot, engine.device))
        gradient = np.concatenate((np.asarray(theta_cot), np.asarray(beta_cot)))
        mace_seconds = time.perf_counter()-tick
        if not np.isfinite(gradient).all():
            raise RuntimeError('Nonfinite full backward gradient')
        return dict(energy_hartree=float(value), forward_seconds=forward_seconds,
            implicit_backward_seconds=implicit_seconds, native_backward_seconds=native_seconds,
            mace_backward_seconds=mace_seconds,
            full_backward_seconds=implicit_seconds+native_seconds+mace_seconds,
            converged_stable=True), gradient

    cold, reference_gradient = profile()
    profiles = []
    for _ in range(args.repeats):
        record, gradient = profile()
        np.testing.assert_allclose(gradient, reference_gradient, atol=2e-9, rtol=2e-8)
        profiles.append(record)
        print('BACKWARD', json.dumps(record), flush=True)
    # Verify the separately timed adjoint agrees with the actual combined
    # value_and_grad entry point used during production training.
    tick = time.perf_counter()
    value, production_gradient, details = engine.molecule(initial, args.structure_id)
    production_warmup = time.perf_counter()-tick
    np.testing.assert_allclose(value, cold['energy_hartree'], atol=1e-9, rtol=0)
    np.testing.assert_allclose(production_gradient, reference_gradient, atol=2e-8, rtol=2e-7)
    pilot.atomic_json(args.output_dir/'production-diagnostics.json', details)
    optimizer = training.JointAdam(len(theta), learning_rate=args.learning_rate,
        exponent_learning_rate=args.exponent_learning_rate, log_bound=args.log_bound,
        trainable_mask=engine.trainable_mask)
    # Compile the update without accepting it or advancing the training state.
    jax.block_until_ready(optimizer.update(initial, production_gradient, optimizer.init(initial)))
    epochs = []
    original_write = pilot.atomic_json
    def record_progress(path, payload):
        original_write(path, payload)
        if Path(path).name == 'progress.json' and 'epoch_elapsed_seconds' in payload:
            epochs.append(dict(payload))
    pilot.atomic_json = record_progress
    try:
        summary = training.base.trainer.run_training(initial, engine.evaluate, optimizer,
            args.output_dir/'training', metadata, epochs=args.epochs)
    finally:
        pilot.atomic_json = original_write
    if summary['status'] != 'complete':
        raise RuntimeError(f'Training benchmark failed: {summary}')
    np.savez(args.output_dir/'initial-gradient.npz', parameters=initial,
             profile_gradient=reference_gradient, production_gradient=production_gradient)
    result = dict(metadata=metadata, cold=cold, profiles=profiles, epochs=epochs,
        production_warmup_seconds=production_warmup,
        full_backward_median_seconds=float(np.median([r['full_backward_seconds'] for r in profiles])),
        epoch_median_seconds=float(np.median([r['epoch_elapsed_seconds'] for r in epochs])),
        training=summary)
    original_write(args.output_dir/'result.json', result)
    print('BENCHMARK_COMPLETE', result['full_backward_median_seconds'], result['epoch_median_seconds'], flush=True)


if __name__ == '__main__':
    main()
