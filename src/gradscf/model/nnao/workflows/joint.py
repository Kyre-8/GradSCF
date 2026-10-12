"""Run alternating coefficient and exponent calibration."""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import numpy as np

from .. import runtime as pilot
from ..training import trainer
from ..training.alternating import read_parent, JointEngine, run_joint
from ..config import MODEL_CONFIG, EXPONENT_KEYS

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--coefficient-steps', type=int, default=2)
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--maxfev', type=int, default=12)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--log-bound', type=float, default=.25)
    parser.add_argument('--implicit-tolerance', type=float, default=1e-8)
    parser.add_argument('--scf-rescue-level-shift', type=float, default=.2)
    parser.add_argument('--auxbasis', default='def2-universal-jkfit')
    parser.add_argument('--platform', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args(argv)
    if (args.coefficient_steps < 0 or args.cycles < 1 or args.maxfev < 1
            or not np.isfinite([args.learning_rate, args.log_bound, args.implicit_tolerance]).all()
            or min(args.learning_rate, args.log_bound, args.implicit_tolerance) <= 0):
        parser.error('Steps/budgets and numerical parameters must be finite and positive.')
    if (args.output_dir / 'last.npz').exists(): parser.error('Use a fresh output directory; the accepted checkpoint is preserved.')
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu' if args.platform == 'cuda' else 'cpu'
    os.environ['JAX_ENABLE_X64'] = '1'; os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    threads = os.environ.get('SLURM_CPUS_PER_TASK', '16')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[name] = threads
    rows = pilot.read_structures(args.structures); theta, parent_identity = read_parent(args.checkpoint, rows)
    parent_hash = pilot.file_hash(args.checkpoint); args.output_dir.mkdir(parents=True, exist_ok=True)
    import jax
    jax.config.update('jax_enable_x64', True)
    identity = dict(parent_checkpoint_sha256=parent_hash, parent_identity=parent_identity,
        structures_sha256=pilot.file_hash(args.structures), structure_ids=[r['structure_id'] for r in rows],
        scientific_source_sha256=pilot.scientific_source_hash(pilot.repository_root()), driver_sha256=pilot.file_hash(__file__),
        imported_sources=pilot.verify_source_imports(pilot.repository_root()), model_config=MODEL_CONFIG,
        exponent_keys=[list(key) for key in EXPONENT_KEYS], coefficient_gradient='strict implicit SCF',
        exponent_optimizer='bounded element-wise derivative-free Powell',
        optimizer_reset=True, configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        devices=[str(d) for d in jax.devices()], device_kinds=[d.device_kind for d in jax.devices()],
        integral_devices=[str(d) for d in jax.devices('cpu')], scf_platform=args.platform,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), cpu_threads=int(threads), hostname=platform.node(),
        python_version=platform.python_version(), versions={name: version(name) for name in
            ('jax', 'jaxlib', 'flax', 'mace-jax', 'e3nn-jax', 'optax', 'scipy')},
        scf_config=dict(max_cycle=150, conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9,
            init_guess='hcore', rescue_level_shift=args.scf_rescue_level_shift), dtype='float64')
    from gradscf.integrals import _native
    library = Path(_native.__file__).parent / 'lib' / ('libgradscf_integrals.dylib' if platform.system() == 'Darwin' else 'libgradscf_integrals.so')
    identity['native_library_sha256'] = pilot.file_hash(library)
    pilot.atomic_json(args.output_dir / 'metadata.json', identity)
    optimizer = trainer.Adam(args.learning_rate)
    engine = JointEngine(rows, args, parent_identity)
    def save(state):
        metadata = dict(identity=identity, loss_hartree=state['loss_hartree'], energies=state['energies'],
            molecules=state['molecules'], theta_steps=state['theta_steps'], alpha_steps=state['alpha_steps'],
            cache_manifest=state['cache']['manifests'], history=state['history'])
        temporary = args.output_dir / 'last.tmp.npz'
        with temporary.open('wb') as stream:
            np.savez(stream, theta=np.asarray(state['theta']), beta=state['beta'], gradient=state['gradient'],
                optimizer=np.frombuffer(optimizer.encode(state['optimizer']), dtype=np.uint8),
                metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
        os.replace(temporary, args.output_dir / 'last.npz')
        progress = dict(status='running', theta_steps=state['theta_steps'], alpha_steps=state['alpha_steps'],
            loss_hartree=state['loss_hartree'], beta=state['beta'].tolist(), phase=state['history'][-1]['phase'])
        pilot.atomic_json(args.output_dir / 'progress.json', progress)
        print(f'Joint {progress["phase"]}: theta={progress["theta_steps"]} alpha={progress["alpha_steps"]} '
              f'mean E={progress["loss_hartree"]:.12f} Ha', flush=True)
    def on_failure(error, theta, beta, summary):
        path = args.output_dir / 'failed-candidate.npz'
        trainer.save_failure_snapshot(path, theta, error, dict(summary, beta=np.asarray(beta).tolist()))
        summary['failure_snapshot'] = path.name
    state, summary = run_joint(theta, np.zeros(14), rows, optimizer, engine.prepare, engine.evaluate,
        engine.forward, save=save, coefficient_steps=args.coefficient_steps, cycles=args.cycles,
        maxfev=args.maxfev, log_bound=args.log_bound, on_failure=on_failure)
    summary.update(identity=identity, timings=engine.timings,
        original_checkpoint_bytes_unchanged=pilot.file_hash(args.checkpoint) == parent_hash,
        final_beta=state['beta'].tolist() if state else None, final_molecules=state['molecules'] if state else None)
    pilot.atomic_json(args.output_dir / 'summary.json', summary)
    pilot.atomic_json(args.output_dir / 'candidate_history.json', summary['history'])
    pilot.atomic_json(args.output_dir / 'progress.json', {key: summary[key] for key in
        ('status', 'accepted_theta_steps', 'accepted_alpha_steps', 'loss_hartree', 'final_beta')})
    print(json.dumps({key: summary[key] for key in ('status', 'loss_hartree', 'accepted_theta_steps', 'accepted_alpha_steps', 'final_beta')}, indent=2), flush=True)
    return int(summary['status'] != 'complete' or not summary['original_checkpoint_bytes_unchanged'])


if __name__ == "__main__":
    raise SystemExit(main())
