"""Run native-AD NNAO joint training."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import numpy as np

from ..training import alternating as base
from ..training.native import (compilation_cache_path, read_initial_parameters,
    NativeJointEngine, JointAdam, MODEL_CONFIG)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--exponent-learning-rate', type=float, default=1e-3)
    parser.add_argument('--log-bound', type=float, default=.25)
    parser.add_argument('--implicit-tolerance', type=float, default=1e-8)
    parser.add_argument('--scf-rescue-level-shift', type=float, default=.2)
    parser.add_argument('--auxbasis', default='def2-universal-jkfit')
    parser.add_argument('--platform', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--cpu-jit-cache-size',type=int,default=8)
    parser.add_argument('--compilation-cache-dir',type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    values = [args.learning_rate, args.exponent_learning_rate, args.log_bound, args.implicit_tolerance]
    if args.epochs < 0 or args.cpu_jit_cache_size<1 or not np.isfinite(values).all() or min(values) <= 0:
        parser.error('Require nonnegative epochs and finite positive numerical settings.')
    if (args.output_dir/'last.npz').exists() and not args.resume:
        parser.error('Existing accepted checkpoint: use --resume or a fresh output directory.')
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu' if args.platform == 'cuda' else 'cpu'
    os.environ['JAX_ENABLE_X64'] = '1'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import jax
    jax.config.update('jax_enable_x64', True)
    rows = base.pilot.read_structures(args.structures)
    theta,beta,parent=read_initial_parameters(args.checkpoint,log_bound=args.log_bound)
    parent_identity=parent['identity']
    parent_energies=None
    # The legacy parity gate applies only to its exact original coefficients,
    # zero exponent scales and dataset. New datasets get a fresh stable baseline.
    if (base.pilot.file_hash(args.checkpoint)==base.PARENT_CHECKPOINT_SHA
            and base.pilot.file_hash(args.structures)==parent_identity.get('structures_sha256')):
        parent_energies={r['structure_id']:r['energy_hartree'] for r in parent['molecules']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from gradscf.integrals import _native
    library = Path(_native.__file__).parent/'lib'/('libgradscf_integrals.dylib'
        if platform.system() == 'Darwin' else 'libgradscf_integrals.so')
    from gradscf.tools.jax_runtime import configure_jax_persistent_cache
    native_hash=base.pilot.file_hash(library)
    cache_directory=configure_jax_persistent_cache(cache_dir=str(compilation_cache_path(args,native_hash)))
    identity = dict(parent_checkpoint_sha256=base.pilot.file_hash(args.checkpoint),
        structures_sha256=base.pilot.file_hash(args.structures), structure_ids=[r['structure_id'] for r in rows],
        scientific_source_sha256=base.pilot.scientific_source_hash(base.pilot.repository_root()),
        driver_sha256=base.pilot.file_hash(__file__), native_library_sha256=native_hash,
        compilation_cache_dir=cache_directory,
        imported_sources=base.pilot.verify_source_imports(base.pilot.repository_root()), model_config=MODEL_CONFIG,
        exponent_keys=[list(k) for k in base.EXPONENT_KEYS], ntheta=len(theta), dtype='float64',
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                       if k not in {'resume'}},
        coefficient_gradient='native VJP + implicit SCF + MACE VJP',
        exponent_gradient='native VJP + implicit SCF', optimizer_reset=True,
        parent_epoch=parent.get('epoch'),train_count=len(rows),heldout_count=0,
        stability_policy='real_internal_df_rhf',
        fixed_core_primitives=6, fixed_auxiliary_basis=True, primitive_cache=False)
    runtime = dict(hostname=platform.node(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        devices=[str(d) for d in jax.devices()], device_kinds=[d.device_kind for d in jax.devices()],
        cpu_devices=[str(d) for d in jax.devices('cpu')], jax_version=jax.__version__)
    base.pilot.atomic_json(args.output_dir/'runtime.json', runtime)
    base.pilot.atomic_json(args.output_dir/'metadata.json', identity)
    initial = np.concatenate((theta,beta))
    engine = NativeJointEngine(rows, args, parent_identity, None if args.resume else parent_energies)
    identity.update(trainable_mace_parameters=int(engine.trainable_mask.sum()),
                    frozen_parameter_indices=np.flatnonzero(~engine.trainable_mask).tolist())
    base.pilot.atomic_json(args.output_dir/'metadata.json', identity)
    if not args.resume:
        checks = [engine.check_gradient(initial, key) for key in ('W4-11/ch4', 'W4-11/n2')
                  if key in engine.contexts]
        base.pilot.atomic_json(args.output_dir/'gradient-checks.json', checks)
        print('Joint native/GPU gradient checks:', json.dumps(checks), flush=True)
    optimizer = JointAdam(len(theta), learning_rate=args.learning_rate,
        exponent_learning_rate=args.exponent_learning_rate, log_bound=args.log_bound,
        trainable_mask=engine.trainable_mask)
    summary = base.trainer.run_training(initial, engine.evaluate, optimizer, args.output_dir,
        identity, epochs=args.epochs, resume=args.resume)
    print(json.dumps({k: summary[k] for k in ('status', 'accepted_epoch', 'loss_hartree') if k in summary}), flush=True)
    return int(summary['status'] != 'complete')


if __name__ == "__main__":
    raise SystemExit(main())
