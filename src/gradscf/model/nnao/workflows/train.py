"""Train shared fixed-exponent NNAO contractions."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import socket
import numpy as np

from .. import runtime as _pilot
from ..training.trainer import (verify_validation, verify_validation_sources,
    make_evaluators, evaluate_batch, evaluate_values, Adam, run_training)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--validation-summary', type=Path, required=True)
    parser.add_argument('--heldout-structures', type=Path)
    parser.add_argument('--heldout-validation-summary', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--final-learning-rate', type=float)
    parser.add_argument('--validation-interval', type=int, default=20)
    parser.add_argument('--cache-storage', choices=('device', 'host', 'mmap'), default='device')
    parser.add_argument('--implicit-tolerance', type=float, default=1e-8)
    parser.add_argument('--scf-rescue-level-shift', type=float, default=0.0)
    parser.add_argument('--auxbasis', default='def2-universal-jkfit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not np.isfinite(args.scf_rescue_level_shift) or args.scf_rescue_level_shift < 0:
        parser.error('SCF rescue level shift must be finite and nonnegative.')
    if (args.heldout_structures is None) != (args.heldout_validation_summary is None):
        parser.error('--heldout-structures and --heldout-validation-summary must be provided together.')
    if (args.epochs < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not np.isfinite(args.implicit_tolerance) or args.implicit_tolerance <= 0):
        parser.error('Epochs, learning rate and implicit tolerance must be positive and finite.')
    if args.validation_interval < 1 or (args.final_learning_rate is not None and (
            not np.isfinite(args.final_learning_rate) or not 0 < args.final_learning_rate <= args.learning_rate)):
        parser.error('Validation interval must be positive; final learning rate must be finite and in (0, initial].')
    rows = _pilot.read_structures(args.structures)
    heldout_rows = _pilot.read_structures(args.heldout_structures) if args.heldout_structures else []
    if set(row['structure_id'] for row in rows) & set(row['structure_id'] for row in heldout_rows):
        raise ValueError('Train and held-out structure IDs overlap.')
    validation = verify_validation(args.validation_summary, args.structures,
        [row['structure_id'] for row in rows], tolerance=args.implicit_tolerance,
        rescue_level_shift=args.scf_rescue_level_shift)
    root = _pilot.repository_root()
    source_hash = _pilot.scientific_source_hash(root)
    verify_validation_sources(validation, root, scientific_hash=source_hash)
    heldout_validation = None
    if heldout_rows:
        heldout_validation = verify_validation(args.heldout_validation_summary, args.heldout_structures,
            [row['structure_id'] for row in heldout_rows], tolerance=args.implicit_tolerance,
            rescue_level_shift=args.scf_rescue_level_shift)
        verify_validation_sources(heldout_validation, root, scientific_hash=source_hash)
    jax, experiment_class = _pilot.numerical_runtime('cuda')
    initial, evaluators, value_evaluators, manifests, model_config = make_evaluators(rows + heldout_rows, args, jax, experiment_class)
    for record in validation['results'] + (heldout_validation['results'] if heldout_validation else []):
        if manifests[record['structure_id']]['cache_sha256'] != record.get('cache_sha256'):
            raise ValueError(f"{record['structure_id']}: training DF cache differs from validation.")
    identity = dict(structures_sha256=_pilot.file_hash(args.structures),
        structure_ids=[row['structure_id'] for row in rows], caches=manifests,
        scientific_source_sha256=source_hash, source_sha256=_pilot.file_hash(__file__),
        experiment_source_sha256=_pilot.file_hash(Path(__file__).parents[1]/'training/objective.py'),
        validation_source_sha256=_pilot.file_hash(Path(__file__).with_name('validate.py')),
        validation_summary_sha256=_pilot.file_hash(args.validation_summary),
        model_config=json.loads(json.dumps(model_config)), seed=args.seed,
        parameter_count=int(initial.size), method='RHF', jk_backend='df', core_primitives=6,
        auxbasis=args.auxbasis, dtype='float64', implicit_tolerance=args.implicit_tolerance,
        scf_rescue_level_shift=args.scf_rescue_level_shift,
        optimizer='Optax clip_by_global_norm(1.0) + Adam', learning_rate=args.learning_rate,
        loss='equal-weight mean of all selected reconverged total energies in Hartree',
        scf_config=dict(max_cycle=150, conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9),
        implicit_config=dict(tolerance=args.implicit_tolerance, max_iter=100, restart=40))
    identity.update(cache_storage=args.cache_storage, final_learning_rate=args.final_learning_rate,
        learning_rate_schedule='cosine' if args.final_learning_rate is not None else 'constant',
        learning_rate_decay_updates=args.epochs if args.final_learning_rate is not None else None,
        initialization='fresh seeded MACE with trainable reference bias; no earlier weights',
        heldout_structure_ids=[row['structure_id'] for row in heldout_rows],
        heldout_structures_sha256=_pilot.file_hash(args.heldout_structures) if heldout_rows else None,
        heldout_validation_summary_sha256=_pilot.file_hash(args.heldout_validation_summary) if heldout_rows else None,
        validation_interval=args.validation_interval if heldout_rows else None,
        gradient_batch='full train dataset, equal mean, one Adam update per epoch')
    from importlib.metadata import version
    metadata = dict(identity=identity, target_epochs=args.epochs, device_kinds=[d.device_kind for d in jax.devices()],
        devices=[str(d) for d in jax.devices()], backend=jax.default_backend(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), imported_sources=_pilot.verify_source_imports(root),
        python_version=platform.python_version(), hostname=socket.gethostname(),
        parameter_dtype=str(initial.dtype),
        train_count=len(rows), heldout_count=len(heldout_rows),
        fixed_cache_storage={'device':'all fixed caches on GPU',
            'host':'full NumPy arrays on CPU; NPZ is not memory mapped',
            'mmap':'read-only numeric memmap views of original ZIP_STORED NPZ; OS can reclaim file-backed pages; current structure transferred to GPU'}[args.cache_storage],
        mace_source=str(Path(__import__('mace_jax').__file__).resolve()),
        versions={name: version(name) for name in ('jax', 'jaxlib', 'flax', 'optax', 'e3nn-jax', 'mace-jax')})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _pilot.atomic_json(args.output_dir / 'metadata.json', metadata)
    def evaluate(vector):
        _pilot.atomic_json(args.output_dir / 'batch_progress.json', dict(completed=0, total=len(rows), molecules=[]))
        return evaluate_batch(vector, evaluators[:len(rows)], progress=lambda records: _pilot.atomic_json(
            args.output_dir / 'batch_progress.json', dict(completed=len(records), total=len(rows), molecules=records)))
    def heldout(vector):
        _pilot.atomic_json(args.output_dir / 'heldout_progress.json', dict(completed=0, total=len(heldout_rows), molecules=[]))
        return evaluate_values(vector, value_evaluators[len(rows):], progress=lambda records: _pilot.atomic_json(
            args.output_dir / 'heldout_progress.json', dict(completed=len(records), total=len(heldout_rows), molecules=records)))
    summary = run_training(initial, evaluate, Adam(args.learning_rate,
        final_learning_rate=args.final_learning_rate, epochs=args.epochs), args.output_dir,
        identity, epochs=args.epochs, resume=args.resume,
        heldout=heldout if heldout_rows else None, validation_interval=args.validation_interval)
    return int(summary['status'] != 'complete')


if __name__ == "__main__":
    raise SystemExit(main())
