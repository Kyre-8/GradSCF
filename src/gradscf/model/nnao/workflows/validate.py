"""Prepare and validate NNAO training inputs."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import platform as host_platform

from ..runtime import (read_structures, file_hash, scientific_source_hash,
    numerical_runtime, repository_root)
from ..training.validation import run_suite, prepare_cache, validate

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare-cache', 'validate'))
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--platform', choices=('cpu', 'cuda'))
    parser.add_argument('--auxbasis', default='def2-universal-jkfit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--adam-steps', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--fd-step', type=float, default=1e-4)
    parser.add_argument('--fd-atol', type=float, default=2e-6)
    parser.add_argument('--fd-rtol', type=float, default=2e-5)
    parser.add_argument('--implicit-tolerance', type=float, default=1e-9)
    parser.add_argument('--scf-rescue-level-shift', type=float, default=0.0)
    parser.add_argument('--cache-storage', choices=('device', 'host', 'mmap'), default='device')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.implicit_tolerance) or args.implicit_tolerance <= 0:
        parser.error('Implicit tolerance must be finite and positive.')
    if not math.isfinite(args.scf_rescue_level_shift) or args.scf_rescue_level_shift < 0:
        parser.error('SCF rescue level shift must be finite and nonnegative.')
    if args.adam_steps < 0 or min(args.learning_rate, args.fd_step, args.fd_atol, args.fd_rtol) <= 0:
        parser.error('Adam steps must be nonnegative; learning rate and FD settings must be positive.')
    platform = args.platform or ('cpu' if args.mode == 'prepare-cache' else 'cuda')
    if args.mode == 'prepare-cache' and platform != 'cpu':
        parser.error('Native integral cache preparation requires --platform cpu.')
    if args.mode == 'prepare-cache' and args.cache_storage != 'device':
        parser.error('Native integral cache preparation requires --cache-storage device.')
    rows = read_structures(args.structures, ids=args.ids, limit=args.limit)
    config = dict(mode=args.mode, platform=platform, seed=args.seed,
        source_sha256=file_hash(__file__), experiment_source_sha256=file_hash(
            Path(__file__).parents[1]/'training/objective.py'),
        structures_sha256=file_hash(args.structures), auxbasis=args.auxbasis,
        cache_dir=str(args.cache_dir.resolve()), adam_steps=args.adam_steps,
        learning_rate=args.learning_rate, fd_step=args.fd_step,
        fd_atol=args.fd_atol, fd_rtol=args.fd_rtol,
        implicit_tolerance=args.implicit_tolerance, cache_storage=args.cache_storage,
        scf_rescue_level_shift=args.scf_rescue_level_shift)
    from importlib.metadata import PackageNotFoundError, version
    config['python_version'] = host_platform.python_version()
    config['scientific_source_sha256'] = scientific_source_hash(repository_root())
    config['software_versions'] = {}
    for package in ('jax', 'jaxlib', 'flax', 'optax', 'e3nn-jax', 'mace-jax'):
        try:
            config['software_versions'][package] = version(package)
        except PackageNotFoundError:
            config['software_versions'][package] = 'not installed as a distribution'
    jax, experiment_class = numerical_runtime(platform)

    def operation(row, directory):
        common = dict(cache_dir=args.cache_dir, auxbasis=args.auxbasis,
                      jax=jax, experiment_class=experiment_class,
                      implicit_tolerance=args.implicit_tolerance,
                      scf_rescue_level_shift=args.scf_rescue_level_shift)
        if args.mode == 'prepare-cache':
            return prepare_cache(row, directory, **common)
        return validate(row, directory, seed=args.seed, adam_steps=args.adam_steps,
            learning_rate=args.learning_rate, fd_step=args.fd_step,
            fd_atol=args.fd_atol, fd_rtol=args.fd_rtol,
            cache_storage=args.cache_storage, **common)

    summary = run_suite(rows, args.output_dir, config, operation, resume=not args.no_resume)
    print(f"Passed {summary['passed']}; failed {summary['failed']}", flush=True)
    return summary['exit_code']


if __name__ == "__main__":
    raise SystemExit(main())
