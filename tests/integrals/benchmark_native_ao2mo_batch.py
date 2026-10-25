"""Matched single/batch native AO->MO timing in separate frozen processes."""
import argparse
import hashlib
import json
from math import prod
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from gradscf import ao2mo, integrals
from gradscf.ao2mo import native
from gradscf.integrals.backends.native.jk import NativeDirectBasis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('single', 'batch'), required=True)
    parser.add_argument('--requests', choices=('cc', 'ovov'), default='cc')
    parser.add_argument('--basis', default='6-31g*')
    parser.add_argument('--max-memory', type=float, default=60000.)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    geometry = 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587'
    topology, parameters = integrals.prepare_basis(geometry, basis=args.basis, cart=True)
    basis = NativeDirectBasis(integrals.make_plan(topology, backend='native'), parameters)
    rng = np.random.default_rng(6205)
    q, _ = np.linalg.qr(rng.normal(size=(topology.nao, topology.nao)))
    nocc = min(5, topology.nao//2)
    coefficients = (jnp.asarray(q[:, :nocc]), jnp.asarray(q[:, nocc:]))
    requests = ((0, 0, 0, 0), (0, 1, 0, 0), (0, 1, 0, 1),
                (0, 0, 1, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 1, 1, 1))
    if args.requests == 'ovov':
        requests = ((0, 1, 0, 1),)

    def function(cs):
        if args.mode == 'batch':
            return ao2mo.transform_eri_blocks(cs, requests, direct_basis=basis,
                max_memory=args.max_memory)
        return tuple(ao2mo.transform_eri_block(tuple(cs[i] for i in request),
            direct_basis=basis, max_memory=args.max_memory) for request in requests)

    report = dict(complete=False, command=sys.argv, mode=args.mode, request_set=args.requests,
        geometry=geometry, geometry_unit='Angstrom', basis=args.basis, cart=True,
        coefficient_seed=6205, coefficient_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
        nao=topology.nao, nocc=nocc, nvir=topology.nao-nocc, requests=requests,
        dtype='float64', max_memory_mb=args.max_memory, reserved_bytes=0,
        affinity=sorted(os.sched_getaffinity(0)), jax=jax.__version__,
        device=str(coefficients[0].device), environment={key: os.getenv(key) for key in
            ('JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
             'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}, source_sha256={})
    report['slab_plans'] = []
    original_plan = native._slab_plan

    def record_plan(*arguments, **options):
        plan = original_plan(*arguments, **options)
        if plan is not None:
            b, cs = arguments[:2]
            requests = arguments[6] if len(arguments) > 6 else ((0, 1, 2, 3),)
            qg, rg, nq, nr = plan
            sizes = tuple(c.shape[1] for c in cs)
            first = dict.fromkeys(r[0] for r in requests)
            second = dict.fromkeys((r[0], r[3]) for r in requests)
            half = dict.fromkeys((r[0], r[1], r[3]) for r in requests)
            report['slab_plans'].append(dict(q_groups=len(qg), r_groups=len(rg),
                nq=nq, nr=nr, native_slab_calls=len(qg)*len(rg),
                slab_staging_bytes=3*8*b.nao**2*nq*nr,
                first_bytes=8*sum(sizes[i] for i in first)*nq*nr*b.nao,
                second_bytes=8*sum(sizes[i]*sizes[j] for i, j in second)*nq*nr,
                half_buffers_bytes=3*8*sum(sizes[i]*sizes[j]*sizes[k]
                    for i, j, k in half)*nr,
                output_buffers_bytes=3*8*sum(prod(sizes[i] for i in r) for r in requests)))
        return plan

    native._slab_plan = record_plan
    package = Path(ao2mo.__file__).parent
    for path in (*package.glob('*.py'), package.parent/'_memory.py'):
        report['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    native_library = package.parent/'integrals/_native/lib/libgradscf_integrals.so'
    report['native_sha256'] = hashlib.sha256(native_library.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.write_text(json.dumps(report, indent=2)+'\n')

    save()
    started = perf_counter()
    executable = jax.jit(function).lower(coefficients).compile()
    report['compile_seconds'] = perf_counter()-started
    analysis = executable.memory_analysis()
    report['xla_estimated_bytes'] = {key: getattr(analysis, key, None) for key in
        ('argument_size_in_bytes', 'output_size_in_bytes', 'alias_size_in_bytes',
         'temp_size_in_bytes', 'host_temp_size_in_bytes')}
    save()
    started = perf_counter()
    values = jax.block_until_ready(executable(coefficients))
    report['first_seconds'] = perf_counter()-started
    times = []
    for _ in range(args.repeats):
        started = perf_counter()
        values = jax.block_until_ready(executable(coefficients))
        times.append(perf_counter()-started)
    arrays = tuple(np.asarray(value) for value in values)
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError('Nonfinite AO->MO result')
    np.savez(args.output.with_suffix('.npz'), *arrays)
    report.update(complete=True, warm_seconds=times,
        warm_median_seconds=float(np.median(times)),
        output_shapes=[array.shape for array in arrays],
        output_norms=[float(np.linalg.norm(array)) for array in arrays])
    save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
