"""Compare direct-CI column scheduling on a fixed synthetic Hamiltonian.

Run source variants in fresh processes; this measures a sigma action, not a
converged CI calculation. All Hamiltonian algebra comes from src.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.ci.direct import build_direct_hamiltonian
from gradscf.ci.space import make_ci_space
from test_direct_action import _integrals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nmo', type=int, default=12)
    parser.add_argument('--nocc', type=int, default=4)
    parser.add_argument('--columns', type=int, default=24)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    h, g = _integrals(args.nmo)
    space = make_ci_space(args.nmo, args.nocc, max_excitation=2)
    vectors = jnp.asarray(np.random.default_rng(2312).normal(
        size=(space.size, args.columns)))
    operator = build_direct_hamiltonian(h, g, space)
    started = time.perf_counter()
    executable = jax.jit(operator.apply).lower(vectors).compile()
    compile_seconds = time.perf_counter()-started
    result = jax.block_until_ready(executable(vectors))
    times = []
    for _ in range(5):
        started = time.perf_counter()
        result = jax.block_until_ready(executable(vectors))
        times.append(time.perf_counter()-started)
    if not np.isfinite(np.asarray(result)).all():
        raise RuntimeError('Nonfinite direct CI action')
    analysis = executable.memory_analysis()
    import gradscf.ci.direct as direct
    import gradscf.solvers.memory as shared_memory
    sources = (Path(direct.__file__), Path(shared_memory.__file__),
               Path(__file__), Path(__file__).with_name('test_direct_action.py'))
    data = dict(nmo=args.nmo, nocc=args.nocc, determinants=space.size,
        columns=args.columns, seed=2312, dtype=str(vectors.dtype),
        backend=jax.default_backend(), affinity=sorted(os.sched_getaffinity(0)),
        command=sys.argv, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sources},
        cpu=next(line.split(':', 1)[1].strip() for line in
                 Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')},
        jax=jax.__version__, compile_seconds=compile_seconds, warm_seconds=times,
        warm_median_seconds=statistics.median(times),
        xla_temp_bytes=analysis.temp_size_in_bytes,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        input_sha256=hashlib.sha256(np.asarray(vectors).tobytes()).hexdigest())
    np.savez(args.output.with_suffix('.npz'), result=np.asarray(result))
    args.output.write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(data), flush=True)


if __name__ == '__main__':
    main()
