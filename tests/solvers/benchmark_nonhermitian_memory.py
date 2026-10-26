"""Small matched CPU primal benchmark; run each mode in a fresh process.

PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python \
    tests/solvers/benchmark_nonhermitian_memory.py {original,budgeted}
RSS includes the process runtime/compiled executable, separately from the static
known-array estimate. Compilation precedes all warm timing/RSS samples.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.solvers import LinearOperator, NonHermitianSolverConfig, solve_nonhermitian
from gradscf.solvers.memory import plan_nonhermitian


def rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])*1024
    raise RuntimeError('Linux process RSS is unavailable')


def main(mode):
    if mode not in {'original', 'budgeted'}:
        raise ValueError('Expected original or budgeted')
    n = 256
    diagonal = jnp.linspace(.4, 10., n)
    matrix = (jnp.diag(diagonal)+jnp.diag(jnp.full(n-1, .005), 1)
              +jnp.diag(jnp.full(n-1, .001), -1))
    op = LinearOperator(matrix.shape, matrix.dtype, lambda v: matrix@v,
        diagonal=diagonal, matmat=lambda v: matrix@v,
        transpose_matvec=lambda v: matrix.T@v)
    cfg = NonHermitianSolverConfig(method='davidson', nroots=2, guard_roots=1,
                                   max_space=32, maxiter=160, atol=1e-9)
    resident = matrix.nbytes+diagonal.nbytes
    minimum = 4*(cfg.nroots+cfg.guard_roots)+2
    _, minimum_bytes, _ = plan_nonhermitian(n, matrix.dtype,
                                            replace(cfg, max_space=minimum), 8.)
    _, original_bytes, _ = plan_nonhermitian(n, matrix.dtype, cfg, 8.,
                                             resident_bytes=resident)
    limit = (resident+minimum_bytes+2*n*8*cfg.nroots+64)*1.25/1e6
    planned, reservation, columns = plan_nonhermitian(n, matrix.dtype, cfg,
                                      limit, resident_bytes=resident)
    options = {} if mode == 'original' else dict(max_memory=limit,
                                                 reserved_bytes=matrix.nbytes)
    solve = jax.jit(lambda: solve_nonhermitian(op, config=cfg, **options))
    initial_rss = rss_bytes()
    started = time.perf_counter()
    result = jax.block_until_ready(solve())
    compile_and_first = time.perf_counter()-started
    after_compile = rss_bytes()
    stopped = threading.Event()
    samples = [after_compile]

    def monitor():
        while not stopped.wait(.002):
            samples.append(rss_bytes())

    sampler = threading.Thread(target=monitor, daemon=True)
    sampler.start()
    elapsed = []
    try:
        for _ in range(5):
            started = time.perf_counter()
            result = jax.block_until_ready(solve())
            elapsed.append(time.perf_counter()-started)
            samples.append(rss_bytes())
    finally:
        stopped.set()
        sampler.join()
    assert np.all(result.converged), 'Benchmark Ritz solve did not converge'
    record = dict(mode=mode, n=n, dtype=str(matrix.dtype), backend=jax.default_backend(),
        platform=platform.platform(), jax_version=jax.__version__,
        cpu_model=next(line.split(':', 1)[1].strip() for line in
                       Path('/proc/cpuinfo').read_text().splitlines()
                       if line.startswith('model name')),
        matrix_sha256=hashlib.sha256(np.asarray(matrix).tobytes()).hexdigest(),
        cpu_affinity=sorted(os.sched_getaffinity(0)), seed=cfg.seed, nroots=cfg.nroots,
        guard_roots=cfg.guard_roots, maxiter=cfg.maxiter, atol=cfg.atol,
        original_max_space=cfg.max_space,
        planned_max_space=cfg.max_space if mode == 'original' else planned.max_space,
        action_columns=None if mode == 'original' else columns,
        max_memory_mb=None if mode == 'original' else limit,
        resident_bytes=resident,
        planned_solver_bytes=original_bytes if mode == 'original' else reservation,
        compile_and_first_seconds=compile_and_first, warm_seconds=elapsed,
        warm_median_seconds=statistics.median(elapsed), initial_rss_bytes=initial_rss,
        after_compile_rss_bytes=after_compile, warm_peak_rss_bytes=max(samples),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        values=np.asarray(result.values).tolist(),
        residuals=np.asarray(result.residual_norms).tolist(),
        left_residuals=np.asarray(result.left_residual_norms).tolist(),
        guard_residuals=np.asarray(result.guard_residual_norms).tolist(),
        response_valid=np.asarray(result.response_valid).tolist(),
        iterations=int(result.iterations), restarts=int(result.restarts),
        subspace_dimension=int(result.subspace_dimension), sha256={})
    root = Path(__file__).resolve().parents[2]
    for name in ('src/gradscf/solvers/memory.py',
                 'src/gradscf/solvers/eigen/nonhermitian.py',
                 'src/gradscf/solvers/eigen/nonhermitian_davidson.py',
                 'tests/solvers/benchmark_nonhermitian_memory.py'):
        record['sha256'][name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
