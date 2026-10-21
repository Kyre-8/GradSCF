"""Matched CPU spin virtual-action measurements; algorithms stay in src."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from test_spin_virtual_action import _action, _inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("implementation", choices=("explicit", "factored"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--nocc", type=int, default=8)
    parser.add_argument("--nvir", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    values = _inputs(args.nocc, args.nvir)
    rng = np.random.default_rng(6525)
    x = jnp.asarray(rng.normal(size=(args.nocc, args.nocc, args.nvir, args.nvir))*.03)
    digest = hashlib.sha256()
    for a in (*values, x):
        digest.update(np.asarray(a).tobytes())
    function = jax.jit(lambda v, x: _action(v, x, args.implementation == "explicit"))
    start = time.perf_counter()
    lowered = function.lower(values, x)
    lower_time = time.perf_counter()-start
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_time = time.perf_counter()-start
    start = time.perf_counter()
    result = jax.block_until_ready(compiled(values, x))
    first_time = time.perf_counter()-start
    timings = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(values, x))
        timings.append(time.perf_counter()-start)
    memory = compiled.memory_analysis()
    np.save(args.output.with_suffix(".npy"), np.asarray(result))
    report = dict(implementation=args.implementation, nocc=args.nocc, nvir=args.nvir,
        input_sha256=digest.hexdigest(), dtype=str(result.dtype), seed=6521,
        rhs_seed=6525, backend=jax.default_backend(), devices=list(map(str, result.devices())),
        affinity=sorted(os.sched_getaffinity(0)), jax=jax.__version__,
        python=platform.python_version(), command=sys.argv,
        cpu=next(s.partition(":")[2].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                 if s.startswith("model name")),
        environment={k: os.environ.get(k) for k in ("JAX_PLATFORMS", "JAX_ENABLE_X64",
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        lower_seconds=lower_time, compile_seconds=compile_time, first_seconds=first_time,
        warm_seconds=timings, warm_median_seconds=float(np.median(timings)),
        xla_memory={k: getattr(memory, k) for k in ("argument_size_in_bytes",
            "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")},
        process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        finite=bool(np.isfinite(result).all()), output_shape=list(result.shape),
        note="Seeded algebraic action, not a physical molecule or whole CC solve.")
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
