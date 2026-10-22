"""Time the same small public MP workload; all physics comes from src.

Run each execution/source variant in a fresh process. Public first-call time
includes tracing/compilation. Only the explicit outer-JIT mode reports isolated
lowering and compilation times; neither is counted as warm execution time.
"""
import argparse
import hashlib
import json
import os
import resource
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import mp
from gradscf.mp import _tensor, series
from test_series_staging import model


def ready(result):
    return jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                        result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", choices=("public", "outer-jit"), default="public")
    parser.add_argument("--algorithm", choices=("series", "tensor"), default="tensor")
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    h, g, no = model()
    cfg = mp.MPConfig(order=args.order, algorithm=args.algorithm)
    evaluate = lambda hh, gg: mp.run_mp(hh, gg, nocc=no, config=cfg)
    timings = {"lower_seconds": None, "compile_seconds": None, "xla_memory_bytes": None}
    if args.execution == "outer-jit":
        start = time.perf_counter()
        lowered = jax.jit(evaluate).lower(h, g)
        timings["lower_seconds"] = time.perf_counter()-start
        start = time.perf_counter()
        evaluate = lowered.compile()
        timings["compile_seconds"] = time.perf_counter()-start
        memory = evaluate.memory_analysis()
        timings["xla_memory_bytes"] = {name: int(getattr(memory, name)) for name in
            ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
             "alias_size_in_bytes")}
    start = time.perf_counter()
    result = ready(evaluate(h, g))
    timings["first_seconds"] = time.perf_counter()-start
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        result = ready(evaluate(h, g))
        samples.append(time.perf_counter()-start)
    timings["warm_seconds"] = samples
    timings["warm_median_seconds"] = statistics.median(samples)
    data = dict(execution=args.execution, algorithm=args.algorithm, order=args.order,
                nmo=h.shape[0], nocc=no, dtype=str(h.dtype), backend=jax.default_backend(),
                devices=list(map(str, jax.devices())), jax=jax.__version__,
                affinity=sorted(os.sched_getaffinity(0)), config=repr(cfg),
                input_sha256=[hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in (h, g)],
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (Path(__file__), Path(model.__code__.co_filename), Path(series.__file__),
                     Path(_tensor.__file__), Path(mp.__file__),
                     Path(series.__file__).with_name('_series_memory.py')) if p.exists()},
                valid=bool(result.valid), corrections=np.asarray(result.corrections).tolist(),
                t2_shape=list(result.t2.shape), coefficients_is_none=result.wavefunction_coefficients is None,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **timings)
    args.output.write_text(json.dumps(data, indent=2)+"\n")
    print(json.dumps(data, indent=2), flush=True)


if __name__ == "__main__":
    main()
