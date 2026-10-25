"""One fixed EA intermediate/action pair; no SCF or EOM root solve is timed."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.cc.eom import _charged
from gradscf.cc.integrals import CCIntegrals


def main(mode, legacy_path):
    jax.config.update("jax_enable_x64", True)
    if mode not in {"legacy", "action"}:
        raise ValueError("Use legacy or action")
    paths = (Path(_charged.__file__), Path(legacy_path), Path(__file__))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    module = _charged
    if mode == "legacy":
        spec = importlib.util.spec_from_file_location("gradscf.cc.eom._benchmark_original", legacy_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    no, nv = 8, 64
    n = no+nv
    rng = np.random.default_rng(6562)
    shapes = ((n, n), (n,), (no,)*4, (no, nv, no, no),
              (no, nv, no, nv), (no, no, nv, nv), (no, nv, nv, no),
              (no, nv, nv, nv), (nv,)*4, ())
    digest = hashlib.sha256()
    def array(shape):
        value = np.asarray(rng.normal(scale=.01, size=shape))
        digest.update(value.tobytes())
        return jnp.asarray(value)
    ints = CCIntegrals(*(array(s) for s in shapes))
    t1, t2, r1, r2 = (array(s) for s in ((no, nv), (no, no, nv, nv), (nv,), (no, nv, nv)))
    start = time.perf_counter()
    prepare = jax.jit(lambda a, b, g: vars(module.intermediates(a, b, g, "ea")))
    state = jax.block_until_ready(prepare(t1, t2, ints))
    prepare_seconds = time.perf_counter()-start
    start = time.perf_counter()
    lowered = jax.jit(lambda w, a, b: module.ea_action(a, b, SimpleNamespace(**w))).lower(state, r1, r2)
    executable = lowered.compile()
    compile_seconds = time.perf_counter()-start
    start = time.perf_counter()
    output = jax.block_until_ready(executable(state, r1, r2))
    first_seconds = time.perf_counter()-start
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        jax.block_until_ready(executable(state, r1, r2))
        samples.append(time.perf_counter()-start)
    memory = executable.memory_analysis()
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if before != after:
        raise RuntimeError("Benchmark source changed")
    record = dict(mode=mode, scope="seeded_EA_prepare_and_one_action", nocc=no, nvir=nv,
        seed=6562, dtype=str(t1.dtype), backend=jax.default_backend(), jax=jax.__version__,
        input_sha256=digest.hexdigest(), command=sys.argv, hostname=platform.node(),
        cpu_model=next(line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                       if line.startswith("model name")), affinity=sorted(os.sched_getaffinity(0)),
        exclusive_node=False, source_before=before, source_after=after,
        prepare_seconds=prepare_seconds, action_lower_compile_seconds=compile_seconds,
        first_action_seconds=first_seconds, warm_seconds=samples,
        warm_median_seconds=statistics.median(samples),
        action_argument_bytes=memory.argument_size_in_bytes,
        action_output_bytes=memory.output_size_in_bytes, action_temp_bytes=memory.temp_size_in_bytes,
        measurement_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    np.savez(f"eom-virtual-{mode}.npz", *map(np.asarray, output))
    Path(f"eom-virtual-{mode}.json").write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
