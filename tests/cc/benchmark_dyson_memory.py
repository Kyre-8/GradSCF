"""Matched fixed-frame endpoint component; each mode runs in a fresh process.

This times seeded algebra, not a converged CC/EOM calculation. Supply the exact
original properties.py for the legacy mode; do not use a rewritten endpoint
helper as an old-version baseline. No SCF is run.
"""
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

from gradscf.cc.eom.amplitudes import EOMAmplitudeSpace
from gradscf.cc.eom import properties
from gradscf.cc.ground import METHODS


def main(mode, sector, legacy_path):
    if mode not in {"legacy", "streamed"} or sector not in {"ip", "ea"}:
        raise ValueError("Use legacy/streamed and ip/ea")
    paths = (Path(properties.__file__), Path(legacy_path), Path(__file__))
    def source_hashes():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths}
    before = source_hashes()
    if mode == "legacy":
        spec = importlib.util.spec_from_file_location(
            "gradscf.cc.eom._benchmark_legacy_properties", legacy_path)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        dyson_orbitals = legacy.dyson_orbitals
    else:
        dyson_orbitals = properties.dyson_orbitals
    jax.config.update("jax_enable_x64", True)
    no, nv, roots = 16, 128, 2
    space = EOMAmplitudeSpace(no, nv, sector)
    rng = np.random.default_rng(6541)
    shapes = ((no, nv), (no, no, nv, nv))*2 + ((space.size, roots),)*2
    arrays = tuple(rng.normal(scale=.01, size=shape) for shape in shapes)
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(array.tobytes())
    inputs = tuple(map(jnp.asarray, arrays))

    def evaluate(values):
        t1, t2, l1, l2, right, left = values
        ground = SimpleNamespace(t1=t1, t2=t2, converged=True,
                                  method_id=METHODS.index("ccsd"))
        lambdas = SimpleNamespace(l1=l1, l2=l2, converged=True)
        state = SimpleNamespace(sector=sector, right_vectors=right, left_vectors=left,
            energies=jnp.ones(roots), converged=True, ground_valid=True,
            response_valid=True, vector_response=True)
        # Raise only the legacy allocation guard, keeping the same full space.
        options = dict(max_elements=2*(no+nv)*space.size)
        if mode == "streamed":
            options["max_memory"] = 60000.
        result = dyson_orbitals(ground, lambdas, state, **options)
        return result.left, result.right, result.weights

    start = time.perf_counter()
    lowered = jax.jit(evaluate).lower(inputs)
    lower_seconds = time.perf_counter()-start
    start = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter()-start
    start = time.perf_counter()
    output = jax.block_until_ready(executable(inputs))
    first_seconds = time.perf_counter()-start
    samples = []
    for _ in range(9):
        start = time.perf_counter()
        jax.block_until_ready(executable(inputs))
        samples.append(time.perf_counter()-start)
    memory = executable.memory_analysis()
    after = source_hashes()
    if before != after:
        raise RuntimeError("Benchmark sources changed during measurement")
    cpu_model = next((line.partition(":")[2].strip() for line in
                     Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), "unknown")
    record = dict(kind="seeded_endpoint_component", mode=mode, sector=sector,
        nocc=no, nvir=nv, nroots=roots, eom_size=space.size, dtype=str(inputs[0].dtype),
        backend=jax.default_backend(), seed=6541, input_sha256=digest.hexdigest(),
        affinity=sorted(os.sched_getaffinity(0)), command=sys.argv,
        hostname=platform.node(), cpu_model=cpu_model,
        source_before=before, source_after=after,
        max_elements=2*(no+nv)*space.size,
        jax_version=jax.__version__, numpy_version=np.__version__,
        lower_seconds=lower_seconds, compile_seconds=compile_seconds,
        first_seconds=first_seconds, warm_seconds=samples,
        warm_median_seconds=statistics.median(samples),
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        xla_argument_bytes=memory.argument_size_in_bytes,
        xla_output_bytes=memory.output_size_in_bytes, xla_temp_bytes=memory.temp_size_in_bytes,
        outputs=[np.asarray(x).tolist() for x in output])
    Path(f"dyson-{sector}-{mode}.json").write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
