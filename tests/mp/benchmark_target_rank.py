"""Measure one production tensor action, with all or only requested targets.

This is a contraction benchmark, not a complete MP solver benchmark. Inputs
and full output vectors are retained for a comparison on the requested ranks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.mp import _tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-rank", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(2325)
    nmo, nocc = 6, 3
    space = _tensor.make_tensor_space(nmo, nocc, max_excitation=6)
    g = rng.normal(size=(nmo,)*4)*.01
    f = np.diag(np.linspace(-1.5, 1.5, nmo))
    vector = rng.normal(size=space.size)
    arrays = (g, f, vector)
    hashes = [hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for a in arrays]
    g, f, vector = map(jnp.asarray, arrays)

    def action(g, f, vector):
        return _tensor.build_tensor_action(g, (f,), space).apply(
            vector, max_input_rank=6, max_output_rank=args.output_rank)

    report = dict(complete=False, scope="one_tensor_action_not_complete_MP",
        norb=nmo, nocc=nocc, determinants=space.size, seed=2325,
        output_rank=args.output_rank, input_sha256=hashes,
        backend=jax.default_backend(), jax=jax.__version__, numpy=np.__version__,
        affinity=sorted(os.sched_getaffinity(0)), dtype=str(vector.dtype),
        source_sha256={str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for p in (__file__, _tensor.__file__)})
    start = perf_counter()
    executable = jax.jit(action).lower(g, f, vector).compile()
    report["compile_seconds"] = perf_counter()-start
    analysis = executable.memory_analysis()
    report["xla_memory_bytes"] = {name: int(getattr(analysis, name)) for name in
        ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")}
    start = perf_counter()
    value = jax.block_until_ready(executable(g, f, vector))
    report["first_seconds"] = perf_counter()-start
    samples = []
    for _ in range(7):
        start = perf_counter()
        value = jax.block_until_ready(executable(g, f, vector))
        samples.append(perf_counter()-start)
    value = np.asarray(value)
    report.update(complete=bool(np.isfinite(value).all()), warm_seconds=samples,
                  warm_median_seconds=float(np.median(samples)),
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), action=value, ranks=np.asarray(space.ranks))
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
    if not report["complete"]:
        raise RuntimeError("Nonfinite tensor action")


if __name__ == "__main__":
    main()
