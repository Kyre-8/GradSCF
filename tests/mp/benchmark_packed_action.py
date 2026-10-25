"""Matched dense-expansion and packed-tile actions, not a full MP benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.mp import _tensor, _tensor_packed
from gradscf.ci import direct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--nmo", type=int, default=6)
    parser.add_argument("--nocc", type=int, default=3)
    parser.add_argument("--source-rank", type=int, default=6)
    parser.add_argument("--target-rank", type=int, default=2)
    parser.add_argument("--tile", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(2423)
    n = args.nmo
    arrays = (rng.normal(size=(n,) * 4) * .02,
              rng.normal(size=(n, n)) * .1)
    space = _tensor.make_tensor_space(n, args.nocc, max_excitation=args.source_rank,
                                     packed_only=args.packed)
    arrays += (rng.normal(size=space.size),)
    report = dict(complete=False, scope="single_action_not_complete_MP",
        packed=args.packed, nmo=n, nocc=args.nocc, source_rank=args.source_rank,
        target_rank=args.target_rank, tile=args.tile, seed=2423,
        coefficients=space.size, dense_elements=space.dense_elements,
        dtype=str(arrays[0].dtype), python=platform.python_version(),
        platform=platform.platform(),
        input_sha256=[hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
                      for x in arrays],
        affinity=sorted(os.sched_getaffinity(0)), backend=jax.default_backend(),
        jax=jax.__version__, numpy=np.__version__,
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for p in (__file__, _tensor.__file__, _tensor_packed.__file__, direct.__file__)})
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    g, f, c = map(jnp.asarray, arrays)

    def apply(g, f, c):
        action = _tensor.build_tensor_action(g, (f,), space)
        if args.packed:
            return action.apply_packed(c, max_input_rank=args.source_rank,
                max_output_rank=args.target_rank, max_tile_elements=args.tile)
        return action.apply(c, max_input_rank=args.source_rank,
                            max_output_rank=args.target_rank)

    start = perf_counter()
    executable = jax.jit(apply).lower(g, f, c).compile()
    report["compile_seconds"] = perf_counter() - start
    memory = executable.memory_analysis()
    report["xla_memory_bytes"] = {key: int(getattr(memory, key)) for key in
        ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")}
    start = perf_counter()
    value = jax.block_until_ready(executable(g, f, c))
    report["first_seconds"] = perf_counter() - start
    samples = []
    for _ in range(7):
        start = perf_counter()
        value = jax.block_until_ready(executable(g, f, c))
        samples.append(perf_counter() - start)
    report.update(complete=bool(np.isfinite(value).all()),
        warm_seconds=samples, warm_median_seconds=float(np.median(samples)),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    np.savez(args.output.with_suffix(".npz"), action=np.asarray(value))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["complete"]:
        raise RuntimeError("Nonfinite packed-action result")


if __name__ == "__main__":
    main()
