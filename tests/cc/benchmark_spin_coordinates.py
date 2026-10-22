"""Matched coordinate-boundary probe; this does not measure a CC residual."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.cc.spin_amplitudes import SpinAmplitudeSpace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('old', 'blocks'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    space = SpinAmplitudeSpace((8, 7), (80, 79))
    shapes = ((8, 80), (7, 79), (8, 8, 80, 80), (8, 7, 80, 79), (7, 7, 79, 79))
    rng = np.random.default_rng(6551)
    arrays = tuple(rng.normal(scale=.01, size=shape) for shape in shapes)
    input_hash = hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest()
    inputs = tuple(jnp.asarray(a) for a in arrays)

    def boundary(*blocks, mode):
        if mode == 'old':
            vector = space.pack(*space.from_blocks(blocks[:2], blocks[2:], jnp.float64))
            singles, doubles = space.to_blocks(*space.unpack(vector))
        else:
            vector = space.pack_blocks(blocks[:2], blocks[2:], jnp.float64)
            singles, doubles = space.unpack_blocks(vector)
        return vector, *singles, *doubles

    start = time.monotonic()
    executable = jax.jit(lambda *x: boundary(*x, mode=args.mode)).lower(*inputs).compile()
    result = jax.block_until_ready(executable(*inputs))
    first = time.monotonic()-start
    samples = []
    for _ in range(5):
        start = time.monotonic()
        result = jax.block_until_ready(executable(*inputs))
        samples.append(time.monotonic()-start)
    # Capture the primary process peak before the independent reference executes.
    primary_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current = next(int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines()
                   if line.startswith('VmRSS:'))
    memory = executable.memory_analysis()
    other = 'old' if args.mode == 'blocks' else 'blocks'
    reference = jax.jit(lambda *x: boundary(*x, mode=other))(*inputs)
    for actual, expected in zip(result, reference):
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-14)
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), 'unknown')
    record = dict(mode=args.mode, scope='physical block coordinate boundary only',
        backend=jax.default_backend(), dtype=str(result[0].dtype), jax=jax.__version__,
        cpu=cpu, affinity=sorted(os.sched_getaffinity(0)), seed=6551,
        input_shapes=shapes, input_sha256=input_hash, first_seconds=first,
        warm_seconds=samples, warm_median_seconds=statistics.median(samples),
        primary_peak_rss_kib=primary_peak, primary_current_rss_kib=current,
        compiled_temp_bytes=memory.temp_size_in_bytes,
        compiled_argument_bytes=memory.argument_size_in_bytes,
        compiled_output_bytes=memory.output_size_in_bytes,
        reference_max_abs_error=max(float(jnp.max(jnp.abs(a-b), initial=0.))
                                    for a, b in zip(result, reference)),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,
            ('src/gradscf/cc/spin_amplitudes.py', 'tests/cc/benchmark_spin_coordinates.py'))})
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
