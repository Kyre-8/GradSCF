"""Matched full-then-slice versus requested-block spatial density export."""
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

from gradscf import fci


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['full','block'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    start=time.perf_counter()
    space=fci.make_fci_space(8,(2,2))
    c=jnp.asarray(np.random.default_rng(772).normal(size=space.shape))
    jax.block_until_ready(c)
    prepare_seconds=time.perf_counter()-start
    axes=((0,1),(2,3),(4,5),(6,7))
    def value(x):
        return (fci.make_rdm2(x,space)[jnp.ix_(*map(jnp.asarray,axes))] if args.mode == 'full'
                else fci.make_rdm2_block(x,space,axes))
    start=time.perf_counter()
    lowered=jax.jit(value).lower(c)
    lower_seconds=time.perf_counter()-start
    start=time.perf_counter()
    executable=lowered.compile()
    compile_seconds=time.perf_counter()-start
    start=time.perf_counter()
    result=jax.block_until_ready(executable(c))
    first_seconds=time.perf_counter()-start
    samples=[]
    for _ in range(10):
        start=time.perf_counter()
        jax.block_until_ready(executable(c))
        samples.append(time.perf_counter()-start)
    memory=executable.memory_analysis()
    np.save(args.output.with_suffix('.npy'),np.asarray(result))
    import gradscf.fci.rdm as source
    record=dict(mode=args.mode,norb=8,nelec=(2,2),determinants=space.size,indices=axes,
        seed=772,dtype=str(result.dtype),backend=jax.default_backend(),jax=jax.__version__,
        command=sys.argv,affinity=sorted(os.sched_getaffinity(0)),
        input_sha256=hashlib.sha256(np.asarray(c).tobytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
        prepare_seconds=prepare_seconds,lower_seconds=lower_seconds,
        compile_seconds=compile_seconds,first_seconds=first_seconds,warm_seconds=samples,
        warm_median_seconds=statistics.median(samples),
        xla_temp_bytes=memory.temp_size_in_bytes,xla_output_bytes=memory.output_size_in_bytes,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cpu=next(line for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        environment={k:os.environ.get(k) for k in ('JAX_PLATFORMS','CUDA_VISIBLE_DEVICES','XLA_FLAGS',
            'JAX_NUM_CPU_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        boundary='Forward algebraic density export only; compiler may remove unused full-output work.')
    args.output.write_text(json.dumps(record,indent=2))
    print(json.dumps(record))


if __name__ == '__main__':
    main()
