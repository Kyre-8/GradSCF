"""Small matched UHF preparation probe; no SCF or molecular scaling claim."""
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

from gradscf.ao2mo import transform_unrestricted_integrals
from gradscf.cc.uccsd import prepare_ucc_integrals
from gradscf.cc.unrestricted_integrals import from_orbitals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('raw', 'selected'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(6514)
    nao, nmo, nocc = 4, 8, (3, 2)
    cs = tuple(jnp.asarray(rng.normal(scale=.2, size=(nao, nmo))) for _ in range(2))
    raw = rng.normal(scale=.04, size=(6, nao, nao))
    factors = jnp.asarray(raw+raw.swapaxes(1, 2))
    raw = rng.normal(scale=.1, size=(nao, nao))
    hcore = jnp.asarray(raw+raw.T)
    hs, gs = transform_unrestricted_integrals(hcore, cs, df_factors=factors, max_memory=2.)
    oracle, occupied, virtual = prepare_ucc_integrals(hs, gs, nocc=nocc, nuclear_repulsion=.7)
    no = sum(occupied)
    # The supplied Focks/energy represent the same solved input for both modes.
    # The legacy route still rebuilds them from its full spatial MO integrals.
    spin_indices = tuple(jnp.concatenate((jnp.arange(sum(occupied[:s]), sum(occupied[:s+1])),
        no+jnp.arange(sum(virtual[:s]), sum(virtual[:s+1])))) for s in range(2))
    fs = tuple(oracle.fock[jnp.ix_(i, i)] for i in spin_indices)
    energy = oracle.reference_energy
    point = (cs, fs, energy, hcore, factors)

    def prepare(c, f, e, h, source):
        if args.mode == 'selected':
            return from_orbitals(c, f, nocc, e, df_factors=source, max_memory=2.)[0]
        one, two = transform_unrestricted_integrals(h, c, df_factors=source, max_memory=2.)
        return prepare_ucc_integrals(one, two, nocc=nocc, nuclear_repulsion=.7)[0]

    start = time.perf_counter()
    executable = jax.jit(prepare).lower(*point).compile()
    compile_s = time.perf_counter()-start
    start = time.perf_counter()
    result = jax.block_until_ready(executable(*point))
    first_execution_s = time.perf_counter()-start
    for actual, expected in zip(result, oracle):
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)
    samples = []
    for _ in range(15):
        start = time.perf_counter()
        jax.block_until_ready(executable(*point))
        samples.append(time.perf_counter()-start)
    output = args.output.with_suffix('.npz')
    np.savez(output, **dict(zip(result._fields, map(np.asarray, result))))
    memory = executable.memory_analysis()
    record = dict(mode=args.mode, description='Algebraic forward preparation, 4 AO/8 MO, DF6, seed6514',
        dtype=str(result.fock.dtype), nocc=nocc, backend=jax.default_backend(),
        device=str(jax.devices()[0]), jax=jax.__version__, compile_s=compile_s,
        first_execution_s=first_execution_s, warm_median_s=statistics.median(samples),
        warm_samples_s=samples, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        xla_temporary_bytes=memory.temp_size_in_bytes, xla_output_bytes=memory.output_size_in_bytes,
        affinity=sorted(os.sched_getaffinity(0)), omp_threads=os.environ.get('OMP_NUM_THREADS'),
        blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'), output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        cpu=next(line for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        limitation='Dense spin vvvv/amplitudes remain; this does not measure molecular SCF, AD or total solver memory.')
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == '__main__':
    main()
