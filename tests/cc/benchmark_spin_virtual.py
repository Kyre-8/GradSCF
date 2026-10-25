"""One matched UCC residual frame; output arrays stay on the measurement host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('dense', 'blocks', 'verify'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()
    if args.mode == 'verify':
        old, new = (json.loads(p.read_text()) for p in (args.baseline, args.candidate))
        assert old['input_sha256'] == new['input_sha256']
        assert old['source_sha256'] == new['source_sha256']
        for path, data in ((args.baseline, old), (args.candidate, new)):
            assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest() == data['output_sha256']
        maximum = 0.
        with np.load(args.baseline.with_suffix('.npz')) as a, np.load(args.candidate.with_suffix('.npz')) as b:
            for key in a.files:
                np.testing.assert_allclose(a[key], b[key], atol=3e-11, rtol=3e-10)
                maximum = max(maximum, float(np.max(np.abs(a[key]-b[key]))))
        record = dict(input_sha256=old['input_sha256'], max_abs_error=maximum,
            baseline_output_sha256=old['output_sha256'], candidate_output_sha256=new['output_sha256'])
    else:
        import jax
        import jax.numpy as jnp
        from gradscf._memory import array_storage_bytes
        from gradscf.cc.unrestricted_integrals import from_orbitals
        from gradscf.cc.spin_amplitudes import SpinAmplitudeSpace
        from gradscf.cc._spin_equations import residual

        rng = np.random.default_rng(6555)
        nmo, occupied = 25, (3, 2)
        virtual = tuple(nmo-n for n in occupied)
        cs = tuple(np.linalg.qr(rng.normal(size=(nmo, nmo)))[0] for _ in range(2))
        fs = tuple(np.diag(np.linspace(-1.4, .9, nmo)) for _ in range(2))
        factors = rng.normal(scale=.03, size=(6, nmo, nmo))
        factors = .5*(factors+factors.swapaxes(1, 2))
        g = np.einsum('Lpq,Lrs->pqrs', factors, factors)
        space = SpinAmplitudeSpace(occupied, virtual)
        size = sum(a.shape[1] for a in space.indices)
        t1, t2 = space.unpack(jnp.asarray(rng.normal(scale=.002, size=size)))
        host = (*cs, *fs, g, np.asarray(t1), np.asarray(t2))
        input_hash = hashlib.sha256(b''.join(a.tobytes() for a in host)).hexdigest()
        cs, fs, g = tuple(map(jnp.asarray, cs)), tuple(map(jnp.asarray, fs)), jnp.asarray(g)
        start = time.monotonic()
        data, no, nv = from_orbitals(cs, fs, occupied, jnp.asarray(-4.), eri=g,
                                     max_memory=512., vvvv=args.mode)
        jax.block_until_ready(data)
        prepare = time.monotonic()-start
        start = time.monotonic()
        executable = jax.jit(residual).lower(t1, t2, data).compile()
        output = jax.block_until_ready(executable(t1, t2, data))
        first = time.monotonic()-start
        samples = []
        for _ in range(5):
            start = time.monotonic()
            output = jax.block_until_ready(executable(t1, t2, data))
            samples.append(time.monotonic()-start)
        arrays = tuple(map(np.asarray, output))
        assert all(np.isfinite(a).all() for a in arrays)
        np.savez(args.output.with_suffix('.npz'), r1=arrays[0], r2=arrays[1])
        memory = executable.memory_analysis()
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), 'unknown')
        record = dict(mode=args.mode, scope='supplied real UHF frame and one residual, no SCF/root solve',
            occupied=no, virtual=nv, nao=nmo, seed=6555, max_memory_mb=512.,
            dtype=str(t1.dtype), backend=jax.default_backend(), jax=jax.__version__,
            affinity=sorted(os.sched_getaffinity(0)), cpu=cpu, input_sha256=input_hash,
            prepare_seconds=prepare, first_seconds=first, warm_seconds=samples,
            warm_median_seconds=statistics.median(samples),
            integral_storage_bytes=array_storage_bytes(*jax.tree.leaves(data)),
            virtual_storage_bytes=array_storage_bytes(*jax.tree.leaves(data.vvvv)),
            compiled_temp_bytes=memory.temp_size_in_bytes,
            compiled_argument_bytes=memory.argument_size_in_bytes,
            compiled_output_bytes=memory.output_size_in_bytes,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            output_sha256=hashlib.sha256(args.output.with_suffix('.npz').read_bytes()).hexdigest(),
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,
                ('src/gradscf/cc/unrestricted_integrals.py', 'src/gradscf/cc/_spin_virtual.py',
                 'src/gradscf/cc/_spin_intermediates.py', 'src/gradscf/cc/_spin_equations.py',
                 'tests/cc/benchmark_spin_virtual.py'))})
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
