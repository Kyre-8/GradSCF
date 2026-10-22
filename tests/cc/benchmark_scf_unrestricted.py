"""Matched LiH+ CCSD/Lambda/RDM1/(T) consumer pipeline, excluding timed SCF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import threading
import time

import jax
import numpy as np

from gradscf import cc, gto, scf
from gradscf.ao2mo import transform_unrestricted_integrals
from gradscf.ao2mo.scf import prepare_scf_source
from gradscf.cc.unrestricted_integrals import from_orbitals
from gradscf.cc.properties import _density_from_lambda, _density_ucc_from_integrals


def _rss():
    return next(int(line.split()[1])*1024 for line in
                Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('raw', 'selected'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    mol = gto.M(atom='Li 0 0 0; H 0 0 3.1', basis='sto-3g',
                charge=1, spin=1, unit='Bohr', cart=True)
    mf = scf.UHF(mol, integral_backend='native', execution_device='cpu',
        max_memory=64., conv_tol=1e-12, conv_tol_density=1e-11,
        conv_tol_grad=1e-10, max_cycle=250).run()
    assert mf.converged
    data = prepare_scf_source(mf)
    point = (data.coefficients, data.focks, data.reference_energy,
             data.hcore, data.ao_kwargs)
    cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-10)
    options = dict(residual_tol=cfg.residual_tol, denominator_tol=cfg.denominator_tol)

    def pipeline(c, f, e, h, source):
        if args.mode == 'raw':
            hs, gs = transform_unrestricted_integrals(h, c, **source, max_memory=64.)
            state = cc.run_ucc(hs, gs, nocc=(2, 1),
                nuclear_repulsion=mf.reference.nuclear_repulsion, config=cfg)
            left = cc.solve_lambda(hs, gs, state, nocc=(2, 1), config=cfg)
            dm = _density_from_lambda(hs, gs, state, left, nocc=(2, 1), frozen=None, config=cfg)
            triples = cc.evaluate_triples(hs, gs, state, nocc=(2, 1), **options)
        else:
            ints, occupied, virtual = from_orbitals(c, f, (2, 1), e,
                                                    **source, max_memory=64.)
            state = cc.run_ucc_from_integrals(ints, occupied, virtual, config=cfg)
            left = cc.solve_lambda_from_integrals(ints, state, config=cfg)
            dm = _density_ucc_from_integrals(ints, state, left, nmo=6, nocc=(2, 1), config=cfg)
            triples = cc.evaluate_triples_from_integrals(ints, state, **options)
        return (state.total_energy, state.t1, state.t2, left.l1, left.l2, dm,
                triples.energy, state.residual_norm, left.residual_norm,
                state.converged, left.converged, triples.valid)

    started = time.perf_counter()
    executable = jax.jit(pipeline).lower(*point).compile()
    compile_seconds = time.perf_counter()-started
    started = time.perf_counter()
    result = jax.block_until_ready(executable(*point))
    first_seconds = time.perf_counter()-started
    assert all(bool(value) for value in result[-3:])
    assert abs(float(result[6])) > 1e-12
    after_compile = _rss()
    stopped = threading.Event()
    sampled = [after_compile]

    def sample():
        while not stopped.wait(.002):
            sampled.append(_rss())

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    elapsed = []
    try:
        for _ in range(5):
            started = time.perf_counter()
            result = jax.block_until_ready(executable(*point))
            elapsed.append(time.perf_counter()-started)
            sampled.append(_rss())
    finally:
        stopped.set()
        thread.join()
    digest = hashlib.sha256()
    for value in jax.tree.leaves(point):
        array = np.asarray(value)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    memory = executable.memory_analysis()
    root = Path(__file__).resolve().parents[2]
    names = ['src/gradscf/cc/'+name+'.py' for name in ('api', 'uccsd',
        'unrestricted_integrals', 'lambda_equations', 'properties', 'triples', '_spin_triples')]
    names.append('tests/cc/benchmark_scf_unrestricted.py')
    record = dict(mode=args.mode, molecule='LiH+; Li(0,0,0), H(0,0,3.1) Bohr',
        basis='STO-3G Cartesian', charge=1, spin=1, nocc=(2, 1), dtype='float64',
        backend=jax.default_backend(), jax=jax.__version__, input_sha256=digest.hexdigest(),
        affinity=sorted(os.sched_getaffinity(0)),
        cpu=next(line for line in Path('/proc/cpuinfo').read_text().splitlines()
                 if line.startswith('model name')),
        compile_seconds=compile_seconds, first_seconds=first_seconds,
        warm_seconds=elapsed, warm_median_seconds=statistics.median(elapsed),
        after_compile_rss_bytes=after_compile, warm_peak_rss_bytes=max(sampled),
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        xla_temporary_bytes=memory.temp_size_in_bytes, xla_output_bytes=memory.output_size_in_bytes,
        total_energy=float(result[0]), triples_energy=float(result[6]),
        cc_residual=float(result[7]), lambda_residual=float(result[8]),
        density_traces=[float(np.trace(d)) for d in result[5]],
        output_leaves=[np.asarray(value).tolist() for value in jax.tree.leaves(result)],
        sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
        scope='Current raw full-MO APIs vs selected snapshot reuse; five warm pipeline calls after compilation. Timed work excludes SCF and outer geometry AD. Dense spin vvvv/amplitudes remain; not a total-method hard cap.')
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({key: value for key, value in record.items() if key != 'output_leaves'}, indent=2))


if __name__ == '__main__':
    main()
