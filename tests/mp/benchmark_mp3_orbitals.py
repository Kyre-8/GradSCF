"""Warm one accepted canonical MP3 point; all numerical work comes from src.

Use run_memory_validation.py for the complete child's exit and peak RSS.
The reference is the saved array/None/with_t2=True accepted MP3 bundle.
This fixed four-orbital point is not a scaling or acceleration comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
from time import perf_counter

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from gradscf.mp import mp3
from gradscf.mp.types import MPResult
import _orbital_frame
from run_mp3_orbital_phases import _restore
from test_mp3_orbitals import _assert_tree, _problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    entry, config, _, _ = _problem('array', None, True)
    assert (reference['case'], reference['phase'], reference['frozen']) == ('array', 'selected', None)
    assert reference['numerical_valid'] and reference['comparison_complete']
    assert reference['configuration'] == repr(config)
    c, focks, energy, _, eri = _orbital_frame.orbital_frame(0., canonical=True)
    inputs = c, focks, energy, eri
    assert all(x.dtype == jnp.float64 for x in jax.tree.leaves(inputs))
    report = dict(complete=False, case='array/None/with_t2=True', nmo=4, nocc=2,
        seed=2426, max_memory_mb=64., config=repr(config), dtype='float64',
        backend=jax.default_backend(), devices=list(map(str, jax.devices())),
        jax=jax.__version__, numpy=np.__version__, python=platform.python_version(),
        affinity=sorted(os.sched_getaffinity(0)), slurm_job=os.environ.get('SLURM_JOB_ID'),
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        reference_signature=reference['signature'],
        input_sha256=[hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
                      for x in jax.tree.leaves(inputs)],
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(_orbital_frame.__file__), Path(mp3.__file__),
            Path(_problem.__code__.co_filename))}, warm_seconds=[])

    def save(phase):
        report['phase'] = phase
        report['self_peak_rss_at_report_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(phase, flush=True)

    def evaluate(coefficients, full_focks, reference_energy, ao_eri):
        return entry(coefficients, full_focks, 2, reference_energy,
            config=config, eri=ao_eri, max_memory=64.)

    save('lower')
    start = perf_counter()
    lowered = jax.jit(evaluate).lower(*inputs)
    report['lower_seconds'] = perf_counter()-start
    save('compile')
    start = perf_counter()
    executable = lowered.compile()
    report['compile_seconds'] = perf_counter()-start
    memory = executable.memory_analysis()
    report['xla_memory_bytes'] = {name: getattr(memory, name) for name in (
        'argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes',
        'alias_size_in_bytes', 'host_temp_size_in_bytes')}
    save('first')
    start = perf_counter()
    result = jax.block_until_ready(executable(*inputs))
    report['first_seconds'] = perf_counter()-start
    save('first_complete')
    for _ in range(5):
        start = perf_counter()
        result = jax.block_until_ready(executable(*inputs))
        report['warm_seconds'].append(perf_counter()-start)
        save('warm')
    expected = MPResult(*(_restore(reference['primal'][name]) for name in MPResult._fields))
    assert bool(result.valid)
    _assert_tree(result, expected)
    report.update(complete=True, accepted_output_matches=True,
        warm_median_seconds=statistics.median(report['warm_seconds']),
        total_energy=float(result.total_energy), e2=float(result.e2), e3=float(result.e3),
        t2_shape=list(result.t2.shape), comparison_atol=3e-11, comparison_rtol=3e-9)
    save('complete')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
