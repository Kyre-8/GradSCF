"""Time the unchanged H6 cached-response test, then repeat it with live caches."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pytest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    test = Path('tests/mp/test_series.py')
    assert hashlib.sha256(test.read_bytes()).hexdigest() == '0cfdaf433020494f4f19746cf9ba8141c3d20513704eff2ce3ef3fd684b65d1d'
    node = str(test)+'::test_cached_nodes_retain_integral_and_gap_response[system2]'
    started = time.perf_counter()
    record = dict(case=node, test_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),
        process_state='fresh', affinity=sorted(os.sched_getaffinity(0)), phases=[],
        python=sys.version, packages={key: importlib.metadata.version(key)
            for key in ('jax', 'jaxlib', 'numpy', 'pyscf', 'pytest')},
        environment={key: os.environ.get(key) for key in ('JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES',
            'JAX_ENABLE_X64', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'JAX_COMPILATION_CACHE_DIR', 'JAX_ENABLE_COMPILATION_CACHE', 'JAX_LOG_COMPILES',
            'JAX_EXPLAIN_CACHE_MISSES')},
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,
            ('src/gradscf/mp/series.py', 'src/gradscf/solvers/operators.py',
             'src/gradscf/solvers/linear/implicit.py'))},
        boundary='Cold/warm are whole unchanged test-body invocations in one process. The repeat reuses module/JAX caches but creates the test original local jit closures again; it is not a pure compiled-kernel warm time.')

    def save():
        record['elapsed_s'] = time.perf_counter()-started
        record['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.write_text(json.dumps(record, indent=2))

    def begin(name):
        phase = dict(name=name, start_s=time.perf_counter()-started,
                     peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     current_rss_kib=next(int(line.split()[1]) for line in
                         Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))
        record['phases'].append(phase)
        save()
        return phase

    def finish(phase, passed):
        phase.update(seconds=time.perf_counter()-started-phase['start_s'], passed=passed,
                     peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save()

    class Repeat:
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self, item):
            _, h, g, no = item.funcargs['system']
            record['input_sha256'] = {name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                                     for name, value in (('h', h), ('g', g))}
            record['nocc'] = no
            phase = begin('cold_body')
            outcome = yield
            finish(phase, outcome.excinfo is None)
            if outcome.excinfo is not None:
                return
            phase = begin('warm_repeat_body')
            try:
                item.runtest()  # Same original function, fixture and assertions.
            except BaseException:
                finish(phase, False)
                raise
            finish(phase, True)

    save()
    code = pytest.main(['-vv', node, '--durations=0', '--log-cli-level=WARNING',
                        '--junitxml='+str(args.output.with_suffix('.xml'))], plugins=[Repeat()])
    record['exit_code'] = int(code)
    save()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
