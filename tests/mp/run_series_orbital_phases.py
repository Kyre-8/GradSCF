"""Run the unchanged raw/selected test bundles in separate bounded processes.

This is diagnostic test orchestration, not an MP implementation or benchmark.
The selected phase runs every original assertion against the saved raw bundle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np
import pytest

from gradscf import ao2mo
from gradscf.mp.types import MPResult
import test_series_orbitals as cases


def encode(value):
    if isinstance(value, MPResult):
        return {'result': [encode(x) for x in value]}
    if isinstance(value, tuple):
        return {'tuple': [encode(x) for x in value]}
    if value is None:
        return None
    array = np.asarray(value)
    return dict(dtype=str(array.dtype), shape=list(array.shape), data=array.tolist())


def decode(value):
    if value is None:
        return None
    if 'result' in value:
        return MPResult(*(decode(x) for x in value['result']))
    if 'tuple' in value:
        return tuple(decode(x) for x in value['tuple'])
    return np.asarray(value['data'], dtype=value['dtype']).reshape(value['shape'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('restricted-df', 'unrestricted-s4'), required=True)
    parser.add_argument('--phase', choices=('raw', 'selected'), required=True)
    args = parser.parse_args()
    unrestricted = args.case == 'unrestricted-s4'
    algorithm, kind = ('tensor', 's4') if unrestricted else ('series', 'df')
    stage = Path.cwd()
    sources = list((stage / 'src').rglob('*.py')) + [Path(cases.__file__),
        Path(cases.__file__).with_name('_orbital_frame.py'), Path(__file__)]
    hashes = {str(p.resolve().relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sources}
    signature = hashlib.sha256(json.dumps(dict(case=args.case, sources=hashes),
                                          sort_keys=True).encode()).hexdigest()
    path = stage / (args.case + '-' + args.phase + '-bundle.json')
    raw_path = stage / (args.case + '-raw-bundle.json')
    started = time.perf_counter()
    print(json.dumps(dict(phase=args.phase + '_bundle_start', signature=signature)), flush=True)
    with pytest.MonkeyPatch.context() as patch:
        if args.phase == 'selected':
            def forbidden(*a, **kw):
                pytest.fail('Selected MP series rebuilt a complete MO reference')
            for name in ('transform_integrals', 'transform_unrestricted_integrals', 'spin_orbital_integrals'):
                patch.setattr(ao2mo, name, forbidden)
        original = cases._problem
        _, nocc, config, _, evaluate = original(unrestricted, algorithm, kind)
        bundle = jax.block_until_ready(evaluate(args.phase == 'selected'))
        finite = all(bool(np.isfinite(x).all()) for x in jax.tree.leaves(bundle[1]))
        report = dict(case=args.case, phase=args.phase, signature=signature,
            configuration=repr(config), nocc=nocc, point=0.,
            seconds=time.perf_counter() - started, numerical_valid=bool(bundle[0].valid) and finite,
            comparison_complete=False)
        print(json.dumps(dict(event=args.phase + '_bundle_done', **report)), flush=True)
        if not report['numerical_valid']:
            path.write_text(json.dumps(report, indent=2) + '\n')
            raise RuntimeError('The original test bundle has nonfinite energy response or invalid state')
        report['bundle'] = encode(bundle)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        if args.phase == 'selected' and raw_path.exists():
            raw = json.loads(raw_path.read_text())
            assert raw['signature'] == signature and raw['phase'] == 'raw' and raw['numerical_valid']
            expected = decode(raw['bundle'])
            def cached_problem(*a, **kw):
                entry, nocc, config, result, _ = original(*a, **kw)
                def compact(*aa, **kk):
                    print('compact_forward_start', flush=True)
                    value = entry(*aa, **kk)
                    jax.block_until_ready(value)
                    print('compact_forward_done', flush=True)
                    return value
                return compact, nocc, config, result, lambda selected: bundle if selected else expected
            patch.setattr(cases, '_problem', cached_problem)
            cases.test_selected_series_joint_frame_fock_energy_source_response(
                unrestricted, algorithm, kind, patch)
            report['comparison_complete'] = True
            report['seconds'] = time.perf_counter() - started
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print('all_original_comparisons_and_output_flags_passed', flush=True)


if __name__ == '__main__':
    main()
