"""Bound each original raw/selected native MP4 geometry response separately."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from run_series_orbital_phases import encode, decode
import test_selected_series_geometry as cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('h2', 'h3'), required=True)
    parser.add_argument('--phase', choices=('raw', 'selected'), required=True)
    args = parser.parse_args()
    stage = Path.cwd()
    files = list((stage/'src').rglob('*.py')) + list((stage/'tests/mp').glob('*.py'))
    files += [stage/'tests/ci/test_correlated_blocks.py']
    hashes = {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    signature = hashlib.sha256(json.dumps(dict(case=args.case, sources=hashes),
                                          sort_keys=True).encode()).hexdigest()
    print(json.dumps(dict(event='source_start', signature=signature, case=args.case)), flush=True)
    started = time.perf_counter()
    source = cases.source.__wrapped__(SimpleNamespace(param=args.case))
    source_seconds = time.perf_counter()-started
    algorithm = 'series' if args.case == 'h2' else 'tensor'
    raw_path = stage/(args.case+'-geometry-raw-bundle.json')
    output = stage/(args.case+'-geometry-'+args.phase+'-bundle.json')
    with pytest.MonkeyPatch.context() as patch:
        if args.phase == 'selected':
            cases._forbid_full_inputs(patch)
        selected, legacy, point, direction = cases._problem(source, algorithm)
        physical = encode((point, direction, source.mo_coeff, source.mo_occ, source.e_tot))
        function = legacy if args.phase == 'raw' else selected
        report = dict(case=args.case, phase=args.phase, signature=signature,
                      source_seconds=source_seconds, comparison_complete=False)
        program = jax.jit(lambda coordinates: cases._bundle(function, coordinates, direction))
        print(json.dumps(dict(event='lower_start', **report)), flush=True)
        started = time.perf_counter()
        lowered = program.lower(point)
        report['lower_seconds'] = time.perf_counter()-started
        print(json.dumps(dict(event='compile_start', **report)), flush=True)
        started = time.perf_counter()
        compiled = lowered.compile()
        report['compile_seconds'] = time.perf_counter()-started
        print(json.dumps(dict(event='execute_start', **report)), flush=True)
        started = time.perf_counter()
        bundle = jax.block_until_ready(compiled(point))
        report['first_execute_seconds'] = time.perf_counter()-started
        started = time.perf_counter()
        jax.block_until_ready(compiled(point))
        report['single_warm_seconds'] = time.perf_counter()-started
        if not all(np.isfinite(x).all() for x in jax.tree.leaves(bundle)):
            raise RuntimeError('Nonfinite native MP4 geometry bundle')
        report.update(physical=physical, bundle=encode(bundle), numerical_valid=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(event='bundle_done', **{k:v for k,v in report.items()
              if k not in ('physical', 'bundle')})), flush=True)
        if args.phase == 'selected':
            raw = json.loads(raw_path.read_text())
            assert raw['signature'] == signature and raw['numerical_valid']
            for a, b in zip(jax.tree.leaves(decode(physical)),
                            jax.tree.leaves(decode(raw['physical']))):
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
            expected = decode(raw['bundle'])
            patch.setattr(cases, '_problem', lambda *a, **kw: (selected, legacy, point, direction))
            patch.setattr(cases, '_evaluate', lambda f, *a: bundle if f is selected else expected)
            print('all_original_comparisons_and_two_finite_differences_start', flush=True)
            cases.test_selected_series_relaxed_geometry_matches_raw(source, algorithm, patch)
            report['comparison_complete'] = True
            output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print('all_original_comparisons_and_two_finite_differences_passed', flush=True)


if __name__ == '__main__':
    main()
