"""Run each original canonical MP3 response comparison in bounded phases."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np
import pytest

from gradscf.mp import mp3
from gradscf.mp.types import MPResult
import test_mp3_orbitals as cases


def _array(value):
    if value is None:
        return None
    value = np.asarray(value)
    return dict(dtype=str(value.dtype), shape=list(value.shape), data=value.tolist())


def _restore(value):
    return None if value is None else np.asarray(value['data'], dtype=value['dtype']).reshape(value['shape'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('array', 's4', 'df'), required=True)
    parser.add_argument('--phase', choices=('raw', 'selected'), required=True)
    args = parser.parse_args()
    frozen, with_t2 = {'array': (None, True), 's4': (1, False), 'df': ([0, 3], True)}[args.case]
    stage = Path.cwd()
    files = list((stage/'src').rglob('*.py')) + list((stage/'tests/mp').glob('*.py'))
    hashes = {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    signature = hashlib.sha256(json.dumps(dict(case=args.case, sources=hashes),
                                          sort_keys=True).encode()).hexdigest()
    path = stage/(args.case+'-mp3-'+args.phase+'-bundle.json')
    raw_path = stage/(args.case+'-mp3-raw-bundle.json')
    started = time.perf_counter()
    print(json.dumps(dict(event='bundle_start', case=args.case, phase=args.phase,
                         signature=signature)), flush=True)
    with pytest.MonkeyPatch.context() as patch:
        if args.phase == 'selected':
            def forbidden(*a, **kw):
                pytest.fail('Selected MP3 re-entered a full-MO/raw-integral preparer')
            patch.setattr(mp3, 'prepare_integrals', forbidden)
            patch.setattr(mp3, 'prepare_mp_integrals', forbidden)
        entry, config, result, evaluate = cases._problem(args.case, frozen, with_t2)
        selected = args.phase == 'selected'
        primal = jax.block_until_ready(result(0., selected))
        primal_seconds = time.perf_counter()-started
        print(json.dumps(dict(event='primal_done', seconds=primal_seconds)), flush=True)
        response = jax.block_until_ready(evaluate(selected))
        assert bool(primal.valid)
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(response))
        report = dict(case=args.case, phase=args.phase, signature=signature,
            configuration=repr(config), frozen=frozen, primal_seconds=primal_seconds,
            seconds=time.perf_counter()-started, numerical_valid=True, comparison_complete=False,
            primal={k:_array(v) for k,v in primal._asdict().items()},
            response=[_array(v) for v in jax.tree.leaves(response)])
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(event='bundle_done', **{k:v for k,v in report.items()
              if k not in ('primal', 'response')})), flush=True)
        if selected:
            raw = json.loads(raw_path.read_text())
            assert raw['signature'] == signature and raw['numerical_valid']
            expected = MPResult(*(_restore(raw['primal'][k]) for k in MPResult._fields))
            leaves = [_restore(v) for v in raw['response']]
            expected_response = ((leaves[0], leaves[1]), (leaves[2], leaves[3]))
            patch.setattr(cases, '_problem', lambda *a: (entry, config,
                lambda x, chosen: primal if chosen else expected,
                lambda chosen: response if chosen else expected_response))
            cases.test_selected_mp3_keeps_raw_canonical_energy_amplitudes_and_joint_response(
                args.case, frozen, with_t2, patch)
            report['comparison_complete'] = True
            report['seconds'] = time.perf_counter()-started
            path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print('all_original_mp3_value_amplitude_response_assertions_passed', flush=True)


if __name__ == '__main__':
    main()
