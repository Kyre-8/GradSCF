"""Measure public CC candidate planning separately from AO->MO preparation."""
import argparse
import json
import time
from unittest.mock import patch

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf import cc
from gradscf.ao2mo import VirtualERI
from gradscf.cc.integrals import from_orbitals
from gradscf.scf.facade import RKS
from test_native_cc_auto import frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-memory', type=float, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = frame.__wrapped__()
    model = cc.CCSD(RKS(None, xc='hf', max_memory=args.max_memory))
    model.reference, model._nmo = source, source.coefficients.shape[1]
    paid = jax.tree.leaves((source.coefficients, source.focks,
                           source.reference_energy, source.ao_kwargs))
    reservation = model._held_bytes(*paid)
    selected = lambda: model._restricted_integrals(source, model._config(), None, None, reservation)
    samples = {'planned': [], 'preparation_only': []}
    original = jax.eval_shape
    depth = 0
    metadata_seconds = []

    def timed_shape(*values, **options):
        nonlocal depth
        outer = depth == 0
        depth += 1
        start = time.perf_counter()
        try:
            return original(*values, **options)
        finally:
            depth -= 1
            if outer:
                metadata_seconds.append(time.perf_counter()-start)

    def measure(function):
        metadata_seconds.clear()
        start = time.perf_counter()
        with patch.object(jax, 'eval_shape', timed_shape):
            value = jax.block_until_ready(function())
        return value, {'total_seconds': time.perf_counter()-start,
                       'metadata_seconds': sum(metadata_seconds),
                       'metadata_calls': len(metadata_seconds)}

    actual, cold = measure(selected)
    mode = 'direct' if isinstance(actual.vvvv, VirtualERI) else 'dense'
    baseline = lambda: from_orbitals(source.coefficients, source.focks[0], source.nocc,
        source.reference_energy, max_memory=args.max_memory, reserved_bytes=reservation,
        vvvv=mode, model='ccsd', **source.ao_kwargs)
    expected, baseline_cold = measure(baseline)
    error = 0.
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.isfinite(np.asarray(x)).all() and np.isfinite(np.asarray(y)).all()
        error = max(error, float(np.max(np.abs(np.asarray(x)-np.asarray(y)), initial=0.)))
    assert error < 1e-12
    del actual, expected
    for iteration in range(args.repeats):
        order = ('planned', 'preparation_only') if iteration % 2 == 0 else ('preparation_only', 'planned')
        for name in order:
            value, timing = measure(selected if name == 'planned' else baseline)
            samples[name].append(timing)
            del value
    report = {'max_memory_mb': args.max_memory, 'selected_mode': mode,
              'shape': {'nao': source.coefficients.shape[0], 'nocc': source.nocc, 'nvir': 32},
              'scope': 'Fixed algebraic frame, candidate planning and integral preparation only; no SCF/CC solve.',
              'max_leaf_error': error, 'cold': {'planned': cold, 'preparation_only': baseline_cold},
              'warm': samples}
    with open(args.output, 'w') as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
