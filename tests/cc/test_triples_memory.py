"""Triples preflight follows the precision of the actual moment operands."""
import importlib

import jax.numpy as jnp
import pytest

from gradscf import cc
from gradscf.cc.integrals import prepare_integrals
from gradscf.cc.types import CCResult
from gradscf.cc.uccsd import SpinCCIntegrals


@pytest.mark.parametrize('unrestricted', [False, True])
@pytest.mark.parametrize('moment', ['ovvv', 'ooov', 'oovv'])
def test_mixed_precision_triples_rejects_before_physical_work(unrestricted, moment, monkeypatch):
    zeros = lambda shape: jnp.zeros(shape, dtype=jnp.float32)
    if unrestricted:
        no, nv = 8, 4
        ints = SpinCCIntegrals(*map(zeros, (
            (12, 12), (no,)*4, (no, no, no, nv), (no, no, nv, nv),
            (no, nv, no, nv), (no, nv, nv, nv), (nv,)*4, (),
        )))
        t1 = (zeros((4, 2)), zeros((4, 2)))
        t2 = tuple(zeros((4, 4, 2, 2)) for _ in range(3))
        field = moment
    else:
        ints = prepare_integrals(zeros((6, 6)), zeros((6,)*4), nocc=4)
        t1, t2 = zeros((4, 2)), zeros((4, 4, 2, 2))
        field = {'ovvv': 'ovvv', 'ooov': 'ovoo', 'oovv': 'ovov'}[moment]
    ints = ints._replace(**{field: getattr(ints, field).astype(jnp.float64)})
    state = CCResult(zeros(()), zeros(()), zeros(()), t1, t2, zeros(()),
                     zeros(()), 0, True, jnp.asarray(1.), 2)
    owner = importlib.import_module('gradscf.cc._spin_triples' if unrestricted
                                   else 'gradscf.cc.triples')

    class PhysicalWorkReached(Exception):
        pass

    def reached(*args, **kwargs):
        raise PhysicalWorkReached

    monkeypatch.setattr(owner, 'residual', reached)
    if unrestricted:
        monkeypatch.setattr(owner.SpinAmplitudeSpace, 'from_blocks', reached)
    # This finite budget admits the old float32 panel estimate, but cannot
    # hold the same existing panel at its actual float64 contraction precision.
    with pytest.raises(ValueError, match='memory|budget|resident'):
        cc.evaluate_triples_from_integrals(ints, state, algorithm='streamed', max_memory=.65)
    with pytest.raises(PhysicalWorkReached):
        cc.evaluate_triples_from_integrals(ints, state, algorithm='streamed', max_memory=32.)
