"""Eager EOM retains known SCF, reference and Lambda owners at its boundary."""
import importlib
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from gradscf import cc
from gradscf._memory import tree_storage_bytes


@pytest.mark.parametrize('prepared', [False, True])
def test_eom_facade_forwards_held_source_and_lambda_arrays(prepared, monkeypatch):
    module = importlib.import_module('gradscf.cc.eom.api')
    # Declared owner metadata, not an artificial numerical CC reference.
    reference = SimpleNamespace(h1=jnp.ones((2, 2)), eri=jnp.ones((2,)*4),
                                metadata=jnp.ones(17), nocc=1)
    ints = SimpleNamespace(block=jnp.ones(9)) if prepared else None
    state = SimpleNamespace(t2=jnp.ones(5))
    ground = SimpleNamespace(max_memory=8., reference=reference, _integrals=ints,
        result=state, lambda_result=SimpleNamespace(left=jnp.ones(19)),
        mf=SimpleNamespace(held=jnp.ones(13)), frozen=None,
        _config=lambda: cc.CCConfig())
    method = cc.EOMEE(ground)
    method._source = lambda: reference
    callee_owned = (ints, state) if prepared else (reference.h1, reference.eri, state)
    expected = tree_storage_bytes(ground.mf, reference, ground.lambda_result,
                                  exclude=callee_owned)

    class Captured(Exception):
        pass

    def checked(*args, **kwargs):
        assert kwargs['max_memory'] == 8.
        assert kwargs.get('reserved_bytes') == expected
        raise Captured

    monkeypatch.setattr(module, 'run_eom_from_integrals' if prepared else 'run_eom', checked)
    with pytest.raises(Captured):
        method.kernel()
