"""The CC property facade shares one caller ledger through Lambda and RDMs."""
import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc
from test_rdm2_blocks import _states


def _facade():
    n,nocc,result,left = _states(False,None)
    ref = cc.CCReference(jnp.eye(n),jnp.zeros((n,)*4),nocc)
    method = cc.CCSD(ref,max_memory=32.)
    method.reference,method.result,method.lambda_result = ref,result,left
    method._ready = lambda:ref
    return method,ref,left


def test_facade_rdm2_forwards_selection_and_all_retained_owners(monkeypatch):
    module = importlib.import_module('gradscf.cc.api')
    method,ref,left = _facade()
    selection = ((3,0),(1,),(0,2),(2,0))
    method.solve_lambda = lambda:setattr(method,'converged_lambda',True)
    captured = []

    def checked(h,g,result,adjoint,**kwargs):
        assert kwargs['indices'] == selection and kwargs['spin'] == 'ab'
        assert kwargs['max_memory'] == 32.
        assert kwargs['reserved_bytes'] == method._held_bytes(h,g,result,adjoint)
        captured.append(kwargs)
        return jnp.zeros(tuple(map(len,selection)))

    monkeypatch.setattr(module,'_rdm2_from_lambda',checked)
    method.make_rdm2(indices=selection,spin='ab')
    assert len(captured) == 1


def test_observable_reserves_its_inputs_and_solves_lambda_once(monkeypatch):
    module = importlib.import_module('gradscf.cc.api')
    method,ref,left = _facade()
    one = jnp.arange(16,dtype=jnp.float64).reshape(4,4)/100
    two = jnp.ones((4,)*4)*.03
    calls = []

    def solved(*,_extra=()):
        assert _extra[0] is one and _extra[1] is two
        calls.append('lambda')
        method.converged_lambda = True
        return left.l1,left.l2

    def density(reference,*,extra=()):
        assert reference is ref and extra[0] is one and extra[1] is two
        calls.append('one')
        return jnp.eye(4)

    def contracted(operator,result,adjoint,**kwargs):
        assert operator is two and adjoint is left
        minimum = method._held_bytes(result,adjoint,operator,extra=(one,two))
        assert kwargs['reserved_bytes'] >= minimum
        assert kwargs['max_memory'] == 32.
        calls.append('two')
        return jnp.asarray(.7)

    method.solve_lambda = solved
    monkeypatch.setattr(method,'_rdm1_with_left',density,raising=False)
    monkeypatch.setattr(module,'_contract_rdm2_from_states',contracted,raising=False)
    function = getattr(method,'expectation_value',None)
    assert callable(function),'Missing direct CC observable reduction'
    np.testing.assert_allclose(function(one,two),np.trace(one)+.7,atol=1e-14)
    assert calls == ['lambda','one','two']


def test_facade_rejects_impossible_rdm_output_before_lambda(monkeypatch):
    method,_,left = _facade()
    method.max_memory = .01
    method._nmo = 1000
    method.reference = ref = SimpleNamespace(nocc=1,
        coefficients=jax.ShapeDtypeStruct((1000,1000),jnp.float64),
        hcore=jax.ShapeDtypeStruct((1000,1000),jnp.float64))
    method._integrals = SimpleNamespace(fock=jnp.ones((2,2)))
    method._ready = lambda:ref

    def forbidden(*args,**kwargs):
        pytest.fail('Impossible CC output reached the Lambda solve')

    method.solve_lambda = forbidden
    with pytest.raises(ValueError,match='memory|budget|output'):
        method.make_rdm2()


def test_scalar_budget_uses_observable_promoted_output_dtype(monkeypatch):
    module = importlib.import_module('gradscf.cc.properties')
    from gradscf._memory import tree_storage_bytes
    n,nocc,result,left = _states(False,None)
    result = result._replace(t1=result.t1.astype(jnp.float32),t2=result.t2.astype(jnp.float32))
    left = left._replace(l1=left.l1.astype(jnp.float32),l2=left.l2.astype(jnp.float32))
    operator = jnp.ones((n,)*4,jnp.float64)
    expected = 256+tree_storage_bytes(result,left,operator)+4*(4*n)+6*8

    def checked(memory,*,resident_bytes,device=None):
        assert resident_bytes == expected
        raise ValueError('checked promoted scalar memory budget')

    monkeypatch.setattr(module,'workspace_bytes',checked)
    with pytest.raises(ValueError,match='checked promoted'):
        module._contract_rdm2_from_states(operator,result,left,nmo=n,nocc=nocc,
            dtype=jnp.float32,max_memory=32.,reserved_bytes=256)
