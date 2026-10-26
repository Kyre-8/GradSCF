"""Requested CC two-body densities retain the original real spin chart."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.cc.types import CCResult, LambdaResult


def _states(unrestricted, frozen):
    n = 3 if unrestricted else 4
    counts = (2,1) if unrestricted else (2,2)
    frozen_sets = (((),()) if frozen is None else frozen) if unrestricted else (((),()) if frozen is None else (tuple(frozen),)*2)
    no = tuple(sum(i not in fr for i in range(count)) for count,fr in zip(counts,frozen_sets))
    nv = tuple(sum(i not in fr for i in range(count,n)) for count,fr in zip(counts,frozen_sets))
    rng = np.random.default_rng(713)
    singles = tuple((a,b) for a,b in zip(no,nv))
    doubles = ((no[0],no[0],nv[0],nv[0]),(no[0],no[1],nv[0],nv[1]),(no[1],no[1],nv[1],nv[1]))
    values = [tuple(jnp.asarray(rng.normal(scale=.03,size=shape)) for shape in group)
              for group in (singles,doubles,singles,doubles)]
    t1,t2,l1,l2 = values
    if not unrestricted:
        t1,t2,l1,l2 = t1[0],t2[1],l1[0],l2[1]
    zero = jnp.asarray(0.)
    result = CCResult(zero,zero,zero,t1,t2,zero,zero,jnp.asarray(0),jnp.asarray(True),jnp.asarray(1.),jnp.asarray(2))
    left = LambdaResult(l1,l2,jnp.zeros(1),zero,jnp.asarray(True))
    return n,counts if unrestricted else 2,result,left


@pytest.mark.parametrize('unrestricted,frozen,spin',[
    (False,None,None),(False,[0,3],None),
    (True,None,'ab'),(True,([0],[]),'bb')])
def test_requested_rdm2_does_not_build_a_complete_spin_density(unrestricted,frozen,spin,monkeypatch):
    module = importlib.import_module('gradscf.cc.properties')
    n,nocc,result,left = _states(unrestricted,frozen)
    indices = ((n-1,0),(1,0),(0,n-1),(n-1,0))
    oracle = getattr(module,'_rdm2_dense_from_states',module._rdm2_from_states)
    full = oracle(result,left,nmo=n,nocc=nocc,frozen=frozen,dtype=jnp.float64)
    expected = (full[{'aa':0,'ab':1,'bb':2}[spin]] if unrestricted else full)[jnp.ix_(*map(jnp.asarray,indices))]

    def forbidden(*args,**kwargs):
        pytest.fail('Requested CC density materialized a complete active spin 2-RDM')

    monkeypatch.setattr(module,'active_density_parts',forbidden)
    actual = module._rdm2_from_states(result,left,nmo=n,nocc=nocc,frozen=frozen,
        dtype=jnp.float64,indices=indices,spin=spin,max_memory=32.,reserved_bytes=256)
    assert actual.shape == tuple(map(len,indices))
    np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=0)


def test_full_spatial_output_is_rejected_before_spin_amplitude_expansion(monkeypatch):
    module = importlib.import_module('gradscf.cc.properties')
    _,_,result,left = _states(False,None)
    singles = jax.ShapeDtypeStruct((1,999),jnp.float64)
    doubles = jax.ShapeDtypeStruct((1,1,999,999),jnp.float64)
    result = result._replace(t1=singles,t2=doubles)
    left = left._replace(l1=singles,l2=doubles)

    def forbidden(*args,**kwargs):
        pytest.fail('Impossible CC output budget reached spin amplitude expansion')

    monkeypatch.setattr(module.SpinAmplitudeSpace,'from_blocks',forbidden)
    with pytest.raises(ValueError,match='memory|budget|output'):
        module._rdm2_from_states(result,left,nmo=1000,nocc=1,dtype=jnp.float64,max_memory=.01)


@pytest.mark.parametrize('unrestricted,spin',[(False,None),(True,None),(True,'ba')])
def test_four_axis_clamped_boxes_and_direct_reduction(unrestricted,spin):
    module = importlib.import_module('gradscf.cc.properties')
    n,nocc,result,left = _states(unrestricted,None)
    # Unequal axes, repeated entries and a nondivisible tail in each large axis.
    indices = ((n-1,0,1),(1,0),(0,n-1,0),(n-1,))
    full = module._rdm2_dense_from_states(result,left,nmo=n,nocc=nocc,dtype=jnp.float64)
    channels = full if unrestricted and spin is None else ((full[1].transpose(2,3,0,1),)
                                                          if spin == 'ba' else (full,))
    expected = tuple(value[jnp.ix_(*map(jnp.asarray,indices))] for value in channels)
    _,spins = module._density_layout(n,nocc,indices,spin)
    metadata = dict(nmo=n,frame=module._density_frame(n,nocc,None),dtype=jnp.dtype('float64'),
                    indices=indices,spins=spins,widths=(2,2,2,1))
    actual = module._density_execute(result,left,(),contract=False,**metadata)
    for value,reference in zip(actual,expected):
        np.testing.assert_allclose(value,reference,atol=2e-13,rtol=0)
    # Full observable reduction uses the same boxes and masks, not a full RDM.
    operators = tuple(jnp.arange(n**4,dtype=jnp.float64).reshape((n,)*4)/(n**4)+.2
                      for _ in range(3 if unrestricted else 1))
    raw = full if unrestricted else (full,)
    weights = (.5,1.,.5) if unrestricted else (.5,)
    target = sum(w*jnp.sum(g*d) for w,g,d in zip(weights,operators,raw))
    _,spins = module._density_layout(n,nocc,None,None)
    actual = module._density_execute(result,left,operators,nmo=n,
        frame=module._density_frame(n,nocc,None),dtype=jnp.dtype('float64'),
        indices=(tuple(range(n)),)*4,spins=spins,widths=(2,)*4,contract=True)
    np.testing.assert_allclose(actual,target,atol=2e-13,rtol=0)


def test_requested_empty_axis_and_mixed_left_dtype():
    module = importlib.import_module('gradscf.cc.properties')
    n,nocc,result,left = _states(False,None)
    result = result._replace(t1=result.t1.astype(jnp.float32),t2=result.t2.astype(jnp.float32))
    indices = ((3,0),(1,),(),(2,0))
    actual = module._rdm2_from_states(result,left,nmo=n,nocc=nocc,dtype=jnp.float32,
                                      indices=indices,max_memory=32.)
    assert actual.shape == (2,1,0,2) and actual.dtype == jnp.float32
    expected = module._rdm2_dense_from_states(result,left,nmo=n,nocc=nocc,dtype=jnp.float32)
    actual = module._rdm2_from_states(result,left,nmo=n,nocc=nocc,dtype=jnp.float32,max_memory=32.)
    np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)
