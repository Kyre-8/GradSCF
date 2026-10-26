from importlib import import_module
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.model.nnao import prepare_direct_basis as _prepare_direct_basis
from functools import partial
prepare_direct_basis=partial(_prepare_direct_basis,core_primitives=3)
from gradscf import integrals


def test_4s4p2d_expands_primitives_without_changing_ao_space_size():
    old=prepare_direct_basis('C 0 0 0; H 0 0 1.09',basis_family='szp3_direct')
    new=prepare_direct_basis('C 0 0 0; H 0 0 1.09')
    assert new.topology.primitive_counts==(3,4,4,2,3,1)
    assert new.topology.nao==old.topology.nao==14
    assert new.nelectron==old.nelectron==7
    assert all(e is None for e in new.ecps)
    assert new.reference_outputs().shape==(2,3,4)
    for i,role in enumerate(new.roles):
        a=np.asarray(new.parameters.exponents[i]);b=np.asarray(old.parameters.exponents[i])
        np.testing.assert_array_equal(a[:len(b)],b)
        if len(a)>len(b):assert new.parameters.coefficients[i][-1,0]==0
        if role=='core':np.testing.assert_array_equal(new.parameters.coefficients[i],old.parameters.coefficients[i])
    p=new.bind(new.reference_outputs());q=old.bind(old.reference_outputs())
    for op in ('overlap','kinetic','nuclear','eri'):
        np.testing.assert_allclose(integrals.make_plan(new.topology).evaluate(op,p),
                                   integrals.make_plan(old.topology).evaluate(op,q),atol=2e-10,rtol=1e-11)


@pytest.mark.parametrize('symbol',['C','N','O','F','Cl','Br','I'])
def test_new_d_channel_is_trainable_and_core_stays_fixed(symbol):
    b=prepare_direct_basis(f'{symbol} 0 0 0')
    x=b.reference_outputs();p=b.bind(x)
    changed=jax.jit(b.bind)(x.at[0,2,1].set(.2))
    for i,(role,slot) in enumerate(zip(b.roles,b.slots)):
        equal=np.allclose(p.coefficients[i],changed.coefficients[i])
        assert equal == (slot!=2)
        if role=='core':assert len(p.exponents[i])==3
    slots={s:len(a) for s,a in zip(b.slots,p.exponents) if s>=0}
    assert slots=={0:4,1:4,2:2}


def test_expanded_methane_energy_gradient_for_added_primitives():
    from pathlib import Path
    import runpy
    cls=vars(import_module("gradscf.model.nnao.training.objective"))['BasisRHF']
    ex=cls(basis_family='szp442_direct',core_primitives=3)
    assert ex.layout.topology.nao==26 and ex.primitive_topology.nao==53 and ex.nelectron==10
    x=ex.layout.reference_outputs();energy,g,info=ex.evaluate(x)
    direction=jnp.zeros_like(x).at[0,0,3].set(.2).at[0,1,3].set(-.3).at[0,2,1].set(.4)
    h=1e-4;e=lambda a:ex.evaluate(x+a*direction)[0]
    fd=(8*(e(h)-e(-h))-e(2*h)+e(-2*h))/(12*h)
    np.testing.assert_allclose(jnp.sum(g*direction),fd,atol=2e-6,rtol=2e-5)
    assert float(jnp.linalg.norm(g[0,:,1:]))>1e-5


def test_real_mace_expanded_head_and_no_hidden_reference():
    pytest.importorskip('cuequivariance');pytest.importorskip('mace_jax')
    from flax import nnx
    from gradscf.model.nnao import MACEBasisModel,build_graph
    b=prepare_direct_basis('C 0 0 0; H 0 0 1.09')
    model=MACEBasisModel(elements=(1,6),channels=4,num_interactions=1,max_ell=1,rngs=nnx.Rngs(0))
    g=build_graph([6,1],[[0,0,0],[0,0,1.09]],element_order=model.elements)
    raw=nnx.jit(lambda m,g:m(g))(model,g)
    assert raw.shape==(2,3,4)
    np.testing.assert_allclose(raw,b.reference_outputs(),atol=1e-14)
    model.head_bias[...]=jnp.zeros_like(model.head_bias[...])
    np.testing.assert_array_equal(model(g),jnp.zeros_like(raw))
    with pytest.raises(ValueError,match='zero'):model.assemble(b,g)


def test_non_p_block_templates_do_not_gain_angular_shells():
    for symbol in ['H','He','Li','Be','Na','Mg','K','Ca','Rb','Sr']:
        old=prepare_direct_basis(f'{symbol} 0 0 0',basis_family='szp3_direct')
        new=prepare_direct_basis(f'{symbol} 0 0 0')
        assert new.topology==old.topology
        assert new.nelectron==old.nelectron
