from importlib import import_module
import runpy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.model.nnao import prepare_direct_basis
from gradscf.integrals.basis.data import load_basis_from_snapshot


@pytest.mark.parametrize('symbol','Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca'.split())
def test_six_primitive_core_uses_631g_and_is_fixed(symbol):
    new=prepare_direct_basis(f'{symbol} 0 0 0',core_primitives=6)
    old=prepare_direct_basis(f'{symbol} 0 0 0',core_primitives=3)
    source=load_basis_from_snapshot('6-31g',symbol)
    counters={}
    changed=jax.jit(new.bind)(new.reference_outputs()+.1)
    assert new.topology.nao==old.topology.nao
    assert new.nelectron==old.nelectron
    for i,(l,role) in enumerate(zip(new.topology.angular_momenta,new.roles)):
        if role=='core':
            shells=[s for s in source if s[0]==l]
            index=counters.get(l,0);counters[l]=index+1
            rows=np.asarray(shells[index][1:])
            assert len(rows)==6
            np.testing.assert_array_equal(new.parameters.exponents[i],rows[:,0])
            np.testing.assert_array_equal(new.parameters.coefficients[i],rows[:,1:])
            np.testing.assert_array_equal(changed.coefficients[i],new.parameters.coefficients[i])
        else:
            np.testing.assert_array_equal(new.parameters.exponents[i],old.parameters.exponents[i])
            np.testing.assert_array_equal(new.parameters.coefficients[i],old.parameters.coefficients[i])


def test_methane_core6_default_energy_and_gradient():
    cls=vars(import_module("gradscf.model.nnao.training.objective"))['BasisRHF']
    experiment=cls()
    assert experiment.layout.topology.primitive_counts[0]==6
    assert experiment.layout.topology.nao==26
    assert experiment.primitive_topology.nao==56
    x=experiment.layout.reference_outputs()
    assert x.shape==(5,3,4)
    energy,gradient,info=experiment.evaluate(x)
    assert info['converged'] and energy < -40.1
    direction=jnp.zeros_like(x).at[0,0,3].set(.2).at[0,1,3].set(-.3).at[0,2,1].set(.4)
    step=1e-4
    e=lambda h:experiment.evaluate(x+h*direction)[0]
    fd=(8*(e(step)-e(-step))-e(2*step)+e(-2*step))/(12*step)
    np.testing.assert_allclose(jnp.sum(gradient*direction),fd,atol=2e-6,rtol=2e-5)


def test_core_choice_is_explicit_and_never_silently_padded():
    with pytest.raises(ValueError,match='core_primitives'):
        prepare_direct_basis('C 0 0 0',core_primitives=4)
    with pytest.raises(ValueError,match='six-primitive core'):
        prepare_direct_basis('I 0 0 0',core_primitives=6)
    assert prepare_direct_basis('I 0 0 0',core_primitives=3).nelectron==53
    for symbol in ('H','He'):
        assert prepare_direct_basis(f'{symbol} 0 0 0',core_primitives=6).topology==prepare_direct_basis(f'{symbol} 0 0 0',core_primitives=3).topology


@pytest.mark.parametrize('symbol',['C','Cl','Ca'])
def test_core6_integrals_match_independent_reference(symbol):
    pytest.importorskip('pyscf')
    from pyscf import gto
    from gradscf import integrals
    b=prepare_direct_basis(f'{symbol} .1 .2 .3; H .2 .1 1.4')
    p=b.bind(b.reference_outputs());labels=[symbol+'0','H1']
    mol=gto.M(atom=list(zip(labels,np.asarray(p.nuclear_coords))),basis=dict(zip(labels,b.atom_shells(p))),
              unit='Bohr',cart=False,spin=b.nelectron%2,verbose=0)
    plan=integrals.make_plan(b.topology)
    for op,name in [('overlap','int1e_ovlp'),('kinetic','int1e_kin'),('nuclear','int1e_nuc'),('eri','int2e')]:
        np.testing.assert_allclose(plan.evaluate(op,p),mol.intor(name),atol=1e-9,rtol=1e-11)


def test_real_mace_assembles_default_core6_without_larger_head():
    pytest.importorskip('cuequivariance');pytest.importorskip('mace_jax')
    from flax import nnx
    from gradscf.model.nnao import MACEBasisModel,build_graph
    b=prepare_direct_basis('C 0 0 0; H 0 0 1.09')
    model=MACEBasisModel(elements=(1,6),channels=4,num_interactions=1,max_ell=1,rngs=nnx.Rngs(0))
    graph=build_graph([6,1],[[0,0,0],[0,0,1.09]],element_order=model.elements)
    raw=model(graph)
    assert raw.shape==(2,3,4)
    np.testing.assert_allclose(raw,b.reference_outputs(),atol=1e-14)
    p=nnx.jit(lambda m,g:m.assemble(b,g))(model,graph)
    np.testing.assert_array_equal(p.coefficients[0],b.parameters.coefficients[0])
    assert p.coefficients[0].shape==(6,1)
