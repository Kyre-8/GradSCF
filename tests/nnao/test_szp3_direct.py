from importlib import import_module
from pathlib import Path
import runpy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import integrals
from gradscf.model.nnao import prepare_direct_basis as _prepare_direct_basis
from functools import partial
prepare_direct_basis=partial(_prepare_direct_basis,basis_family="szp3_direct")


def test_default_direct_path_has_full_charges_core_and_electrons():
    b=prepare_direct_basis('C 0 0 0; H 0 0 1.09')
    assert b.topology.nuclear_charges==(6,1)
    assert b.nelectron==7
    assert all(e is None for e in b.ecps)
    assert b.topology.nao==14
    assert b.roles.count('core')==1
    assert b.topology.primitive_counts==(3,3,3,1,3,1)
    x=b.reference_outputs();p=b.bind(x)
    updated=jax.jit(b.bind)(x.at[0,0,0].add(.3))
    for i,role in enumerate(b.roles):
        if role=='core':np.testing.assert_array_equal(p.coefficients[i],updated.coefficients[i])
    assert not np.allclose(p.coefficients[1],updated.coefficients[1])


@pytest.mark.parametrize('symbol,charge,nd',[('F',9,0),('Cl',17,0),('Br',35,1),('I',53,2)])
def test_halogen_explicit_core_and_all_electron_integrals(symbol,charge,nd):
    pytest.importorskip('pyscf')
    from pyscf import gto
    b=prepare_direct_basis(f'{symbol} .1 .2 .3; H .2 .1 1.4')
    assert b.topology.nuclear_charges==(charge,1)
    assert all(e is None for e in b.ecps)
    assert sum(l==2 and r=='core' for l,r in zip(b.topology.angular_momenta,b.roles))==nd
    p=b.bind(b.reference_outputs());labels=[symbol+'0','H1']
    mol=gto.M(atom=list(zip(labels,np.asarray(p.nuclear_coords))),basis=dict(zip(labels,b.atom_shells(p))),
              unit='Bohr',cart=False,spin=b.nelectron%2,verbose=0)
    plan=integrals.make_plan(b.topology)
    for op,name in [('overlap','int1e_ovlp'),('nuclear','int1e_nuc')]:
        np.testing.assert_allclose(plan.evaluate(op,p),mol.intor(name),atol=1e-9,rtol=1e-11)


def test_methane_all_electron_energy_and_gradient_without_ecp(monkeypatch):
    from gradscf.integrals.plan import IntegralPlan
    original=IntegralPlan.evaluate
    def reject_ecp(self,operator,*args,**kwargs):
        assert operator!='ecp','The all-electron path must not evaluate an ECP'
        return original(self,operator,*args,**kwargs)
    monkeypatch.setattr(IntegralPlan,'evaluate',reject_ecp)
    cls=vars(import_module("gradscf.model.nnao.training.objective"))['BasisRHF']
    experiment=cls(basis_family='szp3_direct')
    assert experiment.nelectron==10 and experiment.layout.topology.nao==26
    x=experiment.layout.reference_outputs();energy,gradient,info=experiment.evaluate(x)
    assert info['converged'] and energy<-39.
    direction=jnp.zeros_like(x).at[0,0,0].set(.2).at[0,1,1].set(-.3).at[1:,0,2].set(.1)
    h=1e-4;e=lambda a:experiment.evaluate(x+a*direction)[0]
    fd=(8*(e(h)-e(-h))-e(2*h)+e(-2*h))/(12*h)
    np.testing.assert_allclose(jnp.sum(gradient*direction),fd,atol=2e-6,rtol=2e-5)


def test_real_mace_direct_three_primitive_head():
    pytest.importorskip('cuequivariance');pytest.importorskip('mace_jax')
    from flax import nnx
    from gradscf.model.nnao import MACEBasisModel,build_graph
    b=prepare_direct_basis('C 0 0 0; H 0 0 1.09')
    model=MACEBasisModel(elements=(1,6),channels=4,num_interactions=1,max_ell=1,
                         basis_family='szp3_direct',rngs=nnx.Rngs(0))
    graph=build_graph([6,1],[[0,0,0],[0,0,1.09]],element_order=model.elements)
    raw=model(graph)
    assert raw.shape==(2,2,3)
    np.testing.assert_allclose(raw,b.reference_outputs(),atol=1e-14)
    p=model.assemble(b,graph)
    assert all(e is None for e in b.ecps)
    for i,role in enumerate(b.roles):
        if role=='core':np.testing.assert_array_equal(p.coefficients[i],b.parameters.coefficients[i])
    model.head_bias[...]=jnp.zeros_like(model.head_bias[...])
    np.testing.assert_array_equal(model(graph),jnp.zeros_like(raw))
    with pytest.raises(ValueError,match='zero'):model.assemble(b,graph)
