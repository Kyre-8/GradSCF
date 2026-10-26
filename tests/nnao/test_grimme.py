from importlib import import_module
import importlib
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import integrals


def grimme():
    m=importlib.import_module('gradscf.model.nnao')
    assert hasattr(m,'prepare_grimme_basis'),'Direct Grimme basis API is missing'
    return m.prepare_grimme_basis


def test_original_grimme_shapes_and_direct_output():
    b=grimme()('C 0 0 0; H 0 0 1.09')
    assert b.topology.nao==13  # spherical C spd (9) + H sp (4)
    assert b.topology.nuclear_charges==(4,1)
    assert b.nelectron==5
    assert b.topology.primitive_counts==(4,4,2,5,1)
    raw=b.reference_outputs()
    p=b.bind(raw)
    scaled=b.bind(raw.at[0,0].multiply(2.))
    np.testing.assert_allclose(p.coefficients[0],scaled.coefficients[0],atol=1e-14)
    different=jax.jit(b.bind)(raw.at[0,0,0].set(.7))
    assert not np.allclose(p.coefficients[0],different.coefficients[0])
    assert p.coefficients[0].shape==(4,1)
    with pytest.raises(ValueError,match='zero'): b.bind(jnp.zeros_like(raw))


@pytest.mark.parametrize('symbol',['C','F','Cl','Br','I'])
@pytest.mark.parametrize('cart',[False,True])
def test_native_ecp_matches_pyscf(symbol,cart):
    pytest.importorskip('pyscf')
    from pyscf import gto
    b=grimme()(f'{symbol} .1 .2 .3; H .2 .3 1.4',cart=cart)
    p=b.bind(b.reference_outputs())
    labels=[symbol+'0','H1']
    shells=b.atom_shells(p)
    mol=gto.M(atom=list(zip(labels,np.asarray(p.nuclear_coords))),basis=dict(zip(labels,shells)),
              ecp={labels[0]:b.ecps[0].as_raw()},cart=cart,unit='Bohr',spin=b.nelectron%2,verbose=0)
    plan=integrals.make_plan(b.topology)
    actual=plan.evaluate('ecp',p,ecps=b.ecps)
    np.testing.assert_allclose(actual,mol.intor('ECPscalar'),atol=2e-9,rtol=1e-10)
    np.testing.assert_allclose(plan.evaluate('nuclear',p),mol.intor('int1e_nuc'),atol=2e-10,rtol=1e-11)
    np.testing.assert_allclose(jax.jit(lambda p:plan.evaluate('ecp',p,ecps=b.ecps))(p),actual,atol=1e-12)
    from gradscf.integrals.basis.contraction import primitive_basis, contraction_matrix, contract_integrals
    pt,pp=primitive_basis(b.topology,p)
    ep=integrals.make_plan(pt).evaluate('ecp',pp,ecps=b.ecps)
    def value(raw):
        t=contraction_matrix(b.topology,b.bind(raw))
        return jnp.sum(t.T@ep@t)
    raw=b.reference_outputs();direction=jnp.zeros_like(raw).at[0,0,0].set(1.)
    g=jax.grad(value)(raw)[0,0,0]
    np.testing.assert_allclose(g,(value(raw+1e-5*direction)-value(raw-1e-5*direction))/2e-5,atol=2e-6,rtol=1e-6)


def test_mace_exposes_direct_basis_family():
    import inspect
    from gradscf.model.nnao.encoder import MACEBasisModel
    assert 'basis_family' in inspect.signature(MACEBasisModel.__init__).parameters


def test_real_mace_predicts_direct_coefficients():
    pytest.importorskip('cuequivariance');pytest.importorskip('mace_jax')
    from flax import nnx
    from gradscf.model.nnao import MACEBasisModel,build_graph
    b=grimme()('C 0 0 0; H 0 0 1.09')
    model=MACEBasisModel(elements=(1,6),channels=4,num_interactions=1,max_ell=1,
                         basis_family='qvszps',rngs=nnx.Rngs(1))
    graph=build_graph([6,1],[[0,0,0],[0,0,1.09]],element_order=model.elements)
    out=model(graph)
    np.testing.assert_allclose(out,b.reference_outputs(),atol=1e-14)
    assert out.shape==(2,3,5)
    np.testing.assert_allclose(model.assemble(b,graph).coefficients[0],b.parameters.coefficients[0],atol=1e-14)
    # With a zero kernel and zero trainable bias, no hidden empirical baseline
    # may survive the network. The assembler must reject the zero contraction.
    model.head_bias[...]=jnp.zeros_like(model.head_bias[...])
    np.testing.assert_array_equal(model(graph),jnp.zeros_like(out))
    with pytest.raises(ValueError,match='zero'):model.assemble(b,graph)


def test_grimme_methane_implicit_gradient():
    from pathlib import Path
    import runpy
    cls=vars(import_module("gradscf.model.nnao.training.objective"))['BasisRHF']
    experiment=cls(basis_family='qvszps')
    x=experiment.layout.reference_outputs()
    energy,gradient,info=experiment.evaluate(x)
    assert info['converged'] and experiment.nelectron==8
    direction=jnp.zeros_like(x).at[0,0,0].set(.2).at[0,1,1].set(-.3).at[0,2,0].set(.4).at[1:,0,2].set(.1)
    step=1e-4
    e=lambda a:experiment.evaluate(x+a*direction)[0]
    fd=(8*(e(step)-e(-step))-e(2*step)+e(-2*step))/(12*step)
    np.testing.assert_allclose(jnp.sum(gradient*direction),fd,atol=2e-6,rtol=2e-5)
    assert experiment.evaluate(x-1e-3*gradient)[0]<energy


def test_direct_ecp_geometry_ad_is_explicitly_unsupported():
    from dataclasses import replace
    b=grimme()('C 0 0 0; H 0 0 1.09')
    p=b.bind(b.reference_outputs());plan=integrals.make_plan(b.topology)
    assert integrals.backend_capabilities('native').supports('ecp')
    assert not integrals.backend_capabilities('native').supports('ecp',variable='centers',derivative_order=1)
    with pytest.raises(ValueError,match='cannot be differentiated'):
        jax.grad(lambda centers:jnp.sum(plan.evaluate('ecp',replace(p,centers=centers),ecps=b.ecps)))(p.centers)


def test_all_main_group_grimme_templates():
    from gradscf.model.nnao import supported_elements
    for symbol in supported_elements():
        b=grimme()(f'{symbol} 0 0 0')
        p=b.bind(b.reference_outputs())
        assert np.isfinite(np.concatenate([np.asarray(c).ravel() for c in p.coefficients])).all()
        assert b.nelectron>0
        assert b.topology.nao==(4 if symbol in ('H','He') else 9)


def test_grimme_input_requires_physical_nuclear_identity():
    from dataclasses import replace
    from gradscf.data.molecule import parse_molecule_spec
    spec=replace(parse_molecule_spec('C 0 0 0'),charges=jnp.array([4.]))
    with pytest.raises(ValueError,match='physical nuclear'):
        grimme()(spec)
