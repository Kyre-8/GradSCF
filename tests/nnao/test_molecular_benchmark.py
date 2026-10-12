"""Non-methane geometry handling and memory-safe integral paths."""
from importlib import import_module
import runpy

import jax.numpy as jnp
import numpy as np
import pytest


def aniline_size_geometry():
    # A simple finite geometry is enough for topology and allocation tests.
    return dict(name='aniline-size',symbols=['C']*6+['N']+['H']*7,
                coords_angstrom=[[1.4*i,0.,0.] for i in range(14)],charge=0,spin=0)


@pytest.mark.parametrize('family',['szp442_direct','szp663_direct'])
def test_aniline_training_configures_direct_jk_without_eri(monkeypatch,family):
    from gradscf.integrals.plan import IntegralPlan
    operators=[]
    def bounded(self,operator,*args,**kwargs):
        assert operator!='eri'
        operators.append(operator)
        n=self.topology.nao
        return jnp.eye(n) if operator=='overlap' else jnp.zeros((n,n))
    monkeypatch.setattr(IntegralPlan,'evaluate',bounded)
    geometry=aniline_size_geometry()
    cls=vars(import_module("gradscf.model.nnao.training.objective"))['BasisRHF']
    experiment=cls(geometry=geometry,basis_family=family)
    assert experiment.jk_backend=='direct' and experiment.rep is None
    assert operators==['overlap','kinetic','nuclear']


def test_nitrogen_geometry_energy_gradient_and_species_mapping():
    from gradscf.model.nnao import prepare_direct_basis
    geometry=dict(name='ammonia',symbols=['N','H','H','H'],charge=0,spin=0,
                  coords_angstrom=[[0,0,.12],[0,.94,-.28],[.814,-.47,-.28],[-.814,-.47,-.28]])
    cls=vars(import_module("gradscf.model.nnao.training.objective"))['BasisRHF']
    ex=cls(geometry=geometry,basis_family='szp442_direct')
    assert ex.nelectron==10 and ex.molecule=='ammonia' and ex.bond is None
    assert ex.layout.topology.nao==22
    x=ex.layout.reference_outputs()
    energy,gradient,info=ex.evaluate(x)
    assert info['converged'] and energy < -55.
    direction=jnp.zeros_like(x).at[0,0,3].set(.2).at[0,1,3].set(-.3).at[1:,0,1].set(.1)
    h=1e-4;e=lambda v:ex.evaluate(x+v*direction)[0]
    fd=(8*(e(h)-e(-h))-e(2*h)+e(-2*h))/(12*h)
    np.testing.assert_allclose(jnp.sum(gradient*direction),fd,atol=2e-6,rtol=2e-5)
    assert prepare_direct_basis(list(zip(geometry['symbols'],geometry['coords_angstrom']))).topology==ex.layout.topology


def test_comparison_resource_guard_precedes_integrals(monkeypatch):
    from gradscf.integrals.plan import IntegralPlan
    def forbidden(*args,**kwargs):pytest.fail('ERI budget must be checked before integral calls')
    monkeypatch.setattr(IntegralPlan,'evaluate',forbidden)
    geometry=aniline_size_geometry()
    calculate=runpy.run_path('tools/compare_methane_core6_basis_sets.py')['calculate']
    with pytest.raises(MemoryError,match='contracted ERI alone'):
        calculate('cc-pvtz',geometry)


def test_current_mace_namespace_accepts_carbon_hydrogen_nitrogen():
    pytest.importorskip('cuequivariance');pytest.importorskip('mace_jax')
    from flax import nnx
    from gradscf.model.nnao import MACEBasisModel,build_graph,prepare_direct_basis
    symbols=['C','N','H'];coords=[[0.,0.,0.],[1.4,0.,0.],[0.,1.1,0.]]
    layout=prepare_direct_basis(list(zip(symbols,coords)),basis_family='szp663_direct')
    model=MACEBasisModel(elements=(1,6,7),channels=4,num_interactions=1,max_ell=1,
                         basis_family='szp663_direct',rngs=nnx.Rngs(0))
    graph=build_graph([6,7,1],coords,element_order=model.elements)
    outputs=nnx.jit(lambda m,g:m(g))(model,graph)
    assert outputs.shape==(3,3,6)
    np.testing.assert_allclose(outputs,layout.reference_outputs(),atol=1e-14)
    parameters=model.assemble(layout,graph)
    assert len(parameters.coefficients)==len(layout.roles)
