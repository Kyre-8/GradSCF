from importlib import import_module
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.model.nnao import prepare_direct_basis
from gradscf import integrals
from gradscf.integrals.basis.contraction import primitive_basis, contraction_matrix


def test_expanded_pool_embeds_previous_basis_with_same_ao_count():
    atom='C 0 0 0; H 0 0 1.09'
    old=prepare_direct_basis(atom)
    new=prepare_direct_basis(atom,basis_family='szp663_direct')
    assert new.topology.primitive_counts==(6,6,6,3,6,2)
    assert new.topology.nao==old.topology.nao==14
    assert new.reference_outputs().shape==(2,3,6)
    for a,b,c in zip(new.parameters.exponents,old.parameters.exponents,new.parameters.coefficients):
        np.testing.assert_array_equal(a[:len(b)],b)
        np.testing.assert_array_equal(c[len(b):],0.)
    assert new.slots==(-1,0,1,2,0,1)
    for operator in ('overlap','kinetic','nuclear','eri'):
        np.testing.assert_allclose(integrals.make_plan(new.topology).evaluate(operator,new.parameters),
                                   integrals.make_plan(old.topology).evaluate(operator,old.parameters),atol=2e-10,rtol=1e-11)
    methane=prepare_direct_basis('C 0 0 0; H 1 1 1; H -1 -1 1; H -1 1 -1; H 1 -1 -1',basis_family='szp663_direct')
    assert methane.topology.nao==26
    assert primitive_basis(methane.topology,methane.parameters)[0].nao==93


def test_new_hydrogen_p_channel_is_differentiable_and_core_is_fixed():
    b=prepare_direct_basis('C 0 0 0; H 0 0 1.09',basis_family='szp663_direct')
    x=b.reference_outputs();direction=jnp.zeros_like(x).at[1,1,1].set(.3).at[0,0,4].set(.2)
    top,params=primitive_basis(b.topology,b.parameters)
    primitive_overlap=integrals.make_plan(top).evaluate('overlap',params)
    def overlap(v):
        t=contraction_matrix(b.topology,b.bind(v))
        return t.T@primitive_overlap@t
    tangent=jax.jvp(overlap,(x,),(direction,))[1]
    step=1e-5
    np.testing.assert_allclose(tangent,(overlap(x+step*direction)-overlap(x-step*direction))/(2*step),atol=1e-8,rtol=1e-7)
    assert np.linalg.norm(tangent)>1e-3
    changed=jax.jit(b.bind)(x+direction)
    np.testing.assert_array_equal(changed.coefficients[0],b.parameters.coefficients[0])
    assert not np.allclose(changed.coefficients[-1],b.parameters.coefficients[-1])


def test_real_mace_expanded_head_and_no_fixed_residual():
    pytest.importorskip('cuequivariance');pytest.importorskip('mace_jax')
    from flax import nnx
    from gradscf.model.nnao import MACEBasisModel,build_graph
    b=prepare_direct_basis('C 0 0 0; H 0 0 1.09',basis_family='szp663_direct')
    m=MACEBasisModel(elements=(1,6),channels=4,num_interactions=1,max_ell=1,basis_family='szp663_direct',rngs=nnx.Rngs(0))
    g=build_graph([6,1],[[0,0,0],[0,0,1.09]],element_order=m.elements)
    raw=nnx.jit(lambda m,g:m(g))(m,g)
    np.testing.assert_allclose(raw,b.reference_outputs(),atol=1e-14)
    m.head_bias[...]=jnp.zeros_like(m.head_bias[...])
    np.testing.assert_array_equal(m(g),jnp.zeros_like(raw))


def test_warm_start_preserves_nonreference_contractions():
    import runpy
    embed=vars(import_module("gradscf.model.nnao.training.objective"))['embedded_outputs']
    old=prepare_direct_basis('C 0 0 0; H 0 0 1.09')
    new=prepare_direct_basis('C 0 0 0; H 0 0 1.09',basis_family='szp663_direct')
    params=old.bind(old.reference_outputs().at[0,0,3].set(.2).at[1,0,1].add(.1))
    restored=new.bind(embed(new,old.atom_shells(params)))
    for op in ('overlap','kinetic','nuclear','eri'):
        np.testing.assert_allclose(integrals.make_plan(new.topology).evaluate(op,restored),
                                   integrals.make_plan(old.topology).evaluate(op,params),atol=2e-10,rtol=1e-11)
    corrupted=old.atom_shells(params);corrupted[0][0][1][0]*=1.1
    with pytest.raises(AssertionError):embed(new,corrupted)
