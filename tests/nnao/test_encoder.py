import importlib
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_graph_units_elements_and_disconnected_batch():
    m=importlib.import_module('gradscf.model.nnao')
    assert hasattr(m,'build_graph')
    order=(1,9,17,35,53)
    graph=m.build_graph([1,53,1],[[0,0,0],[0,0,2],[0,0,0]],element_order=order,batch=[0,0,1])
    assert graph['node_attrs'].shape==(3,5)
    np.testing.assert_array_equal(graph['node_type'],[0,4,0])
    assert np.all(np.asarray(graph['edge_index'])<2)
    assert graph['ptr'].tolist()==[0,2,3]
    with pytest.raises(ValueError,match='element'):m.build_graph([54],[[0,0,0]],element_order=order)
    with pytest.raises(ValueError,match='coincident'):m.build_graph([1,1],[[0,0,0],[0,0,0]],element_order=order)


def test_real_mace_equivariance_basis_and_backward():
    pytest.importorskip('cuequivariance')
    pytest.importorskip('mace_jax')
    from flax import nnx
    from gradscf.model.nnao import MACEBasisModel,build_graph,prepare_basis
    from gradscf.integrals.basis.contraction import primitive_basis, contraction_matrix
    from gradscf import integrals
    z=[1,9,17,35,53]
    r=np.array([[0,0,0],[0,0,1.0],[1.7,.2,0],[.3,2.3,.5],[-2,.1,.2]])
    model=MACEBasisModel(basis_family="szp3",elements=tuple(z),channels=4,num_interactions=2,max_ell=1,correlation=2,zero_init=False,rngs=nnx.Rngs(3))
    graph=build_graph(z,r,element_order=model.elements)
    out=model(graph)
    assert out.shape==(5,2,2) and np.isfinite(out).all()
    np.testing.assert_allclose(nnx.jit(lambda net,g:net(g))(model,graph),out,atol=2e-10,rtol=1e-9)
    rotation=np.linalg.qr(np.random.default_rng(5).normal(size=(3,3)))[0]
    rotated=build_graph(z,r@rotation+[.4,.6,.8],element_order=model.elements)
    np.testing.assert_allclose(model(rotated),out,atol=2e-9,rtol=1e-8)
    perm=np.array([4,1,0,3,2])
    pg=build_graph(np.array(z)[perm],r[perm],element_order=model.elements)
    np.testing.assert_allclose(model(pg),out[perm],atol=2e-9,rtol=1e-8)
    for symbol,number in [('H',1),('F',9),('Cl',17),('Br',35),('I',53)]:
        b=prepare_basis(f'{symbol} 0 0 0')
        g=build_graph([number],[[0,0,0]],element_order=model.elements)
        assert np.isfinite(model.assemble(b,g).coefficients[-1]).all()
    b=prepare_basis('H 0 0 0; F 0 0 1')
    g=build_graph([1,9],r[:2],element_order=model.elements)
    pt,pp=primitive_basis(b.topology,b.parameters)
    kinetic=integrals.make_plan(pt).evaluate('kinetic',pp)
    def objective(net):
        t=contraction_matrix(b.topology,net.assemble(b,g))
        return jnp.trace(t.T@kinetic@t)
    value,grad=nnx.value_and_grad(objective)(model)
    leaves=jax.tree_util.tree_leaves(nnx.to_pure_dict(grad))
    assert np.isfinite(value) and all(np.isfinite(v).all() for v in leaves)
    assert sum(float(jnp.sum(v*v)) for v in leaves)>1e-10
    backbone=jax.tree_util.tree_leaves(nnx.to_pure_dict(grad.backbone))
    assert sum(float(jnp.sum(v*v)) for v in backbone)>1e-12
    index=(0,0,0); initial=model.head_kernel[...]
    step=1e-5
    model.head_kernel[...]=initial.at[index].add(step);plus=objective(model)
    model.head_kernel[...]=initial.at[index].add(-step);minus=objective(model)
    model.head_kernel[...]=initial
    np.testing.assert_allclose(grad.head_kernel[index],(plus-minus)/(2*step),atol=2e-7,rtol=2e-5)
    zero=MACEBasisModel(basis_family="szp3",channels=4,num_interactions=1,max_ell=1,rngs=nnx.Rngs(1))
    n=len(zero.elements)
    isolated=build_graph(zero.elements,np.zeros((n,3)),batch=np.arange(n),element_order=zero.elements)
    np.testing.assert_array_equal(zero(isolated),np.zeros((n,2,2)))
