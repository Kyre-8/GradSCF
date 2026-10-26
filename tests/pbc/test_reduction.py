"""Exact finite-grid spatial stars and differentiable AO reconstruction."""
from dataclasses import replace
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.pbc import gto
from gradscf.pbc.symmetry import Operation


def cell(hexagonal=False,two_atoms=False):
    a=np.array([[5.,0,0],[2.5,np.sqrt(3)*2.5,0],[0,0,7.]]) if hexagonal else np.eye(3)*6
    return gto.M(atom='He 0 0 0'+('; He 3 0 0' if two_atoms else ''),a=a,
        unit='Bohr',basis={'He':[[0,[.8,1.]],[1,[.6,1.]],[2,[.5,1.]]]},
        pseudo='gth-pade',mesh=(9,)*3)


def plan(c,k,operations,**kwargs):
    from gradscf.pbc.reduction import SpatialReduction
    return SpatialReduction.from_cell(c,np.asarray(k)@np.asarray(c.reciprocal_vectors()),
                                      operations=operations,**kwargs)


def test_generator_closure_weights_and_frozen_metadata():
    c=cell();op=Operation.from_cell(c,[[0,-1,0],[1,0,0],[0,0,1]])
    k=np.array([[.2,0,0],[0,.2,0],[-.2,0,0],[0,-.2,0],[0,0,0]])
    p=plan(c,k,[op],kweights=[.2]*5)
    assert p.integrates
    np.testing.assert_array_equal(p.representatives,[0,4])
    np.testing.assert_array_equal(p.full_to_ir,[0,0,0,0,1])
    np.testing.assert_allclose(p.reduce_weights(jnp.full(5,.2)),[.8,.2])
    assert p.weights_consistent(jnp.full(5,.2))
    assert p.weights_consistent(jnp.array([.1,.1,.1,.1,.6]))
    np.testing.assert_allclose(p.reduce_weights(jnp.array([.1,.1,.1,.1,.6])),[.4,.6])
    assert not p.weights_consistent(jnp.array([.1,.3,.2,.2,.2]))
    with pytest.raises(ValueError):p.representatives[0]=1
    n=c.topology.nao;rng=np.random.default_rng(2)
    x=rng.normal(size=(2,n,n))+1j*rng.normal(size=(2,n,n));d=x@x.conj().swapaxes(-1,-2)
    full=p.expand(d);expected=d[0];index=0
    for _ in range(4):
        np.testing.assert_allclose(full[index],expected,atol=1e-11)
        expected=op.density(k[index],expected)
        index=int(np.argmin(np.linalg.norm(k-np.asarray(op.rotate_k(k[index])),axis=1)))


def test_nonunitary_cartesian_mirror_maps_density_and_operator_differently():
    c=cell(hexagonal=True);op=Operation.from_cell(c,[[0,1,0],[1,0,0],[0,0,1]])
    k=np.array([[.13,.21,.07],[.21,.13,.07]]);p=plan(c,k,[op]);n=c.topology.nao
    rng=np.random.default_rng(8);x=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    d=jnp.asarray(x@x.conj().T)[None];f=jnp.asarray(x+x.conj().T)[None]
    moved=p.expand(d);operator=p.expand_operator(f)
    np.testing.assert_allclose(moved[1],op.density(k[0],d[0]),atol=1e-11)
    np.testing.assert_allclose(operator[1],op.operator(k[0],f[0]),atol=1e-11)
    assert np.linalg.norm(np.asarray(p.expand(f)-operator))>.1
    np.testing.assert_allclose(jnp.trace(moved[1]@operator[1]),jnp.trace(d[0]@f[0]),atol=1e-10)
    assert p.consistent(operator) and p.consistent_density(moved)
    assert not p.consistent(moved)


def test_self_stabilizers_are_checked_not_only_reconstruction_paths():
    c=cell();op=Operation.from_cell(c,-np.eye(3));p=plan(c,[[0,0,0]],[op]);n=c.topology.nao
    f=jnp.eye(n,dtype=complex)[None].at[0,0,1].set(1j).at[0,1,0].set(-1j)
    assert not p.consistent(f)
    assert not p.consistent_density(f)
    assert p.consistent(jnp.eye(n)[None])


def test_combined_time_reversal_and_spatial_paths():
    c=cell();op=Operation.from_cell(c,np.diag([-1,-1,1]))
    k=np.array([[.1,.2,.3],[-.1,-.2,.3],[-.1,-.2,-.3],[.1,.2,-.3]])
    p=plan(c,k,[op],time_reversal=True);n=c.topology.nao
    rng=np.random.default_rng(3);x=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));d=(x@x.conj().T)[None]
    full=p.expand(d)
    assert len(p.representatives)==1
    np.testing.assert_allclose(full[2],d[0].conj(),atol=1e-12)
    np.testing.assert_allclose(full[3],op.density(-k[0],d[0].conj()),atol=1e-11)
    assert p.consistent_density(full)
    gamma=plan(c,[[0,0,0]],[],time_reversal=True)
    assert not gamma.consistent(d)


@pytest.mark.parametrize('reason',['translation','gbox','unwrapped','weights'])
def test_explicit_operations_reject_inexact_finite_discretizations(reason):
    c=cell(hexagonal=reason=='gbox',two_atoms=reason=='translation')
    w=np.eye(3);t=np.zeros(3);k=np.array([[.2,0,0],[-.2,0,0]]);kwargs={}
    if reason=='translation':t[0]=.5
    elif reason=='gbox':w=np.array([[0,-1,0],[1,1,0],[0,0,1]]);k=np.zeros((1,3))
    else:w=-np.eye(3)
    if reason=='unwrapped':k=np.array([[0,0,0],[-.5,0,0]])
    if reason=='weights':kwargs['kweights']=[.3,.7]
    op=Operation.from_cell(c,w,t)
    with pytest.raises(ValueError,match='grid|mesh|weight|discretization'):
        plan(c,k,[op],**kwargs)


def test_discovery_filters_unsupported_operations():
    pytest.importorskip('spglib')
    c=cell(two_atoms=True)
    p=plan(c,[[.17,0,0],[-.17,0,0]],None)
    assert len(p.representatives)==1
    assert all(np.allclose(op.translation*np.array(c.mesh),np.rint(op.translation*np.array(c.mesh)))
               for op in p.operations)


@pytest.mark.parametrize('change',['basis','geometry'])
def test_existing_operations_reject_changed_cell(change):
    c=cell();op=Operation.from_cell(c,-np.eye(3))
    if change=='basis':
        exponents=list(c.parameters.exponents);exponents[0]=exponents[0]*1.01
        c.parameters=replace(c.parameters,exponents=tuple(exponents))
    else:c.coords=c.coords.at[0,0].add(.1)
    with pytest.raises(ValueError,match='changed'):plan(c,[[0,0,0]],[op])


def test_sampling_guard_allows_lattice_ad_and_rejects_k_grid_or_weight_changes():
    c=cell();p=plan(c,[[.2,0,0],[-.2,0,0]],[Operation.from_cell(c,-np.eye(3))])
    integers=np.stack(np.meshgrid(*[np.fft.fftfreq(n)*n for n in c.mesh],indexing='ij'),axis=-1).reshape(-1,3)
    def inputs(scale=1.):
        reciprocal=c.reciprocal_vectors()/scale
        return SimpleNamespace(lattice=c.lattice*scale,kpoints=jnp.array([[.2,0,0],[-.2,0,0]])@reciprocal,
                               gvectors=integers@reciprocal,kweights=jnp.array([.5,.5]))
    assert p.sampling_consistent(inputs()) and p.sampling_consistent(inputs(1.07))
    changed=inputs();changed.kpoints=changed.kpoints.at[0,0].add(.01)
    assert not p.sampling_consistent(changed)
    changed=inputs();changed.gvectors=changed.gvectors[:-1]
    assert not p.sampling_consistent(changed)
    changed=inputs();changed.kweights=jnp.array([.4,.6])
    assert not p.sampling_consistent(changed)
    assert jax.jit(lambda scale:p.sampling_consistent(inputs(scale)))(1.03)


@pytest.mark.parametrize('time_reversal',[False,True])
def test_reconstruction_maps_support_jit_jvp_vjp(time_reversal):
    c=cell(hexagonal=True);op=Operation.from_cell(c,[[0,1,0],[1,0,0],[0,0,1]])
    k=np.array([[.1,.2,.03],[.2,.1,.03]])
    if time_reversal:k=np.concatenate([k,-k])
    p=plan(c,k,[op],time_reversal=time_reversal);n=c.topology.nao
    rng=np.random.default_rng(15);x=jnp.asarray(rng.normal(size=(2,1,n,n))+1j*rng.normal(size=(2,1,n,n)))
    for transform in (p.expand,p.expand_operator):
        y=jax.jit(transform)(x)
        _,tangent=jax.jvp(transform,(x,),(x*.3,))
        np.testing.assert_allclose(tangent,y*.3,atol=1e-12)
        loss=lambda v:jnp.sum(jnp.abs(transform(v))**2)
        _,pullback=jax.vjp(loss,x)
        gradient=pullback(jnp.array(1.))[0]
        step=1e-5;fd=(loss(x+step*x*.3)-loss(x-step*x*.3))/(2*step)
        np.testing.assert_allclose(jnp.sum(gradient*x*.3).real,fd,atol=1e-6,rtol=1e-8)


def test_identity_only_plan_preserves_all_targets_and_does_not_reduce():
    c=cell();p=plan(c,[[.1,.2,.3],[-.2,.17,.4]],[]);n=c.topology.nao
    np.testing.assert_array_equal(p.representatives,[0,1])
    x=jnp.arange(2*n*n).reshape(2,n,n)
    np.testing.assert_allclose(p.expand(x),x,atol=1e-13)
    np.testing.assert_allclose(p.expand_operator(x),x,atol=1e-13)
    assert p.consistent(x)
