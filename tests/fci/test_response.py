"""Fixed-topology FCI energy, density and complete-subspace derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import fci
from gradscf.solvers import EigenSolverConfig, EigenResponseConfig


def model(n=4):
    rng=np.random.default_rng(115)
    h=rng.normal(size=(n,n))*.2+np.diag(np.linspace(-1,1,n))
    h=(h+h.T)*.5
    factors=rng.normal(size=(n,n,n))*.13
    factors=(factors+factors.transpose(0,2,1))*.5
    return jnp.asarray(h),jnp.asarray(np.einsum('Lpq,Lrs->pqrs',factors,factors))


@pytest.mark.parametrize('method',['dense','davidson'])
def test_energy_and_density_jvp_vjp(method):
    h,g=model();space=fci.make_fci_space(4,(2,1))
    cfg=EigenSolverConfig(method=method,nroots=2,max_subspace=14,maxiter=150,atol=1e-10)
    policy=EigenResponseConfig(target='eigenpairs')
    dh=jnp.array([[.1,.04,0,0],[.04,-.2,.07,0],[0,.07,.03,.02],[0,0,.02,.09]])
    probe=jnp.arange(16.).reshape(4,4)/25
    def loss(t):
        r=fci.solve_fci(h+t*dh,g*(1+.2*t),space,ecore=.5+.3*t,config=cfg,response=policy)
        d1,d2=fci.make_rdm12(r.coefficients[0],space)
        return jnp.dot(r.total_energies,jnp.array([.3,.7])) + .1*jnp.sum(d1*probe)+.02*jnp.sum(d2**2)
    value,grad=jax.jit(jax.value_and_grad(loss))(.13)
    fd=(loss(.1301)-loss(.1299))/2e-4
    np.testing.assert_allclose(grad,fd,atol=3e-7,rtol=2e-6)
    np.testing.assert_allclose(jax.jvp(loss,(.13,),(1.,))[1],grad,atol=1e-9)
    assert np.isfinite(value)
    r=fci.solve_fci(h,g,space,config=cfg)
    d1,d2=fci.make_rdm12(r.coefficients[0],space)
    def energy(t):
        return fci.solve_fci(h+t*dh,g*(1+.2*t),space,ecore=.3*t,config=cfg).total_energies[0]
    expected=jnp.sum(d1*dh.T)+.1*jnp.sum(d2*g)+.3
    np.testing.assert_allclose(jax.grad(energy)(0.),expected,atol=2e-9)


def test_energy_only_rejects_density_derivative():
    h,g=model();space=fci.make_fci_space(4,(2,1))
    cfg=EigenSolverConfig(method='dense')
    def property(t):
        r=fci.solve_fci(h*jnp.exp(t),g,space,config=cfg)
        return fci.make_rdm1(r.coefficients[0],space)[0,0]
    assert np.isfinite(property(0.))
    assert not np.isfinite(jax.grad(property)(0.))


def test_degenerate_complete_cluster_response():
    space=fci.make_fci_space(3,(1,0))
    g=jnp.zeros((3,)*4)
    perturb=jnp.array([[.3,.1,.3],[.1,.6,-.2],[.3,-.2,.4]])
    cfg=EigenSolverConfig(method='dense',nroots=2)
    def loss(t):
        h=jnp.diag(jnp.array([0.,0.,5.]))+t*perturb
        out=fci.solve_fci(h,g,space,ecore=.7+.2*t,config=cfg,
                          response=EigenResponseConfig(target='subspace'),probes=jnp.eye(3))
        assert out.coefficients is None and out.total_energies is None
        return out.energy_sum+3*out.projection[0,2]-2*out.projection[1,2]
    value,grad=jax.jit(jax.value_and_grad(loss))(0.)
    np.testing.assert_allclose([value,grad],[1.4,1.04],atol=1e-10)
    np.testing.assert_allclose((loss(1e-4)-loss(-1e-4))/2e-4,grad,atol=1e-8)
    def individual(t):
        h=jnp.diag(jnp.array([0.,0.,5.]))+t*perturb
        return fci.solve_fci(h,g,space,config=EigenSolverConfig(method='dense')).total_energies[0]
    assert not np.isfinite(jax.grad(individual)(0.))


def test_orbital_rotation_invariance_and_vacuum():
    h,g=model();space=fci.make_fci_space(4,(2,1));cfg=EigenSolverConfig(method='dense')
    def energy(angle):
        c,s=jnp.cos(angle),jnp.sin(angle)
        u=jnp.eye(4).at[:2,:2].set(jnp.array([[c,-s],[s,c]]))
        gg=jnp.einsum('pqrs,pa,qb,rc,sd->abcd',g,u,u,u,u)
        return fci.solve_fci(u.T@h@u,gg,space,config=cfg).total_energies[0]
    np.testing.assert_allclose(energy(.3),energy(0.),atol=1e-12)
    np.testing.assert_allclose(jax.grad(energy)(.3),0.,atol=1e-10)
    empty=fci.make_fci_space(0,0)
    vacuum=lambda c: fci.solve_fci(jnp.zeros((0,0)),jnp.zeros((0,)*4),empty,ecore=c,config=cfg).total_energies[0]
    np.testing.assert_allclose(jax.jit(jax.value_and_grad(vacuum))(1.3),(1.3,1.),atol=1e-12)


def test_invalid_integrals_and_unconverged_response():
    h,g=model(5);space=fci.make_fci_space(5,(2,2))
    cfg=EigenSolverConfig(maxiter=1,max_subspace=6,atol=1e-13)
    result=fci.solve_fci(h,g,space,config=cfg)
    assert not np.all(result.converged)
    assert not np.isfinite(jax.grad(lambda t:fci.solve_fci(h*t,g,space,config=cfg).total_energies[0])(1.))
    bad=fci.solve_fci(h.at[0,1].add(.1),g,space,config=EigenSolverConfig(method='dense'))
    assert not np.any(bad.converged | bad.response_valid)


def test_davidson_has_no_determinant_square_arrays():
    h,g=model(6);space=fci.make_fci_space(6,(2,2));assert space.size==225
    cfg=EigenSolverConfig(max_subspace=12,atol=1e-9)
    fun=lambda t:fci.solve_fci(h*t,g,space,config=cfg).total_energies[0]
    def visit(value):
        if hasattr(value,'jaxpr'):visit(value.jaxpr)
        elif hasattr(value,'eqns'):
            for eq in value.eqns:
                for v in eq.outvars:
                    assert getattr(getattr(v,'aval',None),'shape',()) != (space.size,space.size)
                visit(eq.params)
        elif isinstance(value,dict):
            for v in value.values():visit(v)
        elif isinstance(value,(tuple,list)):
            for v in value:visit(v)
    visit(jax.make_jaxpr(fun)(1.))
    visit(jax.make_jaxpr(jax.grad(fun))(1.))
