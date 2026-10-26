"""One external Functional contract for complex periodic densities and SCF."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import dft,dfpt
from gradscf.pbc import gto
from gradscf.integrals.periodic.fft import build_inputs


def setup():
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,unit='Bohr',mesh=(9,)*3)
    data=build_inputs(cell,kpts=cell.make_kpts((2,1,1),shift=(.5,0,0)),kweights=[.3,.7])
    model=dft.Functional(lambda s:(s.rho,s.grad_rho,s.tau,s.laplacian_rho,s.weights),
        lambda p,x:p['scale']*jnp.sum(x[4]*(.1*x[0]**2+.02*jnp.sum(x[1]**2,axis=-1)+.01*x[2]+.001*x[3]**2)))
    return cell,data,model


def test_periodic_features_and_complex_potential_kernel():
    cell,data,model=setup()
    dm=jnp.broadcast_to(jnp.array([[.3,.04j],[-.04j,.2]]),(2,2,2,2))
    delta=jnp.broadcast_to(jnp.array([[.1,.03j],[-.03j,-.05]]),dm.shape)
    p={'scale':jnp.array(.2)}
    rho,gradient,tau,lapl,w=model.inputs_for_density(data,dm)
    np.testing.assert_allclose(jnp.dot(w,rho),jnp.einsum('k,skij,kji->',data.kweights,dm,data.overlap).real,atol=1e-10)
    assert gradient.shape==(len(w),3) and tau.shape==rho.shape and lapl.shape==rho.shape
    np.testing.assert_allclose(jnp.dot(w,lapl),0.,atol=1e-10)
    potential=model.potential(p,data,dm)
    np.testing.assert_allclose(potential,potential.conj().swapaxes(-1,-2),atol=1e-12)
    energy=lambda t:model.energy_for_density(p,data,dm+t*delta)
    derivative=jax.grad(energy)(0.)
    expected=jnp.einsum('k,skij,skji->',data.kweights,delta,potential).real
    np.testing.assert_allclose(derivative,expected,atol=1e-11)
    kernel=model.kernel_action(p,data,dm,delta)
    fd=(model.potential(p,data,dm+1e-5*delta)-model.potential(p,data,dm-1e-5*delta))/2e-5
    np.testing.assert_allclose(kernel,fd,atol=1e-9)


@pytest.mark.parametrize('mode',['explicit','implicit'])
def test_external_periodic_scf_parameter_gradient(mode):
    from gradscf.pbc.scf import run_kpoint_scf
    from gradscf.scf import SCFDifferentiationConfig
    cell,data,model=setup()
    def solve(scale):
        return run_kpoint_scf(data,mesh=cell.mesh,nelec=cell.nelec,functional=model,
            params={'scale':scale},temperature=.1,exxdiv=None,
            max_cycle=100,damping=.1,conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10,
            differentiation=SCFDifferentiationConfig(mode=mode,tolerance=1e-11,max_iter=80))
    value,derivative=jax.jit(jax.value_and_grad(lambda p:solve(p).free_energy))(.2)
    assert solve(.2).converged
    fd=(solve(.20001).free_energy-solve(.19999).free_energy)/2e-5
    np.testing.assert_allclose(derivative,fd,atol=1e-8,rtol=1e-5)
    assert abs(derivative)>1e-5
    result=solve(.2)
    problem=dfpt.periodic_problem(data,result,mesh=cell.mesh,nelec=cell.nelec,
        functional=model,params={'scale':jnp.array(.2)},temperature=.1,exxdiv=None)
    b=jnp.broadcast_to(jnp.array([[.1,.02],[.02,-.05]]),data.hcore.shape)
    response=problem.response(potential=b)
    assert response.converged
    np.testing.assert_allclose(response.values['electron_number'],0.,atol=1e-9)
    if mode=='implicit':
        def response_loss(scale):
            problem=dfpt.periodic_problem(data,solve(scale),mesh=cell.mesh,nelec=cell.nelec,
                functional=model,params={'scale':scale},temperature=.1,exxdiv=None)
            return jnp.sum(jnp.abs(problem.response(potential=b).density)**2)
        derivative=jax.jit(jax.grad(response_loss))(.2)
        expected=(response_loss(.20001)-response_loss(.19999))/2e-5
        np.testing.assert_allclose(derivative,expected,atol=1e-8,rtol=2e-5)


def test_external_periodic_facade_and_stale_parameters():
    from gradscf.pbc import dft as pdft
    cell,data,model=setup()
    mf=pdft.KRKS(cell,kpts=data.kpoints,functional=model,params={'scale':jnp.array(.2)},
                 temperature=.1,time_reversal=True).run(conv_tol=1e-11,conv_tol_density=1e-10)
    assert mf.converged and len(mf.reduction.representatives)==1
    with pytest.raises(ValueError,match='Set the periodic Functional'):
        dfpt.DFPT(mf,functional=model,params=mf.params)
    assert dfpt.DFPT(mf).response(potential=jnp.broadcast_to(jnp.eye(2),data.hcore.shape)).converged
    mf.params={'scale':jnp.array(.3)}
    with pytest.raises(ValueError,match='parameters'):
        dfpt.DFPT(mf)
    mf.run(params={'scale':jnp.array(.3)})
    assert mf.converged
    mf.xc='pbe'
    with pytest.raises(ValueError,match='no named xc baseline'):
        mf.kernel()


def test_fixed_density_external_periodic_parameter_gradient():
    _,data,model=setup()
    density=jnp.broadcast_to(jnp.eye(2)*.3,(2,2,2,2)).astype(complex)
    energy=lambda p:model.energy_for_density({'scale':p},data,density)
    np.testing.assert_allclose(jax.grad(energy)(.2),(energy(.20001)-energy(.19999))/2e-5,atol=1e-10)


def test_gamma_two_dimensional_density_potential_and_kernel():
    cell,_,model=setup()
    data=build_inputs(cell)
    density=jnp.array([[.5,.04j],[-.04j,.3]])
    tangent=jnp.array([[.1,.03j],[-.03j,-.05]])
    params={'scale':jnp.array(.2)}
    potential=model.potential(params,data,density)
    assert potential.shape==density.shape
    derivative=jax.grad(lambda t:model.energy_for_density(params,data,density+t*tangent))(0.)
    np.testing.assert_allclose(derivative,jnp.trace(potential@tangent).real,atol=1e-11)
    kernel=model.kernel_action(params,data,density,tangent)
    fd=(model.potential(params,data,density+1e-5*tangent)
        -model.potential(params,data,density-1e-5*tangent))/2e-5
    np.testing.assert_allclose(kernel,fd,atol=1e-9)
