"""WGC99 second-order density-dependent kernel and differentiable response."""
import jax
import jax.numpy as jnp
import numpy as np


def test_wgc_uniform_energy_and_lindhard_response():
    from gradscf.ofdft import periodic_inputs,density_features,KineticFunctional
    data=periodic_inputs(jnp.eye(3)*7.,(7,7,7),nelectron=8.)
    rho0=8/343
    mode=jnp.cos(2*jnp.pi*data.coordinates[:,0]/7.)
    f=KineticFunctional('wgc')
    def energy(t):
        return f({'reference_density':rho0},density_features(jnp.sqrt(rho0+t*mode),data))
    tf=KineticFunctional('tf')({},density_features(jnp.full(343,jnp.sqrt(rho0)),data))
    np.testing.assert_allclose(energy(0.),tf,atol=1e-10)
    eta=(2*np.pi/7)/(2*(3*np.pi**2*rho0)**(1/3))
    L=.5+(1-eta**2)/(4*eta)*np.log(abs((1+eta)/(1-eta)))
    expected=np.pi**2/((3*np.pi**2*rho0)**(1/3)*L)*343/2
    np.testing.assert_allclose(jax.grad(jax.grad(energy))(0.),expected,rtol=1e-7)


def test_wgc_is_not_wt_and_energy_gradient_matches_fd():
    from gradscf.ofdft import periodic_inputs,density_features,KineticFunctional
    data=periodic_inputs(jnp.eye(3)*6.,(5,5,5),nelectron=8.)
    rho0=8/216
    mode=jnp.cos(2*jnp.pi*data.coordinates[:,0]/6.)
    def energy(t,name):
        features=density_features(jnp.sqrt(rho0*(1+t*mode)),data)
        return KineticFunctional(name)({},features)
    value,grad=jax.jit(jax.value_and_grad(lambda t:energy(t,'wgc')))(.3)
    fd=(energy(.30001,'wgc')-energy(.29999,'wgc'))/2e-5
    assert abs(float(value-energy(.3,'wt'))) > 1e-6
    np.testing.assert_allclose(grad,fd,rtol=1e-5,atol=1e-7)


def test_wgc_kernel_table_satisfies_defining_ode():
    from gradscf.ofdft.kinetic.wgc import kernel_values
    eta=jnp.array([.02,.2,.6,.95,1.05,2.,5.,30.])
    w,d1,d2=kernel_values(eta)
    from gradscf.ofdft.kinetic.nonlocal_kernel import inverse_lindhard_minus_tf_vw
    rhs=20*inverse_lindhard_minus_tf_vw(eta)
    np.testing.assert_allclose(d2+(2.7-10)*d1+20*w,rhs,rtol=1e-11,atol=1e-11)
    # d1 is d w / d log(eta), not d w / d eta.
    slope=jax.jvp(lambda t:kernel_values(jnp.exp(t))[0],(jnp.log(eta),),(jnp.ones_like(eta),))[1]
    np.testing.assert_allclose(slope,d1,rtol=1e-10,atol=1e-10)


def test_wgc_matches_independent_numpy_ode_and_analytic_potential():
    import importlib.util
    from pathlib import Path
    from gradscf.ofdft import periodic_inputs,density_features
    from gradscf.ofdft.kinetic.wgc import wang_govind_carter
    path=Path(__file__).resolve().parents[2]/'examples/ofdft/wgc_dftpy.py'
    spec=importlib.util.spec_from_file_location('wgc_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=periodic_inputs(jnp.eye(3)*6.,(7,7,7),nelectron=8.)
    rho0=8/216
    rho=rho0*(1+.2*jnp.cos(2*jnp.pi*data.coordinates[:,0]/6.)+.1*jnp.sin(2*jnp.pi*data.coordinates[:,1]/6.))
    f=density_features(jnp.sqrt(rho),data)
    reference=module.WGCReference(np.asarray(f.gvectors),data.mesh,216.,rho0)
    energy=lambda n:wang_govind_carter(density_features(jnp.sqrt(n),data),rho0)
    value,grad=jax.value_and_grad(energy)(rho)
    expected,potential=reference.evaluate(np.asarray(rho))
    np.testing.assert_allclose(value,expected,atol=2e-9,rtol=2e-8)
    np.testing.assert_allclose(grad/data.weights,potential.reshape(-1),atol=2e-8,rtol=2e-7)


def test_wgc_against_libkedf_published_first_frame():
    """Legacy fixture uses a 5e-5 ODE tolerance and an even skew-cell mesh."""
    from pathlib import Path
    from gradscf.ofdft import KineticFeatures
    from gradscf.integrals.periodic.ao import reciprocal_grid
    from gradscf.ofdft.kinetic.wgc import wang_govind_carter
    with np.load(Path(__file__).with_name('data')/'libkedf_wgc_first_frame.npz') as f:
        rho=f['rho'];lattice=f['lattice'];expected=f['energy'];potential=f['potential'];ref=float(f['reference_density'])
    mesh=rho.shape;volume=np.linalg.det(lattice)
    weights=jnp.full(rho.size,volume/rho.size)
    vectors=reciprocal_grid(jnp.asarray(lattice),mesh)
    def energy(n):
        features=KineticFeatures(n,jnp.zeros((rho.size,3)),weights,jnp.zeros((rho.size,3)),
                                 0.,jnp.sqrt(n),vectors,volume,mesh)
        return wang_govind_carter(features,ref)
    value,derivative=jax.jit(jax.value_and_grad(energy))(jnp.asarray(rho.ravel()))
    np.testing.assert_allclose(value,expected,rtol=0,atol=3e-6)
    np.testing.assert_allclose(np.asarray(derivative/weights).reshape(mesh),potential,rtol=0,atol=1e-5)


def test_wgc_implicit_density_response_matches_fd():
    from dataclasses import replace
    from gradscf.ofdft import periodic_inputs,run_ofdft,OFDFTConfig,KineticFunctional
    data=periodic_inputs(jnp.eye(3)*5.,(3,3,3),nelectron=2.)
    mode=jnp.cos(2*jnp.pi*data.coordinates[:,0]/5.)
    cfg=OFDFTConfig(xc=None,tolerance=1e-9,maxiter=400)
    def loss(strength):
        result=run_ofdft(replace(data,external_potential=strength*mode),kinetic=KineticFunctional('wgc'),
                        config=cfg,kinetic_params={'reference_density':2/125})
        return jnp.sum(data.weights*mode*result.density)
    value,derivative=jax.jit(jax.value_and_grad(loss))(.04)
    fd=(loss(.04001)-loss(.03999))/2e-5
    assert np.isfinite(value) and abs(float(derivative))>.01
    np.testing.assert_allclose(derivative,fd,rtol=2e-5,atol=1e-7)


def test_wgc_reference_density_parameter_derivative():
    from gradscf.ofdft import periodic_inputs,density_features
    from gradscf.ofdft.kinetic.wgc import wang_govind_carter
    data=periodic_inputs(jnp.eye(3)*6.,(5,5,5),nelectron=8.)
    mode=jnp.cos(2*jnp.pi*data.coordinates[:,0]/6.)
    features=density_features(jnp.sqrt((8/216)*(1+.2*mode)),data)
    energy=lambda ref:wang_govind_carter(features,ref)
    derivative=jax.grad(energy)(.04)
    fd=(energy(.040001)-energy(.039999))/2e-6
    np.testing.assert_allclose(derivative,fd,rtol=2e-5,atol=1e-7)


def test_wgc_rejects_nonscalar_or_nonpositive_reference_density():
    import pytest
    from gradscf.ofdft.kinetic.wgc import wgc_kernels
    vectors=jnp.zeros((3,3))
    with pytest.raises(ValueError,match='scalar'):
        wgc_kernels(vectors,jnp.ones(3))
    for value in (0.,-1.,float('nan')):
        with pytest.raises(ValueError,match='positive'):
            wgc_kernels(vectors,value)
