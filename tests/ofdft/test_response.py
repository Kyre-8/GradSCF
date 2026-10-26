"""A resolved density response, including neural parameters and constraints."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_shared_sphere_optimizer():
    from gradscf.solvers.nonlinear import minimize_sphere, SphereConfig
    energy = lambda x, p: jnp.sum(p*x*x)
    result = minimize_sphere(energy, jnp.array([.8,.6]), jnp.array([1.,3.]),
                             config=SphereConfig(maxiter=100, tolerance=1e-9))
    assert result.converged
    np.testing.assert_allclose(result.solution**2, [1.,0.], atol=1e-8)


def test_periodic_self_consistency_and_implicit_density_response():
    from gradscf.ofdft import periodic_inputs, run_ofdft, OFDFTConfig
    data = periodic_inputs(jnp.eye(3)*5., (3,3,3), nelectron=2.)
    probe = jnp.cos(2*jnp.pi*data.coordinates[:,0]/5.)
    cfg = OFDFTConfig(maxiter=300, tolerance=1e-9, xc=None)
    def calculation(strength):
        return run_ofdft(replace(data, external_potential=strength*probe), config=cfg)
    result = calculation(.1)
    assert result.converged
    np.testing.assert_allclose(result.electron_number, 2., atol=1e-12)
    assert jnp.all(result.density >= 0)
    loss = lambda strength: jnp.sum(data.weights*probe*calculation(strength).density)
    ad = jax.jit(jax.grad(loss))(.1)
    fd = (loss(.10001)-loss(.09999))/2e-5
    assert abs(float(ad)) > .01
    np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-7)


def test_neural_kinetic_nested_pytree_training_gradient():
    from gradscf.ofdft import periodic_inputs, run_ofdft, OFDFTConfig
    from gradscf.ofdft.kinetic import NeuralKineticFunctional
    data = periodic_inputs(jnp.eye(3)*5., (3,3,3), nelectron=2.)
    probe = jnp.cos(2*jnp.pi*data.coordinates[:,0]/5.)
    data = replace(data, external_potential=.1*probe)
    def correction(params, f):
        hidden = jnp.tanh(f.rho[:,None]*params['layer']['w']+params['layer']['b'])
        return jnp.sum(f.weights*f.rho*(hidden@params['out']))
    functional = NeuralKineticFunctional(correction)
    cfg = OFDFTConfig(maxiter=400, tolerance=1e-9, xc=None)
    def loss(w):
        params = {'network': {'layer': {'w': jnp.array([w, -.2]), 'b': jnp.array([.1,.3])},
                              'out': jnp.array([.7,.4])}}
        result = run_ofdft(data, kinetic=functional, kinetic_params=params, config=cfg)
        return jnp.sum(data.weights*probe*result.density)
    ad = jax.jit(jax.grad(loss))(.4)
    fd = (loss(.4001)-loss(.3999))/2e-4
    assert abs(float(ad)) > 1e-4
    np.testing.assert_allclose(ad, fd, rtol=2e-4, atol=1e-6)


def test_unconverged_response_is_not_silently_accepted():
    from gradscf.ofdft import periodic_inputs, run_ofdft, OFDFTConfig
    data = periodic_inputs(jnp.eye(3)*5., (3,3,3), nelectron=2.)
    probe = jnp.cos(2*jnp.pi*data.coordinates[:,0]/5.)
    def loss(t):
        result = run_ofdft(replace(data, external_potential=t*probe),
                          config=OFDFTConfig(maxiter=1, tolerance=1e-13, xc=None))
        return jnp.sum(probe*result.density)
    assert jnp.isnan(jax.grad(loss)(.4))


def test_symmetric_stationary_start_retains_explicit_response():
    from gradscf.ofdft import periodic_inputs, run_ofdft, OFDFTConfig
    from gradscf.scf import SCFDifferentiationConfig
    data = periodic_inputs(jnp.eye(3)*5.,(3,3,3),nelectron=2.)
    probe = jnp.cos(2*jnp.pi*data.coordinates[:,0]/5.)
    def loss(p,mode):
        cfg = OFDFTConfig(xc=None,maxiter=50,tolerance=1e-10,
            differentiation=SCFDifferentiationConfig(mode=mode))
        out = run_ofdft(replace(data,external_potential=p*probe),config=cfg)
        return jnp.sum(data.weights*probe*out.density)
    implicit = jax.grad(lambda t:loss(t,'implicit'))(0.)
    explicit = jax.grad(lambda t:loss(t,'explicit'))(0.)
    np.testing.assert_allclose(explicit,implicit,rtol=1e-7,atol=1e-9)


def test_wt_dirac_exchange_reaches_requested_stationarity():
    from gradscf.ofdft import periodic_inputs,run_ofdft,OFDFTConfig,KineticFunctional
    data = periodic_inputs(jnp.eye(3)*6.,(7,7,7),nelectron=8.)
    phase = 2*jnp.pi*data.coordinates/6.
    data = replace(data,external_potential=.15*jnp.cos(phase[:,0])+.1*jnp.sin(phase[:,1])+.05*jnp.cos(phase[:,2]))
    exchange = lambda p,f: -.75*(3/jnp.pi)**(1/3)*jnp.sum(f.weights*jnp.abs(f.phi)**(8/3))
    result = run_ofdft(data,kinetic=KineticFunctional('wt'),xc_energy_fn=exchange,
        config=OFDFTConfig(xc=None,tolerance=1e-9,maxiter=700))
    assert result.converged, result.residual_norm
