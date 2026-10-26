"""Analytic kinetic limits, response, and neural parameter plumbing."""
import jax
import jax.numpy as jnp
import numpy as np


def test_tf_value_gradient_and_hvp():
    from gradscf.ofdft.kinetic import thomas_fermi
    rho = jnp.array([.2, .5, 1.3])
    weights = jnp.array([.4, .6, .8])
    c = .3 * (3 * np.pi**2)**(2/3)
    energy = lambda n: thomas_fermi(n, weights)
    np.testing.assert_allclose(energy(rho), c * np.sum(weights * rho**(5/3)))
    np.testing.assert_allclose(jax.grad(energy)(rho), (5/3)*c*weights*rho**(2/3))
    hvp = jax.jvp(jax.grad(energy), (rho,), (jnp.ones(3),))[1]
    np.testing.assert_allclose(hvp, (10/9)*c*weights*rho**(-1/3))


def test_periodic_vw_and_wt_uniform_limit():
    from gradscf.ofdft import periodic_inputs, density_features
    from gradscf.ofdft.kinetic import KineticFunctional
    data = periodic_inputs(jnp.eye(3)*4., (5,5,5), nelectron=2.)
    q = jnp.ones(125)*jnp.sqrt(2/64)
    features = density_features(q, data)
    np.testing.assert_allclose(features.vw_energy, 0., atol=1e-14)
    tf = KineticFunctional('tf')({}, features)
    wt = KineticFunctional('wt')({}, features)
    np.testing.assert_allclose(wt, tf, atol=1e-12)


def test_wt_uniform_response_matches_lindhard():
    from gradscf.ofdft import periodic_inputs, density_features
    from gradscf.ofdft.kinetic import KineticFunctional
    length, number = 7., 8.
    data = periodic_inputs(jnp.eye(3)*length, (7,7,7), nelectron=number)
    rho0 = number/length**3
    mode = jnp.cos(2*jnp.pi*data.coordinates[:,0]/length)
    functional = KineticFunctional('wt')
    def energy(t):
        return functional({}, density_features(jnp.sqrt(rho0+t*mode), data))
    eta = (2*np.pi/length)/(2*(3*np.pi**2*rho0)**(1/3))
    lindhard = .5 + (1-eta**2)/(4*eta)*np.log(abs((1+eta)/(1-eta)))
    response = np.pi**2 / ((3*np.pi**2*rho0)**(1/3)*lindhard)
    np.testing.assert_allclose(jax.grad(jax.grad(energy))(0.), response*length**3/2, rtol=1e-9)


def test_density_nodes_have_finite_amplitude_hessian():
    from gradscf.ofdft import periodic_inputs, density_features, KineticFunctional
    data = periodic_inputs(jnp.eye(3)*5.,(3,3,3),nelectron=2.)
    q = jnp.ones(27)*.1
    q = q.at[0].set(0.)
    energy = lambda x: KineticFunctional('tfvw')({},density_features(x,data))
    hvp = jax.jvp(jax.grad(energy),(q,),(jnp.ones_like(q),))[1]
    assert jnp.isfinite(hvp).all()


def test_wt_kohn_anomaly_does_not_claim_finite_kernel_derivative():
    from gradscf.ofdft.kinetic.nonlocal_kernel import inverse_lindhard_minus_tf_vw
    assert np.isfinite(inverse_lindhard_minus_tf_vw(1.))
    assert not np.isfinite(jax.grad(inverse_lindhard_minus_tf_vw)(1.))


def test_wt_kohn_anomaly_higher_response_stays_invalid():
    from gradscf.ofdft.kinetic.nonlocal_kernel import inverse_lindhard_minus_tf_vw
    assert not np.isfinite(jax.grad(jax.grad(inverse_lindhard_minus_tf_vw))(1.))
