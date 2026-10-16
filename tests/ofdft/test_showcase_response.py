"""P08 analytic fixed-number OFDFT response and manufactured-potential gates."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dfpt, ofdft
from gradscf.solvers import LinearSolverConfig


@pytest.mark.parametrize('tf,vw', [(1., 0.), (0., 1.), (.8, .6)])
def test_uniform_tf_vw_fourier_response_and_normalized_taylor(tf, vw):
    from examples.ofdft.parameter_density_response import external_kinetic, CONFIG

    length, number = 5., 2.
    data = ofdft.periodic_inputs(jnp.eye(3)*length, (3, 3, 3), nelectron=number)
    data = replace(data, external_potential=jnp.full_like(data.weights, .17))
    params = {'weights': {'tf': jnp.asarray(tf), 'vw': jnp.asarray(vw)}}
    ground = ofdft.run_ofdft(data, kinetic=external_kinetic, kinetic_params=params, config=CONFIG)
    assert ground.converged
    rho = number/length**3
    wave = jnp.cos(2*jnp.pi*data.coordinates[:, 0]/length)
    k2 = (2*np.pi/length)**2
    c_tf = .3*(3*np.pi**2)**(2/3)
    denominator = (10/9)*tf*c_tf*rho**(-1/3)+vw*k2/(4*rho)+4*np.pi/k2
    expected = -wave/denominator
    problem = dfpt.ofdft_problem(data, ground, kinetic=external_kinetic, kinetic_params=params, config=CONFIG)
    direct = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13, max_dense=128)
    response = problem.response(potential=wave, config=direct)
    series = problem.series(potential=wave, order=2, config=direct)
    assert response.converged & series.converged
    np.testing.assert_allclose(ground.density, rho, atol=2e-14, rtol=0)
    np.testing.assert_allclose(response.density, expected, atol=3e-12, rtol=1e-10)
    np.testing.assert_allclose(response.values['electron_number'], 0., atol=2e-12, rtol=0)
    np.testing.assert_allclose(series.density[0], expected, atol=3e-12, rtol=1e-10)
    np.testing.assert_allclose(series.energy[0], 0., atol=2e-12, rtol=0)
    # Series stores the t^2 coefficient E''/2!, not the second derivative.
    np.testing.assert_allclose(2*series.energy[1], -length**3/(2*denominator), atol=2e-10, rtol=1e-10)
    constant = problem.response(potential=jnp.ones_like(wave), config=direct)
    np.testing.assert_allclose(constant.density, 0., atol=2e-12, rtol=0)
    np.testing.assert_allclose(constant.values['chemical_potential'], 1., atol=2e-12, rtol=0)
    np.testing.assert_allclose(constant.energy, number, atol=2e-12, rtol=0)


def test_manufactured_potential_density_number_and_chemical_potential():
    from examples.ofdft.parameter_density_response import make_model, external_kinetic, CONFIG

    data, params, exact = make_model((5, 5, 3))
    ground = ofdft.run_ofdft(data, kinetic=external_kinetic, kinetic_params=params, config=CONFIG)
    assert ground.converged and ground.iterations > 0
    assert np.min(np.asarray(ground.density)) > 0
    np.testing.assert_allclose(ground.electron_number, 2., atol=2e-12, rtol=0)
    np.testing.assert_allclose(ground.density, exact['density'], atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(ground.chemical_potential, exact['chemical_potential'], atol=2e-8, rtol=0)


def test_external_parameter_response_is_a_dynamic_leaf_and_matches_scipy_kkt():
    from examples.ofdft.parameter_density_response import make_model, external_kinetic, CONFIG
    from examples.ofdft.compare_parameter_density_response import independent_response

    data, params, _ = make_model((5, 5, 3))
    ground = ofdft.run_ofdft(data, kinetic=external_kinetic, kinetic_params=params, config=CONFIG)
    assert ground.converged
    def density(weight):
        varied = {'weights': {'tf': params['weights']['tf'], 'vw': weight}}
        return ofdft.run_ofdft(data, kinetic=external_kinetic, kinetic_params=varied, config=CONFIG).density
    weight = params['weights']['vw']
    derivative = jax.jit(lambda w: jax.jvp(density, (w,), (jnp.asarray(1.),))[1])(weight)
    expected = independent_response(np.asarray(data.lattice), data.mesh, np.asarray(ground.density),
                                    float(params['weights']['tf']), float(weight), kind='vw')[0]
    np.testing.assert_allclose(derivative, expected, atol=2e-8, rtol=2e-6)
    assert np.linalg.norm(derivative) > 1e-4
    np.testing.assert_allclose(jnp.dot(data.weights, derivative), 0., atol=2e-10, rtol=0)
    probe = jnp.sin(2*jnp.pi*data.coordinates[:, 0]/5.)+.7*jnp.cos(2*jnp.pi*data.coordinates[:, 1]/5.)
    scalar = jax.jit(jax.grad(lambda w: jnp.sum(data.weights*probe*density(w))))(weight)
    np.testing.assert_allclose(scalar, jnp.sum(data.weights*probe*derivative), atol=2e-9, rtol=2e-7)
    step = 2e-4
    finite = (density(weight+step)-density(weight-step))/(2*step)
    np.testing.assert_allclose(derivative, finite, atol=3e-7, rtol=3e-5)
