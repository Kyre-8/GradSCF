"""Analytic uniform-fluid response at fixed density and widely separated N.

The small collocation grid exactly resolves the declared Fourier perturbation;
this tests KKT numerical units, not material/grid accuracy.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dfpt, ofdft
from gradscf.scf import SCFDifferentiationConfig


@pytest.mark.parametrize('number', [2., 8192.])
def test_intensive_fourier_response_and_constant_gauge_survive_large_number(number):
    rho = 2/125
    length = (number/rho)**(1/3)
    data = ofdft.periodic_inputs(jnp.eye(3)*length, (3,)*3, nelectron=number)
    wave = (jnp.cos(2*jnp.pi*data.coordinates[:, 0]/length)
            +.3*jnp.sin(2*jnp.pi*data.coordinates[:, 1]/length))
    potential = .17+wave
    config = ofdft.OFDFTConfig(xc=None, maxiter=20, tolerance=2*number*1e-9,
        differentiation=SCFDifferentiationConfig(tolerance=1e-10, max_iter=100, restart=40))
    calculation = lambda t: ofdft.run_ofdft(
        replace(data, external_potential=t*potential), config=config)
    ground = jax.jit(calculation)(0.)
    assert ground.converged
    response = jax.jit(lambda t: jax.jvp(lambda value: calculation(value).density,
        (t,), (jnp.asarray(1.),))[1])(0.)
    assert np.isfinite(response).all()
    k2 = (2*np.pi/length)**2
    c_tf = .3*(3*np.pi**2)**(2/3)
    denominator = (10/9)*c_tf*rho**(-1/3)+k2/(4*rho)+4*np.pi/k2
    np.testing.assert_allclose(response, -wave/denominator, atol=2e-11, rtol=2e-7)
    assert abs(float(data.weights@response)/number) < 1e-10
    chemical = jax.jit(jax.grad(lambda t: calculation(t).chemical_potential))(0.)
    np.testing.assert_allclose(chemical, .17, atol=1e-9, rtol=0.)
    curvature = jax.jit(jax.grad(jax.grad(lambda t: calculation(t).total_energy)))(0.)
    np.testing.assert_allclose(curvature, -length**3*(1+.3**2)/(2*denominator), atol=2e-8, rtol=2e-8)
    cotangent = data.weights*wave/number
    reverse = jax.jit(jax.grad(lambda t: cotangent@calculation(t).density))(0.)
    np.testing.assert_allclose(reverse, cotangent@response, atol=1e-10, rtol=1e-8)
    problem = dfpt.ofdft_problem(data, ground, config=config)
    linear = problem.response(potential=potential)
    assert linear.converged
    np.testing.assert_allclose(linear.density, response, atol=2e-11, rtol=2e-7)
    np.testing.assert_allclose(linear.values['chemical_potential'], .17, atol=1e-9, rtol=0.)
    series = problem.series(potential=potential, order=2)
    assert series.converged
    np.testing.assert_allclose(2*series.energy[1], curvature, atol=2e-8, rtol=2e-8)


def test_moving_number_and_volume_mu_chart_keeps_parameter_derivatives():
    config = ofdft.OFDFTConfig(xc=None, maxiter=20, tolerance=1e-8)
    def values(parameters):
        number, length = parameters
        data = ofdft.periodic_inputs(jnp.eye(3)*length, (3,)*3, nelectron=number)
        out = ofdft.run_ofdft(data, config=config)
        return jnp.array([out.total_energy, out.chemical_potential, out.density[0]])
    def exact(parameters):
        number, length = parameters
        density = number/length**3
        c_tf = .3*(3*jnp.pi**2)**(2/3)
        return jnp.array([c_tf*number*density**(2/3), (5/3)*c_tf*density**(2/3), density])
    parameters, direction = jnp.array([8192., 80.]), jnp.array([.7, -.2])
    np.testing.assert_allclose(jax.jit(jax.jacfwd(values))(parameters),
        jax.jacfwd(exact)(parameters), atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(jax.jit(jax.jacrev(values))(parameters),
        jax.jacrev(exact)(parameters), atol=2e-9, rtol=2e-9)
    hvp = lambda fn: jax.jvp(jax.grad(lambda p: fn(p)[0]), (parameters,), (direction,))[1]
    np.testing.assert_allclose(jax.jit(lambda: hvp(values))(), hvp(exact), atol=2e-9, rtol=2e-9)


def test_gaussian_overlap_parameter_agrees_across_moving_and_fixed_charts():
    data = ofdft.OFDFTInputs('gaussian', jnp.ones(2), jnp.zeros((2, 3)),
        jnp.array(4.), jnp.array([[-.3, .04], [.04, .1]]),
        overlap=jnp.array([[1., .1], [.1, 1.2]]), kinetic_matrix=jnp.eye(2),
        ao=jnp.eye(2), ao_gradient=jnp.zeros((2, 2, 3)), eri=jnp.zeros((2,)*4))
    config = ofdft.OFDFTConfig(xc=None, maxiter=200, tolerance=1e-10)
    change = jnp.array([[.1, .03], [.03, -.02]])
    def solve(t):
        return ofdft.run_ofdft(replace(data, overlap=data.overlap+t*change), config=config)
    ground = solve(0.)
    assert ground.converged
    problem = dfpt.ofdft_problem(data, ground, config=config)
    zero = jax.tree.map(jnp.zeros_like, problem.inputs)
    response = problem.response((replace(zero[0], overlap=change), zero[1], zero[2]))
    assert response.converged
    ad = jax.jit(lambda t: jax.jvp(lambda s: solve(s).density, (t,), (jnp.array(1.),))[1])(0.)
    plus, minus = solve(1e-4), solve(-1e-4)
    assert plus.converged and minus.converged
    np.testing.assert_allclose(ad, response.density, atol=1e-9, rtol=1e-8)
    np.testing.assert_allclose(ad, (plus.density-minus.density)/2e-4, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(response.values['electron_number'], 0., atol=1e-10)


def test_dfpt_reference_gate_preserves_raw_stationarity_units():
    data = ofdft.periodic_inputs(jnp.eye(3)*20., (3,)*3, nelectron=128.)
    config = ofdft.OFDFTConfig(xc=None, tolerance=1e-10)
    ground = ofdft.run_ofdft(data, config=config)
    # A fake mu offset gives raw components above tolerance even though dividing
    # by N would make them small. Normalization must not accept this reference.
    invalid = ground._replace(chemical_potential=ground.chemical_potential+1e-7)
    problem = dfpt.ofdft_problem(data, invalid, config=config, tolerance=1e-6)
    assert not problem._base()[1]
