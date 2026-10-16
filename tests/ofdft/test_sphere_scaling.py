"""Dynamic objective normalization preserves the physical sphere result contract."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers.nonlinear import minimize_sphere, SphereConfig


CONFIG = SphereConfig(maxiter=100, tolerance=1e-9)


def energy(x, scale):
    return scale*(jnp.sum(jnp.array([1., 3.])*x*x)+.17)


def solve(scale):
    return minimize_sphere(energy, jnp.array([.8, .6]), scale,
                          config=CONFIG, energy_scale=scale)


@pytest.mark.parametrize('scale', [1., 81.])
def test_traced_scale_returns_raw_energy_residual_and_derivative(scale):
    result = jax.jit(solve)(jnp.asarray(scale))
    assert result.converged
    np.testing.assert_allclose(result.solution**2, [1., 0.], atol=1e-12)
    np.testing.assert_allclose(result.energy, 1.17*scale, atol=2e-12, rtol=0)
    raw_gradient = jax.grad(lambda x: energy(x/jnp.linalg.norm(x), scale))(result.solution)
    np.testing.assert_allclose(result.residual_norm, jnp.linalg.norm(raw_gradient), atol=2e-12, rtol=0)
    assert result.residual_norm <= CONFIG.tolerance
    derivative = jax.jit(jax.grad(lambda number: solve(number).energy))(jnp.asarray(scale))
    np.testing.assert_allclose(derivative, 1.17, atol=2e-11, rtol=0)


@pytest.mark.parametrize('scale', [0., -1., np.inf, np.nan])
def test_invalid_scale_is_rejected_under_jit(scale):
    result = jax.jit(solve)(jnp.asarray(scale))
    assert not result.converged
    assert not np.isfinite(result.energy)
    assert not np.isfinite(result.residual_norm)


def test_default_scale_retains_unscaled_contract():
    unscaled = minimize_sphere(energy, jnp.array([.8, .6]), jnp.asarray(3.), config=CONFIG)
    explicit_one = minimize_sphere(energy, jnp.array([.8, .6]), jnp.asarray(3.),
                                  config=CONFIG, energy_scale=jnp.asarray(1.))
    assert unscaled.converged and explicit_one.converged
    for left, right in zip(unscaled, explicit_one):
        np.testing.assert_allclose(left, right, atol=0., rtol=0.)


@pytest.mark.parametrize('mode', ['implicit', 'explicit'])
def test_ofdft_electron_number_remains_a_dynamic_leaf(mode):
    from gradscf import ofdft
    from gradscf.scf import SCFDifferentiationConfig
    config = ofdft.OFDFTConfig(xc=None, maxiter=20, tolerance=1e-10,
        differentiation=SCFDifferentiationConfig(mode=mode))
    def calculation(number):
        data = ofdft.periodic_inputs(jnp.eye(3)*5., (3, 3, 3), nelectron=number)
        return ofdft.run_ofdft(data, config=config)
    number = jnp.asarray(2.)
    value, derivative = jax.jit(jax.value_and_grad(lambda n: calculation(n).total_energy))(number)
    # Uniform TF has E proportional to N^(5/3); vW and the Hartree zero mode vanish.
    np.testing.assert_allclose(derivative, 5*value/(3*number), atol=2e-11, rtol=0.)
    density_response = jax.jit(lambda n: jax.jvp(lambda m: calculation(m).density,
        (n,), (jnp.asarray(1.),))[1])(number)
    np.testing.assert_allclose(density_response, 1/125, atol=2e-12, rtol=0.)
