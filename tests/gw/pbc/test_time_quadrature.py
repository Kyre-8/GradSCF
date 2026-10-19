"""Independent time-integration controls for the PBC imaginary-axis grid."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_matsubara import _model, _fermi, _dressed_bath_oracles


def _grid(nw=32, beta=400., quadrature='gauss', factor=4):
    module = importlib.import_module('gradscf.gw.pbc.matsubara')
    return module.matsubara_grid(nw=nw, beta=beta, time_oversampling=factor,
                                 time_quadrature=quadrature)


def test_positive_symmetric_quadrature_preserves_frequency_window_and_molecular_grid():
    from gradscf.gw.matsubara import matsubara_grid as molecular_grid
    baseline = molecular_grid(nw=32, beta=400.)
    assert not hasattr(baseline, 'tau_weights')
    for quadrature in ('uniform', 'gauss'):
        grid = _grid(quadrature=quadrature)
        weights, tau = np.asarray(grid.tau_weights), np.asarray(grid.tau)
        assert len(tau) == 256 and np.all(weights > 0)
        np.testing.assert_allclose(weights.sum(), grid.beta, atol=2e-12, rtol=0)
        np.testing.assert_allclose(tau+tau[::-1], grid.beta, atol=1e-13, rtol=0)
        np.testing.assert_allclose(weights, weights[::-1], atol=1e-14, rtol=0)
        np.testing.assert_array_equal(grid.fermion, baseline.fermion)
        np.testing.assert_array_equal(grid.boson, baseline.boson)
        np.testing.assert_array_equal(grid.boson_weights, np.ones(65))
        actual = jax.jit(lambda g: jnp.sum(g.tau_weights))(grid)
        np.testing.assert_allclose(actual, grid.beta, atol=2e-12, rtol=0)
    uniform = _grid(quadrature='uniform', factor=1)
    np.testing.assert_array_equal(uniform.tau, baseline.tau)
    np.testing.assert_array_equal(uniform.boson_weights, baseline.boson_weights)
    with pytest.raises(ValueError, match='quadrature'):
        _grid(quadrature='unknown')


def test_gauss_resolves_endpoint_exponentials_and_retained_product_fourier_modes():
    grid, uniform = _grid(), _grid(quadrature='uniform')
    rate = np.array([.01, .1, 1., 10.])
    exact = -np.expm1(-grid.beta*rate)/rate
    errors = []
    for current in (uniform, grid):
        values = np.exp(-np.asarray(current.tau)[:, None]*rate)
        integrals = np.asarray(current.tau_weights)@values
        errors.append(np.max(abs(integrals/exact-1)))
        if current is grid:
            reverse = np.exp(-(grid.beta-np.asarray(grid.tau))[:, None]*rate)
            np.testing.assert_allclose(integrals, np.asarray(grid.tau_weights)@reverse,
                                       atol=2e-12, rtol=2e-12)
    assert errors[0] > .1
    assert errors[1] < 2e-10
    mode = np.arange(-3*grid.nw, 3*grid.nw+1)
    phase = np.exp(2j*np.pi*np.asarray(grid.tau)[:, None]*mode/grid.beta)
    integral = np.asarray(grid.tau_weights)@phase/grid.beta
    np.testing.assert_allclose(integral, mode == 0, atol=4e-13, rtol=0)


def test_gauss_dressed_complex_bubble_and_sigma_match_literal_lehmann_at_equal_ntau():
    module = importlib.import_module('gradscf.gw.pbc.matsubara')
    model = _model(scale=.002)
    mu, beta = .03, 80.
    weights = np.array([.7, .3])
    offsets = np.array([.2, -.2*.7/.3])
    poles = np.asarray(model.energy)[..., None]+offsets
    occupation = _fermi(beta*(poles-mu))
    density = jax.vmap(jnp.diag)(jnp.asarray(np.sum(weights*occupation, axis=-1)))
    moment = jnp.tile(jnp.eye(2)[None], (2, 1, 1))*float(np.sum(weights*offsets**2))
    errors, pipeline_pi = [], []
    for quadrature in ('uniform', 'gauss'):
        grid = _grid(nw=16, beta=beta, quadrature=quadrature)
        shifted = poles-mu
        exponent = (-np.asarray(grid.tau)[None, :, None, None]*shifted[:, None]
                    -np.logaddexp(0., -beta*shifted[:, None]))
        diagonal_tau = -np.sum(weights*np.exp(exponent), axis=-1)
        gtau = jax.vmap(jax.vmap(jnp.diag))(jnp.asarray(diagonal_tau))
        pi_tau = module._bubble(gtau, model.factors, model.table)
        sigma_tau = module._sigma_tau(gtau, model.factors, model.inverse, pi_tau)
        pi_exact, sigma_exact = _dressed_bath_oracles(model, poles, weights, mu, grid)
        pi = np.einsum('wt,t,tPQ->wPQ',
            np.exp(1j*np.asarray(grid.boson)[:, None]*np.asarray(grid.tau)[None]),
            grid.tau_weights, pi_tau[0])
        sigma = np.einsum('tw,t,ktab->kwab',
            np.exp(1j*np.asarray(grid.tau)[:, None]*np.asarray(grid.fermion)[None]),
            grid.tau_weights, sigma_tau)
        errors.append((np.linalg.norm(pi-pi_exact[0])/np.linalg.norm(pi_exact[0]),
                       np.linalg.norm(sigma-sigma_exact)/np.linalg.norm(sigma_exact)))
        diagonal = np.sum(weights/(1j*np.asarray(grid.fermion)[None, :, None, None]
                                  -shifted[:, None]), axis=-1)
        green = jax.vmap(jax.vmap(jnp.diag))(jnp.asarray(diagonal))
        step = module.gw_matsubara_step(green, model.reference_fock, mu, model, grid,
                                       density=density, sigma_moment=moment)
        pipeline_pi.append(np.linalg.norm(np.asarray(step['polarizability_inu'][0])-pi_exact[0])
                           /np.linalg.norm(pi_exact[0]))
    print('Equal Nt=128 dressed-bath relative Pi/Sigma direct quadrature errors:', errors,
          'full GW Pi errors:', pipeline_pi)
    assert np.min(errors[0]) > 1e-4
    np.testing.assert_allclose(errors[1], 0., atol=3e-12, rtol=0)
    assert pipeline_pi[1] < 2e-11
    assert pipeline_pi[1] < pipeline_pi[0]/1000
