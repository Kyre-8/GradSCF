"""Independent auxiliary-Hamiltonian oracles for numerical G-tail subtraction."""
import importlib
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _module():
    module = importlib.import_module('gradscf.gw.pbc.matsubara')
    assert 'sigma_moment' in inspect.signature(module.dyson_green_and_density).parameters, \
        'Missing matrix-pole density-tail reference capability'
    return module


def _case(module, nw=3, beta=20., mu=.04, endpoints=False):
    grid = module.matsubara_grid(nw=nw, beta=beta)
    fock = np.array([[[-.6, .13+.07j], [.13-.07j, .7]],
                     [[-.4, -.09+.11j], [-.09-.11j, .9]]])
    coupling = np.array([[[.31+.08j, .04-.09j], [.11+.07j, .23-.02j]],
                         [[.27-.04j, -.08+.06j], [.09+.03j, .35+.07j]]])
    bath = np.array([[[-.9, .17+.05j], [.17-.05j, 1.1]],
                     [[-1.2, -.12+.08j], [-.12-.08j, .8]]])
    moment = coupling@coupling.conj().swapaxes(-1, -2)
    sigma, green, density, time = [], [], [], []
    derivative_zero, derivative_beta, spectra, physical_frames = [], [], [], []
    for f, v, a in zip(fock, coupling, bath):
        h = np.block([[f, v], [v.conj().T, mu*np.eye(2)+a]])
        energy, coefficient = np.linalg.eigh(h)
        physical = coefficient[:2]
        small = np.exp(-abs(beta*(energy-mu)))
        occupation = np.where(energy >= mu, small/(1+small), 1/(1+small))
        density.append((physical*occupation)@physical.conj().T)
        derivative_zero.append((physical*((energy-mu)*(1-occupation)))@physical.conj().T)
        derivative_beta.append((physical*((energy-mu)*occupation))@physical.conj().T)
        spectra.append(energy-mu)
        physical_frames.append(physical)
        green.append(np.einsum('as,ws,bs->wab', physical,
            1/(1j*np.asarray(grid.fermion)[:, None]+mu-energy), physical.conj()))
        thermal_time = -np.exp(-np.asarray(grid.tau)[:, None]*(energy-mu)
                              -np.logaddexp(0., -beta*(energy-mu)))
        time.append(np.einsum('as,ts,bs->tab', physical, thermal_time, physical.conj()))
        sigma.append(np.array([v@np.linalg.solve(1j*w*np.eye(2)-a, v.conj().T)
                               for w in np.asarray(grid.fermion)]))
    result = (grid, fock, np.asarray(sigma), moment, np.asarray(green), np.asarray(density), np.asarray(time))
    return result+(np.asarray(derivative_zero), np.asarray(derivative_beta),
                   np.asarray(spectra), np.asarray(physical_frames)) if endpoints else result


@pytest.mark.parametrize('nw,beta', [(3, 20.), (8, 400.)])
def test_one_matrix_pole_density_and_time_match_exact_noncommuting_auxiliary_bath(nw, beta):
    module = _module()
    grid, fock, sigma, moment, expected_g, expected_d, expected_t = _case(module, nw=nw, beta=beta)
    green, density = module.dyson_green_and_density(jnp.asarray(fock), jnp.asarray(sigma),
        .04, grid, sigma_moment=jnp.asarray(moment))
    np.testing.assert_allclose(green, expected_g, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(density, expected_d, atol=2e-12, rtol=2e-12)
    # B=0 makes Pi/W/Sigma irrelevant: this checks only reconstruction of
    # the supplied dressed G, rather than a noninteracting Green function.
    from test_matsubara import _model
    step = module.gw_matsubara_step(green, jnp.asarray(fock), .04, _model(scale=0.), grid,
        sigma_moment=jnp.asarray(moment), density=density)
    np.testing.assert_allclose(step['green_tau'], expected_t, atol=3e-12, rtol=3e-12)
    _, baseline_density = module.dyson_green_and_density(jnp.asarray(fock), jnp.asarray(sigma), .04, grid)
    assert np.max(abs(np.asarray(baseline_density)-expected_d)) > 1e-5


def test_number_response_keeps_mu_dependence_of_auxiliary_bath_reference():
    module = _module()
    grid, fock, sigma, moment, _, _, _ = _case(module)
    _, derivative = jax.jvp(lambda mu: module.dyson_green_and_density(
        jnp.asarray(fock), jnp.asarray(sigma), mu, grid, sigma_moment=jnp.asarray(moment))[1],
        (jnp.array(.04),), (jnp.array(1.),))
    step = 1e-5
    exact_plus = _case(module, mu=.04+step)[5]
    exact_minus = _case(module, mu=.04-step)[5]
    np.testing.assert_allclose(derivative, (exact_plus-exact_minus)/(2*step), atol=3e-8, rtol=3e-8)


def test_negative_moment_is_not_clipped_and_zero_moment_retains_static_reference():
    module = _module()
    grid, fock, sigma, moment, _, _, _ = _case(module)
    _, invalid = module.dyson_green_and_density(jnp.asarray(fock), jnp.asarray(sigma),
        .04, grid, sigma_moment=-jnp.asarray(moment))
    assert not np.isfinite(invalid).all()
    zeros = jnp.zeros_like(jnp.asarray(sigma))
    first = module.dyson_green_and_density(jnp.asarray(fock), zeros, .04, grid)
    second = module.dyson_green_and_density(jnp.asarray(fock), zeros, .04, grid,
                                            sigma_moment=jnp.zeros_like(jnp.asarray(moment)))
    for left, right in zip(first, second):
        np.testing.assert_allclose(left, right, atol=2e-14, rtol=2e-14)
