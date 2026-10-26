"""Real-axis Fan poles, linewidth conventions, spectral matrices, and AD."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _case():
    from gradscf.gw.ep_coupling.types import PhononModel
    energies = jnp.array([-0.3, 0.45])
    model = PhononModel(jnp.array([0.08, 0.2]), jnp.array([
        [[0.06, 0.03 + 0.02j], [0.03 - 0.02j, -0.04]],
        [[0.02, -0.01j], [0.01j, 0.05]],
    ]))
    return energies, model


def test_retarded_complex_vertices_independent_poles_and_causality():
    from gradscf.gw.ep_coupling.spectral import fan_retarded
    energies, model = _case()
    frequencies = np.linspace(-0.9, 0.9, 47)
    mu, beta, eta = 0.07, 17., 0.025
    reference = np.zeros((47, 2, 2), complex)
    for l, omega in enumerate(np.asarray(model.energies)):
        bose = 1 / np.expm1(beta * omega)
        for a, energy in enumerate(np.asarray(energies)):
            fermi = 1 / (1 + np.exp(beta * (energy - mu)))
            poles = ((bose + fermi) / (frequencies + 1j * eta - energy + mu + omega)
                     + (bose + 1 - fermi) / (frequencies + 1j * eta - energy + mu - omega))
            vertex = np.asarray(model.couplings)[l, :, a]
            reference += poles[:, None, None] * np.outer(vertex, vertex.conj())
    sigma = np.asarray(fan_retarded(energies, model, frequencies, mu=mu, beta=beta, eta=eta))
    np.testing.assert_allclose(sigma, reference, rtol=1e-12, atol=1e-13)
    loss = -(sigma - sigma.swapaxes(-1, -2).conj()) / (2j)
    assert np.linalg.eigvalsh(loss).min() >= -1e-13


@pytest.mark.parametrize('smearing', ['gaussian', 'lorentzian'])
def test_rectangular_linewidth_independent_on_shell_formula(smearing):
    # ElectronPhonon.jl a0ecee01b7413af5566134d94cad5b15431d6565,
    # src/selfenergy_electron.jl: imsigma is positive halfwidth, with
    # gaussian(x)=exp(-x*x)/sqrt(pi), not a normal sigma-width Gaussian.
    from gradscf.gw.ep_coupling.spectral import fan_linewidth
    external, internal = np.array([-0.15, 0.23]), np.array([-0.3, 0.04, 0.5])
    omega = np.array([0.08, 0.19])
    g = np.arange(12).reshape(2, 2, 3) * (0.003 + 0.002j)
    mu, beta, eta = 0.03, 13., 0.1
    reference = np.zeros(2)
    for i, e in enumerate(external):
        for a, ei in enumerate(internal):
            f = 1 / (1 + np.exp(beta * (ei - mu)))
            for l, w in enumerate(omega):
                n = 1 / np.expm1(beta * w)
                def delta(x):
                    if smearing == 'gaussian':
                        return np.exp(-(x / eta)**2) / (np.sqrt(np.pi) * eta)
                    return eta / (np.pi * (x*x + eta*eta))
                reference[i] += 2 * np.pi * abs(g[l, i, a])**2 * (
                    (n + f) * delta(e - ei + w) + (n + 1 - f) * delta(e - ei - w))
    actual = fan_linewidth(external, internal, omega, g, mu=mu, beta=beta,
                           eta=eta, smearing=smearing)
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-14)


def test_molecular_linewidth_matches_on_shell_retarded_diagonal():
    from gradscf.gw.ep_coupling.spectral import fan_retarded, electron_linewidth
    energies, model = _case()
    sigma = fan_retarded(energies, model, energies - 0.1, mu=0.1, beta=20., eta=.03)
    actual = electron_linewidth(energies, model, mu=.1, beta=20., eta=.03)
    np.testing.assert_allclose(actual, -2 * np.imag(np.asarray(sigma)[np.arange(2), np.arange(2), np.arange(2)]))


def test_spectral_matrix_dyson_psd_and_sumrule():
    from gradscf.gw.ep_coupling.spectral import fan_retarded, spectral_function
    energies, model = _case()
    frequencies = jnp.linspace(-12., 12., 24001)
    sigma = fan_retarded(energies, model, frequencies, mu=.07, beta=17., eta=.04)
    fock = jnp.array([[-.3, .02j], [-.02j, .45]])
    actual = np.asarray(spectral_function(fock, sigma, frequencies, mu=.07, eta=.04))
    np.testing.assert_allclose(actual, actual.swapaxes(-1, -2).conj(), atol=1e-13)
    assert np.linalg.eigvalsh(actual).min() >= -1e-13
    green = np.linalg.inv((np.asarray(frequencies) + .07 + .04j)[:, None, None]
                          * np.eye(2) - np.asarray(fock) - np.asarray(sigma))
    np.testing.assert_allclose(actual, -(green - green.swapaxes(-1, -2).conj()) / (2j * np.pi), atol=1e-13)
    # Finite window misses the O(eta / window) Lorentzian tail.
    integral = np.sum((actual[1:] + actual[:-1]) * np.diff(frequencies)[:, None, None] / 2, axis=0)
    np.testing.assert_allclose(integral, np.eye(2), atol=.0022)


def test_zero_and_weak_coupling_limits():
    from gradscf.gw.ep_coupling.spectral import fan_retarded, spectral_function
    from gradscf.gw.ep_coupling.types import PhononModel
    energies, model = _case()
    w = jnp.linspace(-1., 1., 101)
    base = fan_retarded(energies, model, w, beta=20.)
    small = PhononModel(model.energies, model.couplings * .001)
    np.testing.assert_allclose(fan_retarded(energies, small, w, beta=20.), base * 1e-6, rtol=1e-12, atol=1e-16)
    zero = fan_retarded(energies, PhononModel(model.energies, model.couplings * 0), w, beta=20.)
    np.testing.assert_array_equal(zero, 0)
    spectrum = spectral_function(jnp.diag(energies), zero, w, eta=.03)
    expected = .03 / (np.pi * ((np.asarray(w)[:, None] - np.asarray(energies))**2 + .03**2))
    np.testing.assert_allclose(np.diagonal(spectrum, axis1=-2, axis2=-1), expected, rtol=1e-12)


def test_retarded_and_spectral_ad_against_finite_difference():
    from gradscf.gw.ep_coupling.spectral import fan_retarded, spectral_function
    from gradscf.gw.ep_coupling.types import PhononModel
    energies, model = _case()
    def loss(params):
        varied = PhononModel(model.energies * params[1], model.couplings * params[0])
        w = jnp.array([-.2, .0, .3])
        sigma = fan_retarded(energies, varied, w, beta=20., eta=.03)
        return jnp.trace(spectral_function(jnp.diag(energies), sigma, w, eta=.02), axis1=-2, axis2=-1).real.sum()
    x, h = jnp.array([.7, 1.1]), 1e-5
    derivative = jax.jit(jax.grad(loss))(x)
    reference = [(loss(x + h * jnp.eye(2)[i]) - loss(x - h * jnp.eye(2)[i])) / (2*h) for i in range(2)]
    np.testing.assert_allclose(derivative, reference, rtol=2e-7, atol=1e-8)


@pytest.mark.parametrize('eta', [0., -.1, np.inf, np.nan])
def test_invalid_broadening_rejected(eta):
    from gradscf.gw.ep_coupling.spectral import fan_retarded
    energies, model = _case()
    with pytest.raises(ValueError, match='eta'):
        fan_retarded(energies, model, jnp.array([0.]), beta=20., eta=eta)
