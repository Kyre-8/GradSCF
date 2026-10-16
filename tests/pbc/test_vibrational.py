"""Analytic harmonic-spectrum checks; no electronic or EELS calculation."""

import importlib
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import physical_constants


def _api():
    name = "gradscf.pbc.vibrational"
    assert importlib.util.find_spec(name) is not None, "missing vibrational spectrum API"
    return importlib.import_module(name)


KB = physical_constants["Boltzmann constant in eV/K"][0] / physical_constants["Hartree energy in eV"][0]
AMU = physical_constants["atomic mass constant"][0] / physical_constants["electron mass"][0]


def test_line_areas_and_finite_temperature_detailed_balance():
    api = _api()
    energy, strength, temperature = 0.002, 1.7, 800.0
    grid = jnp.linspace(-0.01, 0.01, 10001)
    spectrum = api.one_phonon_spectrum([energy], [strength], grid,
                                       temperature=temperature, fwhm=0.0003)
    loss = np.trapezoid(spectrum.loss, grid)
    gain = np.trapezoid(spectrum.gain, grid)
    occupation = 1 / np.expm1(energy / (KB * temperature))
    np.testing.assert_allclose([loss, gain], strength * np.array([occupation + 1, occupation]), rtol=2e-12)
    np.testing.assert_allclose(gain / loss, np.exp(-energy / (KB * temperature)), rtol=2e-12)
    np.testing.assert_allclose(spectrum.total, spectrum.loss + spectrum.gain)
    assert float(grid[jnp.argmax(spectrum.loss)]) > 0
    assert float(grid[jnp.argmax(spectrum.gain)]) < 0


@pytest.mark.parametrize("temperature", [0.0, 1e-6])
def test_low_temperature_has_only_normalized_loss(temperature):
    api = _api()
    grid = jnp.linspace(-0.01, 0.01, 10001)
    spectrum = api.one_phonon_spectrum([0.002, 0.004], [2., 3.], grid,
                                       temperature=temperature, fwhm=0.0002)
    np.testing.assert_array_equal(spectrum.gain, 0.)
    np.testing.assert_allclose(np.trapezoid(spectrum.loss, grid), 5., rtol=2e-12)
    assert np.isfinite(spectrum.total).all()


def test_gaussian_fwhm():
    api = _api()
    energy, width = 0.002, 0.0004
    grid = jnp.array([energy - width / 2, energy, energy + width / 2])
    result = api.one_phonon_spectrum([energy], [1.], grid, temperature=0., fwhm=width)
    np.testing.assert_allclose(result.loss[jnp.array([0, 2])] / result.loss[1], 0.5, rtol=1e-12)


def test_jit_and_derivatives_match_centered_difference():
    api = _api()
    grid = jnp.linspace(-0.006, 0.006, 501)

    def objective(parameters):
        energy, strength, temperature, width = parameters
        result = api.one_phonon_spectrum(jnp.array([energy]), jnp.array([strength]),
                                         grid, temperature=temperature, fwhm=width)
        return jnp.sum(result.total * jnp.exp(70 * grid)) * (grid[1] - grid[0])

    parameters = jnp.array([0.002, 1.3, 600., 0.0005])
    gradient = jax.jit(jax.grad(objective))(parameters)
    steps = np.array([1e-7, 1e-5, 1e-2, 1e-7])
    finite_difference = []
    for index, step in enumerate(steps):
        offset = jnp.zeros(4).at[index].set(step)
        finite_difference.append((objective(parameters + offset) - objective(parameters - offset)) / (2 * step))
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(jax.jit(objective)(parameters), objective(parameters), rtol=1e-13)


@pytest.mark.parametrize("replacement,match", [
    ({"energies": [0.]}, "positive"),
    ({"energies": [-0.002]}, "positive"),
    ({"energies": [0.002j]}, "real"),
    ({"energies": [np.nan]}, "finite"),
    ({"strengths": [-1.]}, "nonnegative"),
    ({"strengths": [1j]}, "real"),
    ({"strengths": [1., 2.]}, "shape"),
    ({"energy_grid": [np.inf]}, "finite"),
    ({"temperature": -1.}, "nonnegative"),
    ({"temperature": np.inf}, "finite"),
    ({"temperature": [300.]}, "scalar"),
    ({"fwhm": 0.}, "positive"),
])
def test_invalid_spectrum_input_is_rejected(replacement, match):
    api = _api()
    arguments = dict(energies=[0.002], strengths=[1.], energy_grid=[0.], temperature=300., fwhm=0.0002)
    arguments.update(replacement)
    with pytest.raises(ValueError, match=match):
        api.one_phonon_spectrum(**arguments)


def test_invalid_traced_mode_is_rejected_without_clipping():
    api = _api()
    evaluate = jax.jit(lambda energy: api.one_phonon_spectrum(
        energy, jnp.ones(1), jnp.zeros(1), temperature=300., fwhm=0.0002).total)
    with pytest.raises(Exception, match="strictly positive"):
        evaluate(jnp.array([-0.001])).block_until_ready()


def test_coherent_weights_mass_units_and_transverse_selection():
    api = _api()
    modes = jnp.eye(3).reshape(1, 3, 3)
    weights = api.mode_scattering_weights([2., 0., 0.], [[0., 0., 0.]], modes,
                                          [12.], form_factors=[3.])
    np.testing.assert_allclose(weights, [36 / (12 * AMU), 0., 0.], rtol=1e-12)


def test_atom_phase_interference_and_flat_mode_layout():
    api = _api()
    modes = jnp.array([[1., 0., 0., 1., 0., 0.]]).T / jnp.sqrt(2.)
    coordinates = jnp.array([[0., 0., 0.], [jnp.pi, 0., 0.]])
    weights = api.mode_scattering_weights([1., 0., 0.], coordinates, modes,
                                          [12., 12.], form_factors=[1., 1.])
    np.testing.assert_allclose(weights, 0., atol=1e-30)
    constructive = api.mode_scattering_weights([2., 0., 0.], coordinates, modes.reshape(2, 3, 1),
                                               [12., 12.], form_factors=[1., 1.])
    np.testing.assert_allclose(constructive, [8 / (12 * AMU)], rtol=1e-12)


def test_complex_lattice_translation_mode_uses_negative_atom_phase():
    api = _api()
    # e=(1,i)/sqrt(2) and Q.tau=(0,pi/2) add constructively with exp(-iQ.tau).
    modes = jnp.array([[1., 0., 0., 1j, 0., 0.]]).T / jnp.sqrt(2.)
    result = api.mode_scattering_weights([1., 0., 0.],
        [[0., 0., 0.], [jnp.pi / 2, 0., 0.]], modes, [12., 12.], form_factors=[1., 1.])
    np.testing.assert_allclose(result, [2 / (12 * AMU)], rtol=1e-12)


def test_degenerate_basis_and_mode_phase_invariance():
    api = _api()
    coordinates = jnp.array([[0., 0., 0.], [0.3, 0.8, 0.2]])
    modes = jnp.eye(6, dtype=jnp.complex128)[:, :2]
    unitary = jnp.array([[1., 1j], [1j, 1.]]) / jnp.sqrt(2.)

    def weights(vectors, positions=coordinates):
        return api.mode_scattering_weights([0.5, 0.3, 0.1], positions, vectors,
                                            [12., 13.], form_factors=[1., 0.7])

    reference = weights(modes)
    np.testing.assert_allclose(weights(modes * jnp.exp(0.7j)), reference, rtol=1e-12)
    np.testing.assert_allclose(weights(modes, coordinates + 0.9), reference, rtol=1e-12)
    rotated = weights(modes @ unitary)
    np.testing.assert_allclose(jnp.sum(rotated), jnp.sum(reference), rtol=1e-12)
    grid = jnp.linspace(-0.005, 0.005, 501)
    first = api.one_phonon_spectrum([0.002, 0.002], reference, grid, temperature=700., fwhm=0.0003)
    second = api.one_phonon_spectrum([0.002, 0.002], rotated, grid, temperature=700., fwhm=0.0003)
    np.testing.assert_allclose(first.total, second.total, rtol=1e-12, atol=1e-15)


def test_coherent_weight_gradient_and_jit():
    api = _api()
    modes = jnp.array([[1., 0., 0., 0.5, 0., 0.]]).T

    def objective(q):
        return api.mode_scattering_weights(jnp.array([q, 0., 0.]),
            [[0., 0., 0.], [0.8, 0., 0.]], modes, [12., 13.], form_factors=[1., 2.]).sum()

    exact = jax.jit(jax.grad(objective))(0.7)
    difference = (objective(0.700001) - objective(0.699999)) / 2e-6
    np.testing.assert_allclose(exact, difference, rtol=1e-8, atol=1e-12)


def test_weights_reject_nonpositive_masses_and_missing_atom_factors():
    api = _api()
    with pytest.raises(ValueError, match="positive"):
        api.mode_scattering_weights([1., 0., 0.], [[0., 0., 0.]], jnp.eye(3), [0.], form_factors=[1.])
    with pytest.raises(ValueError, match="shape"):
        api.mode_scattering_weights([1., 0., 0.], [[0., 0., 0.]], jnp.eye(3), [12.], form_factors=[])
