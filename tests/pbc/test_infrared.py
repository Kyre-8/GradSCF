"""Analytic polar-oscillator checks, independent of an electronic calculation."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import physical_constants
from scipy.integrate import quad


AMU = (physical_constants['atomic mass constant'][0]
       / physical_constants['electron mass'][0])


def model():
    masses = np.array([7.0, 1.0])
    born = np.array([2.0 * np.eye(3), -2.0 * np.eye(3)])
    dielectric = 3.2 * np.eye(3)
    volume, spring = 120.0, 0.2
    optical = np.vstack((np.sqrt(masses[1] / masses.sum()) * np.eye(3),
                         -np.sqrt(masses[0] / masses.sum()) * np.eye(3)))
    translations = np.vstack((np.sqrt(masses[0] / masses.sum()) * np.eye(3),
                              np.sqrt(masses[1] / masses.sum()) * np.eye(3)))
    reduced_mass = np.prod(masses) / masses.sum() * AMU
    omega = np.sqrt(spring / reduced_mass)
    return masses, born, dielectric, volume, spring, optical, translations, omega


def test_longitudinal_transverse_splitting_and_lst_relation():
    api = importlib.import_module('gradscf.pbc.vibrational')
    masses, born, eps, volume, spring, modes, translations, omega = model()
    phi = spring * np.block([[np.eye(3), -np.eye(3)], [-np.eye(3), np.eye(3)]])
    correction = api.nonanalytic_force_constants([1., 0., 0.], born, eps, volume)
    mass_scale = np.repeat(np.sqrt(masses * AMU), 3)
    dynamical = (phi + correction) / mass_scale[:, None] / mass_scale[None, :]
    frequencies_squared = np.linalg.eigvalsh(dynamical)
    strength = 4.0 / (np.prod(masses) / masses.sum() * AMU)
    longitudinal = omega**2 + 4 * np.pi * strength / (volume * eps[0, 0])
    np.testing.assert_allclose(frequencies_squared,
                              [0., 0., 0., omega**2, omega**2, longitudinal], atol=1e-18)
    np.testing.assert_allclose(dynamical @ translations, 0., atol=1e-19)
    charges = api.mode_effective_charges(born, modes, masses)
    np.testing.assert_allclose(charges, np.sqrt(strength) * np.eye(3), atol=1e-17)
    static = api.ionic_dielectric(np.repeat(omega, 3), charges, [0.],
                                 volume=volume, broadening=0.)[0]
    np.testing.assert_allclose((eps[0, 0] + static[0, 0]) / eps[0, 0],
                              longitudinal / omega**2, atol=1e-14)
    for direction in ([3., 0., 0.], [-2., 0., 0.]):
        np.testing.assert_allclose(api.nonanalytic_force_constants(direction, born, eps, volume),
                                   correction, atol=1e-16)


def test_tensor_convention_rotation_and_raw_charge_defect():
    api = importlib.import_module('gradscf.pbc.vibrational')
    tensor = np.array([[1.1, .2, -.1], [.4, 1.8, .3], [.2, -.3, .9]])
    born = np.array([tensor, -tensor])
    eps = np.array([[2., .1, .2], [.1, 3., .3], [.2, .3, 4.]])
    q = np.array([.2, -.4, .7])
    phi = api.nonanalytic_force_constants(q, born, eps, 100.)
    # An independent displacement-energy Hessian fixes both tensor index order
    # and the factor 4*pi: E_macro = 2*pi*(q.P_cell)**2/(V*q.eps.q).
    displacement = np.arange(6).reshape(2, 3) / 10
    dipole = sum(born[a] @ displacement[a] for a in range(2))
    energy = 2 * np.pi * (q @ dipole)**2 / (100 * q @ eps @ q)
    np.testing.assert_allclose(.5 * displacement.ravel() @ phi @ displacement.ravel(), energy)
    rotation = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))[0]
    rotated_born = np.einsum('ij,ajk,lk->ail', rotation, born, rotation)
    rotated = api.nonanalytic_force_constants(rotation @ q, rotated_born,
                                              rotation @ eps @ rotation.T, 100.)
    transform = np.kron(np.eye(2), rotation)
    np.testing.assert_allclose(rotated, transform @ phi @ transform.T, atol=1e-16)
    bad_born = born.copy()
    bad_born[1, 0, 0] += .01
    raw = api.nonanalytic_force_constants(q, bad_born, eps, 100.)
    assert np.linalg.norm(raw @ np.tile(np.eye(3), (2, 1))) > 1e-5


def test_complete_mode_weight_and_degenerate_complex_gauge():
    api = importlib.import_module('gradscf.pbc.vibrational')
    masses, born, _, volume, _, modes, translations, omega = model()
    basis = np.column_stack((translations, modes))
    charges = api.mode_effective_charges(born, basis, masses)
    expected = sum(z @ z.T / (mass * AMU) for z, mass in zip(born, masses))
    np.testing.assert_allclose(charges @ charges.conj().T, expected, atol=1e-18)
    roundoff = 8 * np.finfo(float).eps * np.max(np.abs(charges))
    np.testing.assert_allclose(charges[:, :3], 0., atol=roundoff)
    random = np.random.default_rng(7)
    unitary = np.linalg.qr(random.normal(size=(3, 3)) + 1j * random.normal(size=(3, 3)))[0]
    rotated_charges = api.mode_effective_charges(born, modes @ unitary, masses)
    grid = np.linspace(0., 3 * omega, 101)
    first = api.ionic_dielectric(np.repeat(omega, 3), charges[:, 3:], grid,
                                volume=volume, broadening=omega / 10)
    second = api.ionic_dielectric(np.repeat(omega, 3), rotated_charges, grid,
                                 volume=volume, broadening=omega / 10)
    np.testing.assert_allclose(first, second, atol=1e-13)


def test_retarded_response_cartesian_resolvent_and_oscillator_sum():
    api = importlib.import_module('gradscf.pbc.vibrational')
    masses, born, _, volume, spring, modes, _, omega = model()
    charges = api.mode_effective_charges(born, modes, masses)
    grid = np.linspace(0., 3 * omega, 131)
    width = omega / 8
    response = np.asarray(api.ionic_dielectric(np.repeat(omega, 3), charges, grid,
                                              volume=volume, broadening=width))
    phi = spring * np.block([[np.eye(3), -np.eye(3)], [-np.eye(3), np.eye(3)]])
    mass_scale = np.repeat(np.sqrt(masses * AMU), 3)
    dynamical = phi / mass_scale[:, None] / mass_scale[None, :]
    cartesian_charge = np.concatenate(born, axis=1) / mass_scale
    reference = np.array([4 * np.pi / volume * cartesian_charge @
                          np.linalg.solve(dynamical - (energy + 1j * width)**2 * np.eye(6),
                                          cartesian_charge.T) for energy in grid])
    np.testing.assert_allclose(response, reference, atol=2e-13)
    assert np.min(response.imag.diagonal(axis1=1, axis2=2)) >= -1e-15
    # Analytic rational response, integrated independently to infinite energy.
    weight = float(charges[0] @ charges[0].conj())
    integral = quad(lambda x: x * np.imag(4 * np.pi * weight / volume /
                                        (omega**2 - (x + 1j * width)**2)),
                    0., np.inf, epsabs=1e-12)[0]
    np.testing.assert_allclose(integral, 2 * np.pi**2 * weight / volume, rtol=2e-10)


def test_response_jvp_vjp_and_multistep_differences():
    api = importlib.import_module('gradscf.pbc.vibrational')
    masses, born, eps, volume, _, modes, _, omega = model()

    @jax.jit
    def observable(parameters):
        z = jnp.asarray(born) * parameters[0]
        epsilon = jnp.asarray(eps) + parameters[1] * jnp.eye(3)
        cell_volume = volume * jnp.exp(parameters[2])
        q = jnp.array([1., parameters[3], .3])
        phi = api.nonanalytic_force_constants(q, z, epsilon, cell_volume)
        charge = api.mode_effective_charges(z, modes, jnp.asarray(masses) * parameters[4])
        response = api.ionic_dielectric(jnp.full(3, omega * parameters[5]), charge,
                                        jnp.array([0., omega * .8]), volume=cell_volume,
                                        broadening=omega / 10)
        return jnp.concatenate((phi.ravel(), response.real.ravel(), response.imag.ravel()))

    point = jnp.array([1., .1, 0., .2, 1., 1.])
    direction = jnp.array([.1, -.2, .1, .3, -.1, .1])
    value, tangent = jax.jvp(observable, (point,), (direction,))
    weight = jnp.linspace(.1, .5, value.size)
    _, pullback = jax.vjp(observable, point)
    np.testing.assert_allclose(weight @ tangent, pullback(weight)[0] @ direction, rtol=2e-13)
    errors = []
    for step in (1e-3, 3e-4, 1e-4):
        finite = (observable(point + step * direction) - observable(point - step * direction)) / (2 * step)
        errors.append(float(jnp.max(jnp.abs(finite - tangent))))
    assert errors[-1] < 1e-7
    assert errors[-1] < errors[0] / 20


@pytest.mark.parametrize('which', ['zero_direction', 'indefinite_epsilon', 'asymmetric_epsilon',
                                  'negative_volume', 'zero_mass', 'unstable_mode'])
@pytest.mark.parametrize('transform', ['eager', 'jit', 'jvp', 'vjp'])
def test_invalid_physical_inputs_are_not_clipped(which, transform):
    api = importlib.import_module('gradscf.pbc.vibrational')
    masses, born, eps, volume, _, modes, _, omega = model()
    q = np.array([1., 0., 0.])
    if which == 'zero_direction':
        q *= 0.
    if which == 'indefinite_epsilon':
        eps[1, 1] = -1.
    if which == 'asymmetric_epsilon':
        eps[0, 1] = .1
    if which == 'negative_volume':
        volume *= -1.
    if which == 'zero_mass':
        masses[0] = 0.
    if which == 'unstable_mode':
        omega *= -1.

    def evaluate(q, born, eps, volume, masses, omega):
        phi = api.nonanalytic_force_constants(q, born, eps, volume)
        charge = api.mode_effective_charges(born, modes, masses)
        dielectric = api.ionic_dielectric(jnp.repeat(omega, 3), charge, [0.],
                                          volume=volume, broadening=.001)
        return jnp.sum(phi) + jnp.sum(dielectric.real)

    def scalar(scale):
        return evaluate(q * scale, born, eps, volume, masses, omega)

    if transform == 'eager':
        run = scalar
    elif transform == 'jit':
        run = jax.jit(scalar)
    elif transform == 'jvp':
        run = jax.jit(lambda scale: jax.jvp(scalar, (scale,), (jnp.ones_like(scale),))[1])
    else:
        run = jax.jit(jax.grad(scalar))
    with pytest.raises(Exception, match='nonzero|positive|symmetric'):
        jax.block_until_ready(run(1.))
