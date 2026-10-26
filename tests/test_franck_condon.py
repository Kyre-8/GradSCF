"""Zero-temperature harmonic overlaps: limits and independent quadrature.

All coordinates are mass-weighted with hbar=1. The declared convention is
Q_final = rotation @ Q_initial + shift; rotation must be orthogonal here.
These gates do not choose a molecular Eckart/common-subspace approximation.
"""
from math import factorial

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.special import eval_hermite


def overlaps(initial, final, rotation, shift, cutoff):
    from gradscf_tools.franck_condon import ground_state_overlaps
    return ground_state_overlaps(initial, final, rotation, shift, max_quanta=cutoff)


def test_identical_oscillators_have_only_zero_zero_overlap():
    states, amplitudes = overlaps([.7, 1.2], [.7, 1.2], np.eye(2), np.zeros(2), 6)
    target = np.all(np.asarray(states) == 0, axis=1).astype(float)
    np.testing.assert_allclose(amplitudes, target, atol=2e-14, rtol=0)


def test_one_mode_displacement_matches_unrenormalized_poisson_weights():
    omega, shift = .8, 1.7
    states, amplitudes = overlaps([omega], [omega], np.eye(1), [shift], 4)
    n = np.asarray(states)[:, 0]
    huang_rhys = omega*shift**2/2
    expected = np.array([np.exp(-huang_rhys)*huang_rhys**i/factorial(int(i)) for i in n])
    np.testing.assert_allclose(np.asarray(amplitudes)**2, expected, atol=2e-14, rtol=2e-13)
    assert 0 < np.sum(expected) < .995  # A deliberately unresolved tail is retained.


def test_frequency_change_zero_shift_has_gaussian_zero_zero_and_even_parity():
    initial, final = .9, 1.7
    states, amplitudes = overlaps([initial], [final], np.eye(1), [0.], 12)
    n = np.asarray(states)[:, 0]
    expected = 2*np.sqrt(initial*final)/(initial+final)
    np.testing.assert_allclose(amplitudes[0]**2, expected, atol=2e-14, rtol=0)
    np.testing.assert_allclose(np.asarray(amplitudes)[n % 2 == 1], 0., atol=2e-14, rtol=0)


def test_two_mode_rotation_matches_direct_gauss_hermite_wavefunction_integrals():
    initial, final = np.array([.9, 1.3]), np.array([1.1, .7])
    angle = .33
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = np.array([.15, -.21])
    states, amplitudes = overlaps(initial, final, rotation, shift, 4)
    nodes, weights = hermgauss(64)
    xx = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 2)
    quadrature = np.outer(weights, weights).ravel()
    q = xx/np.sqrt(initial)
    q_final = q@rotation.T+shift
    x_final = q_final*np.sqrt(final)
    # Direct normalized harmonic wavefunctions, without a generating-function
    # recurrence. dq supplies the frequency-scaling Jacobian explicitly.
    initial_ground = np.prod(initial/np.pi)**.25*np.exp(-.5*np.sum(xx**2, axis=1))
    final_ground = np.prod(final/np.pi)**.25*np.exp(-.5*np.sum(x_final**2, axis=1))
    common = quadrature*np.exp(np.sum(xx**2, axis=1))*initial_ground*final_ground/np.sqrt(np.prod(initial))
    expected = []
    for state in states:
        polynomial = np.ones(len(q))
        for mode, number in enumerate(state):
            polynomial *= eval_hermite(number, x_final[:, mode])/np.sqrt(2.**number*factorial(int(number)))
        expected.append(np.sum(common*polynomial))
    np.testing.assert_allclose(amplitudes, expected, atol=3e-13, rtol=2e-12)


def test_reflected_mode_gauge_preserves_franck_condon_factors():
    initial, final = [.9, 1.3], [1.1, .7]
    angle = .33
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = np.array([.15, -.21])
    states, original = overlaps(initial, final, rotation, shift, 4)
    signs = np.diag([-1., 1.])
    changed_states, reflected = overlaps(initial, final, signs@rotation, signs@shift, 4)
    np.testing.assert_array_equal(states, changed_states)
    np.testing.assert_allclose(np.asarray(original)**2, np.asarray(reflected)**2, atol=2e-14, rtol=0)


def test_nonorthogonal_projected_mode_overlap_is_not_silently_used_as_rotation():
    with pytest.raises(ValueError, match="orthogonal"):
        overlaps([.9, 1.3], [1.1, .7], np.diag([1., .98]), [0., 0.], 3)


def test_zero_zero_displacement_sensitivity_matches_multistep_finite_differences():
    import jax
    import jax.numpy as jnp
    from gradscf_tools.franck_condon import zero_zero_probability
    angle = .33
    rotation = jnp.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    function = lambda scale: zero_zero_probability(
        jnp.array([.9, 1.3]), jnp.array([1.1, .7]), rotation, scale*jnp.array([.15, -.21]))
    value, derivative = jax.jit(jax.value_and_grad(function))(1.2)
    assert np.isfinite(value) and abs(float(derivative)) > 1e-4
    errors = []
    for step in (3e-3, 1e-3, 3e-4):
        finite = (function(1.2+step)-function(1.2-step))/(2*step)
        errors.append(abs(float(finite-derivative)))
    assert errors[-1] < 2e-9
    assert errors[-1] < errors[0]


def test_two_mode_factors_match_existing_quadratic_vibronic_hamiltonian():
    import jax.numpy as jnp
    from gradscf.bse import ep_coupling as ep
    from gradscf.solvers import EigenSolverConfig, solve_hermitian
    initial, final = np.array([.9, 1.3]), np.array([1.07, .73])
    angle = .33
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = np.array([.15, -.21])
    states, amplitudes = overlaps(initial, final, rotation, shift, 2)
    curvature = rotation.T@np.diag(final**2)@rotation
    linear = rotation.T@(final**2*shift)
    constant = .5*np.sum(final**2*shift**2)
    coordinate_scale = 1/np.sqrt(2*initial)
    couplings = linear*coordinate_scale
    quadratic = (curvature-np.diag(initial**2))*np.outer(coordinate_scale, coordinate_scale)
    model = ep.PhononModel(jnp.asarray(initial), jnp.asarray(couplings[:, None, None]),
                           jnp.asarray(quadratic[:, :, None, None]))
    target_energies = .7+.5*(final.sum()-initial.sum())+np.asarray(states)@final
    errors = []
    for cutoff in (14, 24):
        matrix, _ = ep.vibronic_hamiltonian(jnp.array([.7+constant]), model,
                                           jnp.array([[1., 0., 0.]]), max_quanta=cutoff)
        result = solve_hermitian(matrix, config=EigenSolverConfig(method="dense",
            nroots=matrix.shape[0], max_dense=matrix.shape[0], atol=1e-10))
        assert np.all(result.converged)
        energies = np.asarray(result.raw_values)
        selected = np.argmin(abs(energies[:, None]-target_energies[None, :]), axis=0)
        errors.append(float(np.max(abs(energies[selected]-target_energies))))
    assert errors[-1] < errors[0]*1e-2
    np.testing.assert_allclose(energies[selected], target_energies, atol=2e-8, rtol=0)
    np.testing.assert_allclose(np.asarray(result.raw_vectors)[0, selected]**2,
                               np.asarray(amplitudes)**2, atol=2e-8, rtol=0)
