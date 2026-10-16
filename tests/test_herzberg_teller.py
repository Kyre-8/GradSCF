"""Linear transition-dipole overlaps in the X11 harmonic coordinate convention."""
from math import factorial

import numpy as np
from numpy.polynomial.hermite import hermgauss
import pytest
from scipy.special import eval_hermite
from scipy.spatial.transform import Rotation


def coordinate_overlaps(initial, final, rotation, shift, cutoff):
    from gradscf_tools.franck_condon import ground_state_coordinate_overlaps
    return ground_state_coordinate_overlaps(initial, final, rotation, shift, max_quanta=cutoff)


def problem():
    angle = .33
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.array([.9, 1.3]), np.array([1.1, .7]), rotation, np.array([.15, -.21])


def test_constant_transition_dipole_reduces_to_all_three_component_fc():
    from gradscf_tools.franck_condon import ground_state_overlaps
    initial, final, rotation, shift = problem()
    states, overlap, coordinate = coordinate_overlaps(initial, final, rotation, shift, 6)
    expected_states, expected = ground_state_overlaps(initial, final, rotation, shift, max_quanta=6)
    dipole = np.array([.2, -.3, .4])
    moments = overlap[:, None]*dipole+coordinate@np.zeros((2, 3))
    np.testing.assert_array_equal(states, expected_states)
    np.testing.assert_allclose(np.sum(moments**2, axis=1), np.dot(dipole, dipole)*expected**2, atol=2e-14, rtol=0)


def test_identical_surface_coordinate_operator_has_one_quantum_selection_rule():
    frequencies = np.array([.7, 1.2])
    states, overlap, coordinate = coordinate_overlaps(frequencies, frequencies, np.eye(2), np.zeros(2), 4)
    expected = np.zeros_like(coordinate)
    for mode, target in enumerate(np.eye(2, dtype=int)):
        selected = np.flatnonzero(np.all(states == target, axis=1))[0]
        expected[selected, mode] = 1/np.sqrt(2*frequencies[mode])
    np.testing.assert_allclose(coordinate, expected, atol=2e-14, rtol=0)
    np.testing.assert_allclose(overlap[0], 1., atol=2e-14, rtol=0)


def test_linear_dipole_matches_independent_two_dimensional_wavefunction_quadrature():
    initial, final, rotation, shift = problem()
    states, overlap, coordinate = coordinate_overlaps(initial, final, rotation, shift, 4)
    dipole = np.array([.2, -.3, .4])
    derivatives = np.array([[.11, -.08], [.07, .13], [-.05, .04]])
    actual = overlap[:, None]*dipole+coordinate@derivatives.T
    nodes, weights = hermgauss(64)
    xx = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 2)
    q = xx/np.sqrt(initial)
    final_x = (q@rotation.T+shift)*np.sqrt(final)
    initial_ground = np.prod(initial/np.pi)**.25*np.exp(-.5*np.sum(xx**2, axis=1))
    final_ground = np.prod(final/np.pi)**.25*np.exp(-.5*np.sum(final_x**2, axis=1))
    common = (np.outer(weights, weights).ravel()*np.exp(np.sum(xx**2, axis=1))
              *initial_ground*final_ground/np.sqrt(np.prod(initial)))
    local_dipole = dipole+q@derivatives.T
    expected_coordinate, expected_moments = [], []
    for state in states:
        polynomial = np.ones(len(q))
        for mode, number in enumerate(state):
            polynomial *= eval_hermite(number, final_x[:, mode])/np.sqrt(2.**number*factorial(int(number)))
        integrand = common*polynomial
        expected_coordinate.append(integrand@q)
        expected_moments.append(integrand@local_dipole)
    np.testing.assert_allclose(coordinate, expected_coordinate, atol=3e-13, rtol=2e-12)
    np.testing.assert_allclose(actual, expected_moments, atol=3e-13, rtol=2e-12)


def test_coordinate_and_linear_dipole_closure_converge_without_rescaling():
    initial, final, rotation, shift = problem()
    dipole = np.array([.2, -.3, .4])
    derivatives = np.array([[.11, -.08], [.07, .13], [-.05, .04]])
    target = np.dot(dipole, dipole)+np.sum(derivatives**2/(2*initial))
    errors = []
    for cutoff in (10, 18, 26):
        _, overlap, coordinate = coordinate_overlaps(initial, final, rotation, shift, cutoff)
        moments = overlap[:, None]*dipole+coordinate@derivatives.T
        errors.append(abs(float(np.sum(moments**2))-target))
    np.testing.assert_allclose(overlap@coordinate, 0., atol=2e-10, rtol=0)
    np.testing.assert_allclose(coordinate.T@coordinate, np.diag(1/(2*initial)), atol=2e-10, rtol=0)
    assert errors[-1] < 2e-10 and errors[-1] < errors[0]*1e-2


@pytest.mark.parametrize("operation", ["electronic_phase", "ground_mode_gauge", "spatial_rotation"])
def test_combined_transition_strengths_respect_phase_and_coordinate_gauges(operation):
    initial, final, rotation, shift = problem()
    dipole = np.array([.2, -.3, .4])
    derivatives = np.array([[.11, -.08], [.07, .13], [-.05, .04]])
    states, overlap, coordinate = coordinate_overlaps(initial, final, rotation, shift, 4)
    expected = overlap[:, None]*dipole+coordinate@derivatives.T
    if operation == "electronic_phase":
        dipole, derivatives = -dipole, -derivatives
    elif operation == "ground_mode_gauge":
        permutation, signs = np.array([1, 0]), np.array([-1., 1.])
        initial = initial[permutation]
        rotation = rotation[:, permutation]*signs
        derivatives = derivatives[:, permutation]*signs
    else:
        cartesian = Rotation.from_rotvec([.2, -.3, .4]).as_matrix()
        dipole, derivatives = cartesian.T@dipole, cartesian.T@derivatives
    changed_states, overlap, coordinate = coordinate_overlaps(initial, final, rotation, shift, 4)
    actual = overlap[:, None]*dipole+coordinate@derivatives.T
    np.testing.assert_array_equal(states, changed_states)
    np.testing.assert_allclose(np.sum(actual**2, axis=1), np.sum(expected**2, axis=1), atol=2e-14, rtol=0)


def test_signed_interference_is_separate_from_nonnegative_total_strength():
    _, overlap, coordinate = coordinate_overlaps([1.], [1.], np.eye(1), [.7], 6)
    fc = overlap[:, None]*np.array([.4, 0., 0.])
    ht = coordinate@np.array([[.2, 0., 0.]])
    interference = 2*np.sum(fc*ht, axis=1)
    total = np.sum((fc+ht)**2, axis=1)
    assert interference[0] < 0 and np.all(total >= 0)
    np.testing.assert_allclose(total, np.sum(fc**2+ht**2, axis=1)+interference, atol=2e-14, rtol=0)
