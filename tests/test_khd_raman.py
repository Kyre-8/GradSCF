"""Signed vibronic scattering amplitudes, independently of electronic inputs."""
from math import factorial

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
import pytest
from scipy.special import eval_hermite
from scipy.spatial.transform import Rotation


def problem():
    angle = .33
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.array([.9, 1.3]), np.array([1.1, .7]), rotation, np.array([.15, -.21])


def vertices(cutoff=20):
    from gradscf_tools.khd_raman import linear_dipole_vertices
    wi, wf, rotation, shift = problem()
    mu = np.array([.2, -.3, .4])
    derivative = np.array([[.11, -.08], [.07, .13], [-.05, .04]])
    return linear_dipole_vertices(wi, wf, rotation, shift, mu, derivative, max_quanta=cutoff)


def test_quadratic_insertions_match_independent_wavefunction_quadrature():
    from gradscf_tools.franck_condon import ground_state_quadratic_coordinate_overlaps
    wi, wf, rotation, shift = problem()
    states, _, _, quadratic = ground_state_quadratic_coordinate_overlaps(wi, wf, rotation, shift, max_quanta=4)
    nodes, weights = hermgauss(64)
    xx = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 2)
    q = xx/np.sqrt(wi)
    final_x = (q@rotation.T+shift)*np.sqrt(wf)
    common = (np.outer(weights, weights).ravel()*np.exp(.5*np.sum(xx**2, axis=1)-.5*np.sum(final_x**2, axis=1))
              *np.prod(wi*wf)**.25/np.pi/np.sqrt(np.prod(wi)))
    expected = []
    for state in states:
        polynomial = np.ones(len(q))
        for mode, number in enumerate(state):
            polynomial *= eval_hermite(number, final_x[:, mode])/np.sqrt(2.**number*factorial(int(number)))
        expected.append(np.einsum("n,na,nb->ab", common*polynomial, q, q))
    np.testing.assert_allclose(quadratic, expected, atol=4e-13, rtol=2e-12)
    np.testing.assert_allclose(quadratic, quadratic.swapaxes(-1, -2), atol=4e-14, rtol=0)


def test_both_vertex_closures_without_renormalization():
    from gradscf_tools.khd_raman import vertex_closure
    wi, *_ = problem()
    mu = np.array([.2, -.3, .4])
    derivative = np.array([[.11, -.08], [.07, .13], [-.05, .04]])
    _, absorption, emission = vertices(26)
    expected_a, expected_b = vertex_closure(wi, mu, derivative)
    np.testing.assert_allclose(absorption.conj().T@absorption, expected_a, atol=3e-11, rtol=0)
    for mode in range(2):
        np.testing.assert_allclose(emission[:, mode].conj().T@emission[:, mode], expected_b[mode], atol=3e-11, rtol=0)


def test_identical_oscillator_exact_nonsymmetric_one_quantum_amplitude():
    from gradscf_tools.khd_raman import linear_dipole_vertices, khd_tensor
    w, gap, laser, gamma = .2, 1.3, .9, .03
    mu = np.array([.2, -.3, .4])
    derivative = np.array([.11, .07, -.05])
    states, a, b = linear_dipole_vertices([w], [w], [[1.]], [0.], mu, derivative[:, None], max_quanta=4)
    resonant, anti = khd_tensor(gap+states[:, 0]*w, a, b, [w], [laser], gamma)
    outgoing = laser-w
    expected_r = (np.outer(derivative, mu)/(gap-laser-1j*gamma)
                  +np.outer(mu, derivative)/(gap-outgoing-1j*gamma))/np.sqrt(2*w)
    expected_a = (np.outer(mu, derivative)/(gap+outgoing-1j*gamma)
                  +np.outer(derivative, mu)/(gap+laser-1j*gamma))/np.sqrt(2*w)
    np.testing.assert_allclose(resonant[0, 0], expected_r, atol=3e-15, rtol=0)
    np.testing.assert_allclose(anti[0, 0], expected_a, atol=3e-15, rtol=0)
    assert np.linalg.norm(expected_r+expected_a-(expected_r+expected_a).T) > 1e-3
    # Two constant or two odd dipoles cannot create one quantum on identical surfaces.
    for m, d in ((mu, np.zeros_like(derivative)), (np.zeros_like(mu), derivative)):
        states, a, b = linear_dipole_vertices([w], [w], [[1.]], [0.], m, d[:, None], max_quanta=4)
        terms = khd_tensor(gap+states[:, 0]*w, a, b, [w], [laser], gamma)
        np.testing.assert_allclose(sum(terms), 0., atol=3e-15, rtol=0)


@pytest.mark.parametrize("operation", ["electronic_phase", "mode_gauge", "cartesian_rotation"])
def test_khd_gauges_and_chunking(operation):
    from gradscf_tools.khd_raman import linear_dipole_vertices, khd_tensor, isotropic_strengths
    wi, wf, rotation, shift = problem()
    mu = np.array([.2, -.3, .4], dtype=complex)
    derivative = np.array([[.11, -.08], [.07, .13], [-.05, .04]], dtype=complex)
    states, a, b = linear_dipole_vertices(wi, wf, rotation, shift, mu, derivative, max_quanta=12)
    args = (4+states@wf, a, b, wi, np.array([2.1, 3.7]), .04)
    original = sum(khd_tensor(*args, state_chunk=17, photon_chunk=1))
    np.testing.assert_allclose(sum(khd_tensor(*args)), original, atol=5e-16, rtol=1e-13)
    expected = original
    if operation == "electronic_phase":
        mu, derivative = mu*np.exp(.43j), derivative*np.exp(.43j)
    elif operation == "mode_gauge":
        p, signs = np.array([1, 0]), np.array([-1., 1.])
        wi, rotation, derivative = wi[p], rotation[:, p]*signs, derivative[:, p]*signs
        expected = original[:, p]*signs[None, :, None, None]
    else:
        r = Rotation.from_rotvec([.3, -.4, .2]).as_matrix()
        mu, derivative = r@mu, r@derivative
        expected = r@original@r.T
    states, a, b = linear_dipole_vertices(wi, wf, rotation, shift, mu, derivative, max_quanta=12)
    actual = sum(khd_tensor(4+states@wf, a, b, wi, [2.1, 3.7], .04))
    np.testing.assert_allclose(actual, expected, atol=4e-15, rtol=2e-12)
    np.testing.assert_allclose(isotropic_strengths(actual), isotropic_strengths(expected), atol=3e-16, rtol=0)


def test_general_complex_tensor_rotational_average():
    from gradscf_tools.khd_raman import isotropic_strengths
    tensor = np.array([[.2+.4j, -.7+.1j, .3], [.5-.2j, .8-.3j, -.1+.7j], [.6j, .4, -.2+.1j]])
    parallel, perpendicular = isotropic_strengths(tensor)
    nodes, weights = leggauss(10)
    measured = np.zeros(2)
    angles = np.arange(16)*2*np.pi/16
    for z, weight in zip(nodes, weights):
        for alpha in angles:
            for gamma in angles:
                r = Rotation.from_euler("ZYZ", [alpha, np.arccos(z), gamma]).as_matrix()
                rotated = r@tensor@r.T
                measured += weight*np.abs(rotated[[2, 0], [2, 2]])**2/(2*16**2)
    np.testing.assert_allclose([parallel, perpendicular], measured, atol=8e-15, rtol=0)
    anti = tensor-tensor.T
    expected = [0., np.sum(abs(anti)**2)/6]
    np.testing.assert_allclose(isotropic_strengths(anti), expected, atol=8e-15, rtol=0)


def test_elastic_principal_value_and_invalid_inputs():
    from gradscf_tools.khd_raman import khd_tensor
    a = np.array([[.2, -.3, .4]])
    terms = khd_tensor([1.3], a, a[:, None, :], [0.], [.4], 1e-10)
    expected = np.outer(a[0], a[0])*2*1.3/(1.3**2-.4**2)
    np.testing.assert_allclose(sum(terms)[0, 0].real, expected, atol=2e-15, rtol=0)
    for gamma in (0., -1., np.nan):
        with pytest.raises(ValueError):
            khd_tensor([1.3], a, a[:, None, :], [0.], [.4], gamma)
    with pytest.raises(ValueError):
        khd_tensor([1.3], a, a[:, None, :], [.5], [.4], .01)


@pytest.mark.parametrize("location", ["frequency", "intermediate", "final", "incident", "gamma_complex", "gamma_vector"])
def test_energy_inputs_are_real_and_gamma_is_scalar(location):
    from gradscf_tools.khd_raman import khd_tensor, vertex_closure
    a = np.array([[.2, -.3, .4]])
    if location == "frequency":
        with pytest.raises(ValueError, match="real"):
            vertex_closure(np.array([.5+.01j]), a[0], np.ones((3, 1)))
        return
    energies, final, laser = np.array([1.3]), np.array([.1]), np.array([.4])
    gamma = .01
    if location == "intermediate":
        energies = energies+1e-3j
    elif location == "final":
        final = final+1e-3j
    elif location == "incident":
        laser = laser+1e-3j
    elif location == "gamma_complex":
        gamma = .01+1e-3j
    else:
        gamma = np.array([.01])
    with pytest.raises(ValueError, match="real"):
        khd_tensor(energies, a, a[:, None, :], final, laser, gamma)
