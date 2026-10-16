"""Finite oscillator assembly, exact harmonic limits and bounded-model gates."""
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import EigenSolverConfig, solve_hermitian


def test_coordinate_fourth_power_uses_padding_at_the_top_basis_state():
    from gradscf_tools.anharmonic import oscillator_powers

    size, mass, frequency = 7, 1.3, .7
    powers = oscillator_powers(frequency, mass, size, max_power=4)
    quantum = np.arange(size)
    expected = 3*(2*quantum**2+2*quantum+1)/(4*mass**2*frequency**2)
    np.testing.assert_allclose(np.diag(powers[4]), expected, atol=2e-13, rtol=1e-13)
    assert abs(float(jnp.linalg.matrix_power(powers[1], 4)[-1, -1])-expected[-1]) > 1.


@pytest.mark.parametrize("frequencies,masses,sizes", [((.023,), (918.58,), (10,)),
    ((.07, .113), (1., 2.), (7, 6))])
def test_polynomial_hamiltonian_recovers_exact_harmonic_levels(frequencies, masses, sizes):
    from gradscf_tools.anharmonic import polynomial_hamiltonian

    dimension = len(sizes)
    coefficients = {tuple(0 for _ in sizes): .012}
    for axis in range(dimension):
        powers = tuple(2 if i == axis else 0 for i in range(dimension))
        coefficients[powers] = .5*masses[axis]*frequencies[axis]**2
    matrix = polynomial_hamiltonian(frequencies, masses, sizes, coefficients)
    solved = solve_hermitian(matrix, config=EigenSolverConfig(method="dense", nroots=6, atol=1e-12))
    assert np.all(solved.converged)
    states = np.array(list(np.ndindex(*sizes)))
    expected = np.sort(.012+((states+.5)*np.array(frequencies)).sum(axis=1))[:6]
    np.testing.assert_allclose(solved.values, expected, atol=2e-13, rtol=0)


def test_two_mode_mixed_monomial_has_the_expected_tensor_product_convention():
    from gradscf_tools.anharmonic import oscillator_powers, polynomial_hamiltonian

    w, m, sizes = (.7, .9), (1., 1.2), (5, 6)
    base = {(2, 0): .5*m[0]*w[0]**2, (0, 2): .5*m[1]*w[1]**2}
    before = polynomial_hamiltonian(w, m, sizes, base)
    after = polynomial_hamiltonian(w, m, sizes, {**base, (2, 2): .08})
    first, second = [oscillator_powers(ww, mm, nn)[2] for ww, mm, nn in zip(w, m, sizes)]
    np.testing.assert_allclose(after-before, .08*jnp.kron(first, second), atol=1e-13, rtol=0)


def test_two_mode_harmonic_linear_dipole_has_only_the_two_fundamentals():
    from gradscf_tools.anharmonic import polynomial_hamiltonian, polynomial_matrix

    frequencies, masses, sizes = (.07, .113), (1., 2.), (7, 6)
    potential = {(2, 0): .5*masses[0]*frequencies[0]**2,
                 (0, 2): .5*masses[1]*frequencies[1]**2}
    solved = solve_hermitian(polynomial_hamiltonian(frequencies, masses, sizes, potential),
        config=EigenSolverConfig(method="dense", nroots=6, atol=1e-12))
    moments = []
    for powers in ((1, 0), (0, 1)):
        matrix = polynomial_matrix(frequencies, masses, sizes, {powers: 1.})
        moments.append(solved.vectors[:, 0]@matrix@solved.vectors[:, 1:])
    strengths = np.sum(np.asarray(moments)**2, axis=0)
    np.testing.assert_allclose(strengths[:2], 1/(2*np.array(masses)*frequencies), atol=2e-13, rtol=0)
    np.testing.assert_allclose(strengths[2:], 0., atol=1e-25, rtol=0)


def test_bad_mass_and_excessive_capacity_fail_before_matrix_allocation():
    from gradscf_tools.anharmonic import oscillator_powers, polynomial_hamiltonian

    with pytest.raises(ValueError, match="positive"):
        oscillator_powers(.2, 0., 6)
    with pytest.raises(ValueError, match="1024"):
        polynomial_hamiltonian((.2, .3), (1., 1.), (40, 40), {(2, 0): .1})


def test_quartic_certificate_distinguishes_positive_negative_and_indeterminate():
    from gradscf_tools.anharmonic import quartic_certificate

    one = quartic_certificate({(1,): 1e-10, (2,): .5, (3,): -.1, (4,): .1})
    assert one["quartic_status"] == "positive" and one["stationary_status"] == "unique"
    assert one["stationary_radius"] < 1e-8
    negative = quartic_certificate({(2,): .5, (4,): -.1})
    assert negative["quartic_status"] == "not_positive"
    multiple = quartic_certificate({(2,): .5, (3,): -2., (4,): 1.})
    assert multiple["quartic_status"] == "positive"
    assert multiple["stationary_status"] == "indeterminate"
    two = quartic_certificate({(2, 0): .5, (0, 2): .7, (3, 0): -.1,
                               (4, 0): .1, (0, 4): .12, (2, 2): .1})
    assert two["quartic_status"] == "positive" and two["stationary_status"] == "unique"
