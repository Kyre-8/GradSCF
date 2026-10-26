"""Zero-temperature band touching: ensemble Green poles and strict domain.

The RPA pole oracle below is an analytic spectral representation, independent
of the contour integration and pole-selection implementation under test.
"""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw.freq import scaled_legendre_grid


query = importlib.import_module('gradscf.gw.pbc.query')


def rpa_example(nw=192):
    gap, a, b = .8, .17, .11
    vector = np.sqrt(np.array([a/2, b/2]))
    pairs = np.array([.15, .17, .2])
    energies = np.array([-.4, 0., .5])
    occupation = np.array([1., .5, 0.])
    differences, signs = np.array([-gap, gap]), np.array([1., -1.])
    block = dict(gram=jnp.asarray(vector[:, None]*vector[None, :]),
        differences=jnp.asarray(differences), signs=jnp.asarray(signs),
        projected=jnp.asarray(vector[:, None]*pairs[None, :]))
    frequencies, weights = scaled_legendre_grid(nw)
    def screened(z):
        return ((a-b)*z+gap*(a+b))/(z*z+(b-a)*z-gap*gap-gap*(a+b))
    positive = screened(1j*frequencies[:, None])*pairs[None, :]**2
    negative = screened(-1j*frequencies[:, None])*pairs[None, :]**2
    center = (a-b)/2
    radius = np.sqrt(center*center+gap*gap+gap*(a+b))
    poles = np.array([center+radius, center-radius])
    residues = ((a-b)*poles+gap*(a+b))/(2*poles+b-a)
    return (energies, occupation, pairs, block, frequencies, weights,
            positive, negative, poles, residues)


@pytest.mark.parametrize('omega', [-.65, -.1, 0., .2, .8])
def test_ensemble_contour_matches_independent_rpa_poles(omega):
    (energies, occupation, pairs, block, frequencies, weights,
     positive, negative, poles, residues) = rpa_example()
    expected = np.sum(pairs**2*(
        (1-occupation)*residues[0]/(omega-energies-poles[0])
        -occupation*residues[1]/(omega-energies-poles[1])))
    actual = query._sigma_imag_full(omega, positive, negative,
        jnp.asarray(energies), 0., frequencies, weights, 0.,
        occupation=jnp.asarray(occupation))
    actual += query._sigma_residue_full(omega, jnp.asarray(energies),
        [block], 0., 0., 1, None, occupation=jnp.asarray(occupation))
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-10)


@pytest.mark.parametrize('fraction', [0., 1/3, .5, 1.])
def test_zero_eta_green_endpoint_matches_independent_spectral_limit(fraction):
    (energies, occupation, pairs, block, frequencies, weights,
     positive, negative, poles, residues) = rpa_example()
    occupation[1] = fraction
    omega = energies[1]
    # The spectral self-energy is regular here. A Green pole lying exactly
    # on the contour requires its half-endpoint term, (1/2-f)*Wc(0); only
    # f=1/2 makes this term vanish. No limiting contour formula is used by
    # this independent oracle.
    expected = np.sum(pairs**2*(
        (1-occupation)*residues[0]/(omega-energies-poles[0])
        -occupation*residues[1]/(omega-energies-poles[1])))
    actual = query._sigma_imag_full(omega, positive, negative,
        jnp.asarray(energies), 0., frequencies, weights, 0.,
        occupation=jnp.asarray(occupation))
    actual += query._sigma_residue_full(omega, jnp.asarray(energies),
        [block], 0., 0., 1, None, occupation=jnp.asarray(occupation))
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-10)


@pytest.mark.parametrize('omega', [-.65, -.1, .2, .8])
def test_fractional_residue_uses_each_signed_green_pole(omega):
    energies, occupation, pairs, block, *_ = rpa_example(8)
    vector_squared = np.diag(np.asarray(block['gram']))
    differences, signs = np.asarray(block['differences']), np.asarray(block['signs'])
    eta = .013
    expected = 0j
    for energy, f, pair in zip(energies, occupation, pairs):
        frequency = omega-energy
        if frequency == 0:
            continue
        prefactor = (1-f) if frequency > 0 else -f
        pole = frequency+1j*eta*np.sign(frequency)
        chi = np.sum(2*signs*vector_squared/(pole+differences+1j*eta*signs))
        expected += prefactor*pair**2*(1/(1-chi)-1)
    actual = query._sigma_residue_full(omega, jnp.asarray(energies),
        [block], 0., eta, 1, None, occupation=jnp.asarray(occupation))
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-12)


def test_bare_response_pole_has_a_finite_screened_zero_eta_limit():
    energies, occupation, pairs, block, *_ = rpa_example(8)
    omega, gap, a, b = .8, .8, .17, .11

    def screened(frequency, eta):
        # Analytic scalar RPA Wc=chi/(1-chi), written as a rational function
        # before evaluation. The bare chi pole at frequency=gap is removable
        # in Wc and has the exact limit -1. All active G poles here are positive.
        empty, occupied = frequency-gap+2j*eta, frequency+gap
        numerator = a*occupied-b*empty
        return numerator/(empty*occupied-numerator)

    assert screened(gap, 0.) == -1.
    values = []
    for eta in (0., 1e-3, 1e-4, 1e-5):
        expected = ((1-occupation[1])*pairs[1]**2*screened(omega-energies[1], eta)
                    +pairs[2]**2*screened(omega-energies[2], eta))
        actual = query._sigma_residue_full(omega, jnp.asarray(energies),
            [block], 0., eta, 1, None, occupation=jnp.asarray(occupation))
        assert np.isfinite(actual)
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-10)
        values.append(complex(actual))
    errors = np.abs(np.asarray(values[1:])-values[0])
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < 3e-6


def test_fractional_imaginary_integral_keeps_both_green_denominators():
    (energies, occupation, _, _, frequencies, weights,
     positive, negative, _, _) = rpa_example(24)
    omega, eta = .18, .007
    frequencies, weights = map(np.asarray, (frequencies, weights))
    positive, negative = map(np.asarray, (positive, negative))
    expected = 0j
    for pole_sign, fraction in ((1, 1-occupation), (-1, occupation)):
        delta = omega-energies+1j*eta*pole_sign
        expected -= np.sum(weights[:, None]*fraction[None, :]*(
            positive/(delta[None, :]-1j*frequencies[:, None])
            +negative/(delta[None, :]+1j*frequencies[:, None])))/(2*np.pi)
    actual = query._sigma_imag_full(omega, jnp.asarray(positive), jnp.asarray(negative),
        jnp.asarray(energies), 0., jnp.asarray(frequencies), jnp.asarray(weights), eta,
        occupation=jnp.asarray(occupation))
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-12)


def test_integer_ensemble_recovers_existing_contour():
    (energies, _, _, block, frequencies, weights,
     positive, negative, _, _) = rpa_example(24)
    energies = jnp.asarray(energies).at[1].set(-.15)
    occupation = jnp.array([1., 1., 0.])
    for omega in (-.65, .2, .8):
        actual = query._sigma_imag_full(omega, positive, negative, energies,
            0., frequencies, weights, .013, occupation=occupation)
        expected = query._sigma_imag_full(omega, positive, negative, energies,
            0., frequencies, weights, .013)
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-12)
        actual = query._sigma_residue_full(omega, energies, [block], 0., .013,
            1, None, occupation=occupation)
        expected = query._sigma_residue_full(omega, energies, [block], 0., .013, 1, None)
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-12)


def test_neutral_touching_is_opt_in_and_preserves_shifted_electron_count():
    source = np.array([[-.7, -.3, .3, .7]])
    target = np.array([[-.4, 0., 0., .4], [-.5, -.2, .2, .5]])
    shifted = target[None]
    with pytest.raises(ValueError, match='gap|touch'):
        query._band_occupations(source, target, shifted, 2, 1e-7)
    ef, occupation = query._band_occupations(source, target, shifted, 2, 1e-7,
                                            band_touching=True)
    assert abs(ef) < 1e-12
    np.testing.assert_array_equal(occupation, [[[1., .5, .5, 0.], [1., 1., 0., 0.]]])
    np.testing.assert_array_equal(np.sum(occupation, axis=-1), [[2., 2.]])


def test_band_overlap_metal_is_rejected_even_with_touching_opt_in():
    source = np.array([[-.7, -.3, .3, .7]])
    target = np.array([[-.5, .1, .3, .5], [-.5, -.3, -.1, .5]])
    assert np.all(target[:, 2]-target[:, 1] > .1)
    with pytest.raises(ValueError, match='gap|overlap|metal'):
        query._band_occupations(source, target, target[None], 2, 1e-7,
                                band_touching=True)


def test_touching_source_is_rejected_to_keep_integer_source_screening():
    source = np.array([[-.7, 0., 0., .7]])
    target = np.array([[-.4, 0., 0., .4]])
    with pytest.raises(ValueError, match='source|gap'):
        query._band_occupations(source, target, target[None], 2, 1e-7,
                                band_touching=True)


def test_larger_frontier_ensemble_preserves_electron_count():
    source = np.array([[-.7, -.3, .2, .4, .7]])
    target = np.array([[-.4, 0., 0., 0., .4]])
    ef, occupation = query._band_occupations(source, target, target[None], 2, 1e-7,
                                            band_touching=True)
    assert abs(ef) < 1e-12
    np.testing.assert_allclose(occupation, [[[1., 1/3, 1/3, 1/3, 0.]]], atol=1e-15)
    np.testing.assert_allclose(np.sum(occupation, axis=-1), [[2.]], atol=1e-15)


def test_requested_touching_cluster_must_be_complete():
    energy = np.array([[-.4, 0., 0., .4]])
    with pytest.raises(ValueError, match='complete.*subspace'):
        query._query_subspaces(energy, np.array([1]), 1e-7)
    assert query._query_subspaces(energy, np.array([1, 2]), 1e-7) == [((1, 2),)]


def test_fractional_negative_exchange_head_is_unitary_covariant():
    left = jnp.eye(3, dtype=jnp.complex128)
    right = jnp.array([[1., 1j], [1j, 1.], [0., 0.]])/jnp.sqrt(2.)
    occupation = jnp.array([1., .5, 0.])
    weights = jnp.ones(3)
    actual = query._negative_head_exchange(-.6, left, right, weights, 2, 3,
                                         matrix=True, occupation=occupation)
    overlap = np.asarray(left).conj().T@np.asarray(right)
    expected = .2*overlap.conj().T@np.diag(occupation)@overlap
    np.testing.assert_allclose(actual, expected, atol=2e-15)
    frame = np.array([[1., 1j], [1j, 1.]])/np.sqrt(2.)
    rotated = query._negative_head_exchange(-.6, left, right@frame, weights, 2, 3,
                                          matrix=True, occupation=occupation)
    np.testing.assert_allclose(rotated, frame.conj().T@expected@frame, atol=2e-15)


@pytest.mark.parametrize('ensemble', [False, True])
@pytest.mark.parametrize('eta', [0., 1e-3])
def test_static_subtraction_resolves_near_green_poles_with_coarse_quadrature(ensemble, eta):
    (energies, occupation, pairs, block, frequencies, weights,
     positive, negative, poles, residues) = rpa_example(16)
    if not ensemble:
        energies[1], occupation[1] = -.15, 1.
    center = energies[1]
    static = np.sum(residues/(-poles))*pairs**2
    helper_occupation = jnp.asarray(occupation) if ensemble else None
    for offset in (-1e-6, -1e-8, 1e-8, 1e-6):
        omega = center+offset
        expected = np.sum(pairs**2*((1-occupation)*residues[0]/(omega-energies-poles[0])
                         -occupation*residues[1]/(omega-energies-poles[1])))
        if eta:
            # Independent convention: the imaginary integral uses G_eta=0;
            # only real-pole W retains its signed time-ordered eta shifts.
            for m, (energy, f) in enumerate(zip(energies, occupation)):
                zeta = omega-energy
                amplitude = (1-f) if zeta > 0 else -f
                if amplitude == 0:
                    continue
                chi = np.sum(2*np.asarray(block['signs'])*np.diag(np.asarray(block['gram']))
                    /(zeta+np.asarray(block['differences'])
                      +1j*eta*(np.sign(zeta)+np.asarray(block['signs']))))
                finite = chi/(1-chi)
                exact = np.sum(residues/(zeta-poles))
                expected += amplitude*pairs[m]**2*(finite-exact)
        actual = query._sigma_imag_full(omega, positive, negative, jnp.asarray(energies),
            0., frequencies, weights, eta, occupation=helper_occupation, static=jnp.asarray(static))
        actual += query._sigma_residue_full(omega, jnp.asarray(energies), [block],
            0., eta, 1, None, occupation=helper_occupation, static=jnp.asarray(static))
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)


def test_static_subtraction_matrix_preserves_the_complex_query_frame():
    (energies, occupation, pairs, block, frequencies, weights,
     positive, negative, poles, residues) = rpa_example(16)
    vectors = np.sqrt(np.diag(np.asarray(block['gram'])))
    moments = np.array([[.15, .05+.03j], [.17, -.06+.08j], [.2, .09-.04j]])
    omega = 1e-8
    coefficients = ((1-occupation)*residues[0]/(omega-energies-poles[0])
                    -occupation*residues[1]/(omega-energies-poles[1]))
    expected = np.einsum('m,mp,mq->pq', coefficients, moments.conj(), moments)

    def evaluate(external):
        products = np.einsum('mp,mq->mpq', external.conj(), external)
        local = dict(block, projected=jnp.asarray(vectors[:, None, None]*external[None, :, :]))
        plus = positive[:, :1, None, None]/pairs[0]**2*products[None]
        minus = negative[:, :1, None, None]/pairs[0]**2*products[None]
        static = jnp.asarray(np.sum(residues/(-poles))*products)
        value = query._sigma_imag_full(omega, plus, minus, jnp.asarray(energies),
            0., frequencies, weights, 0., occupation=jnp.asarray(occupation), static=static)
        value += query._sigma_residue_full(omega, jnp.asarray(energies), [local],
            0., 0., 1, None, occupation=jnp.asarray(occupation), static=static)
        return np.asarray(value)

    original = evaluate(moments)
    np.testing.assert_allclose(original, expected, atol=2e-7, rtol=0)
    frame = np.array([[1., 1j], [1j, 1.]])/np.sqrt(2.)
    np.testing.assert_allclose(evaluate(moments@frame), frame.conj().T@original@frame,
                               atol=3e-13, rtol=0)
