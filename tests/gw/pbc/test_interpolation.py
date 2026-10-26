"""Interpolation is explicit and must preserve matrix causality and state count."""
import numpy as np
import pytest


def test_periodic_trilinear_weights_are_positive_and_reproduce_every_source_node():
    from gradscf.gw.pbc.interpolation import periodic_weights
    source = np.stack(np.meshgrid([0., .5], [0., .5], [0.], indexing='ij'), -1).reshape(-1, 3)
    order = [2, 0, 3, 1]
    source = source[order]
    np.testing.assert_allclose(periodic_weights(source, source), np.eye(4), atol=1e-14)
    query = np.array([[.25, .25, 0.], [-.1, .8, 3.]])
    weights = periodic_weights(source, query)
    assert np.min(weights) >= 0.
    np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-14)
    np.testing.assert_allclose(weights[0], .25, atol=1e-14)
    np.testing.assert_allclose(weights, periodic_weights(source, query+np.array([1., -2., 4.])))


def test_ao_operator_interpolation_is_invariant_to_source_mo_phases():
    from gradscf.gw.pbc.interpolation import interpolate_operator
    rng = np.random.default_rng(8)
    nk, nao = 2, 3
    s = np.tile(np.diag([1., 2., 3.]), (nk, 1, 1))
    c = np.tile(np.diag(1/np.sqrt([1., 2., 3.])), (nk, 1, 1)).astype(complex)
    a = rng.normal(size=(nk, nao, nao))+1j*rng.normal(size=(nk, nao, nao))
    a = a+a.conj().swapaxes(-1, -2)
    phase = np.exp(1j*rng.normal(size=(nk, nao)))
    changed = phase.conj()[:, :, None]*a*phase[:, None, :]
    weight = np.array([[.3, .7], [1., 0.]])
    left = interpolate_operator(weight, s, c, a, c[:2])
    right = interpolate_operator(weight, s, c*phase[:, None, :], changed, c[:2])
    np.testing.assert_allclose(left, right, atol=1e-13)
    np.testing.assert_allclose(left[1], a[0], atol=1e-13)


def test_electronic_dos_counts_uniform_bz_states_not_path_lengths():
    from gradscf.gw.pbc.interpolation import electronic_dos, band_gaps
    energy = np.array([[-.5, .2], [-.4, .3]])
    axis = np.linspace(-1., 1., 5001)
    dos = electronic_dos(axis, energy, weights=[.25, .75], sigma=.03)
    np.testing.assert_allclose(np.trapezoid(dos, axis), 4., atol=1e-12)
    gaps = band_gaps(energy, nocc=1)
    assert gaps['sampled_gap_hartree'] == pytest.approx(.6)
    assert gaps['minimum_direct_gap_hartree'] == pytest.approx(.7)
    with pytest.raises(ValueError, match='converged|finite'):
        band_gaps(energy, nocc=1, valid=np.array([[True, False], [True, True]]))


def test_gap_tracks_all_occupied_and_empty_branches_after_qp_reordering():
    from gradscf.gw.pbc.interpolation import band_gaps
    # Diagonal GW retains reference band labels; their corrected energies
    # need not remain ordered even within the occupied/empty groups.
    energy = np.array([[-.4, -.6, .3, .2], [-.5, -.45, .4, .35]])
    gaps = band_gaps(energy, nocc=2)
    assert gaps['sampled_gap_hartree'] == pytest.approx(.6)
    assert gaps['minimum_direct_gap_hartree'] == pytest.approx(.6)
    invalid = np.ones_like(energy, dtype=bool)
    invalid[0, 0] = False
    with pytest.raises(ValueError, match='converged|finite'):
        band_gaps(energy, nocc=2, valid=invalid)


def test_lorentzian_qp_dos_uses_the_same_retarded_halfwidth_as_interacting_spectra():
    from gradscf.gw.pbc.interpolation import electronic_dos
    energy = np.array([[-.4, .3]])
    axis = np.linspace(-2., 2., 4001)
    halfwidth = .04
    actual = electronic_dos(axis, energy, sigma=halfwidth, smearing='lorentzian')
    expected_integral = 2/np.pi*np.sum(np.arctan((axis[-1]-energy)/halfwidth)
                                     -np.arctan((axis[0]-energy)/halfwidth))
    np.testing.assert_allclose(np.trapezoid(actual, axis), expected_integral, atol=1e-8)
    assert actual.min() > 0.
