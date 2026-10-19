"""Moment-anchored deterministic realizations, independent finite baths."""
import importlib

import numpy as np
import pytest

from test_causal_gram import _bath


def _interpolate():
    return importlib.import_module('gradscf.solvers.poles').interpolate_causal


def _certificate(fit, moment):
    assert fit.converged
    assert fit.iterations == 0
    assert np.all(np.diff(fit.poles) >= 0)
    assert np.linalg.eigvalsh(fit.residues).min() > -1e-12
    np.testing.assert_allclose(fit.residues.sum(axis=0), moment, atol=3e-10, rtol=3e-10)
    assert fit.diagnostics['completion'] == 'zero on orthogonal complement of range(X)'
    assert fit.diagnostics['interpolation_relative_error'] <= 1e-8


def test_real_off_grid_two_pole_measure_is_recovered_without_a_supplied_grid():
    from gradscf.solvers.poles import evaluate_poles
    poles, weight = np.array([-1.1, .8]), np.array([.3, .7])
    z = 1j*np.geomspace(.04, 15., 10)
    values = np.sum(weight/(z[:, None]-poles), axis=1)[:, None, None]
    fit = _interpolate()(z[::2], values[::2], moment=np.ones((1, 1)), tolerance=1e-9)
    _certificate(fit, np.ones((1, 1)))
    np.testing.assert_allclose(fit.poles, poles, atol=2e-8, rtol=0)
    np.testing.assert_allclose(evaluate_poles(fit, z[1::2]), values[1::2], atol=2e-9, rtol=2e-9)


def test_complex_noncommuting_residues_keep_full_matrix_orientation_and_holdout():
    from gradscf.solvers.poles import evaluate_poles
    poles, coupling, z, values = _bath()
    moment = coupling.conj().T@coupling
    fit = _interpolate()(z[::2], values[::2], moment=moment, tolerance=1e-8)
    _certificate(fit, moment)
    np.testing.assert_allclose(fit.poles, poles, atol=2e-7, rtol=2e-8)
    np.testing.assert_allclose(evaluate_poles(fit, z[1::2]), values[1::2], atol=2e-8, rtol=2e-8)
    assert np.max(abs(values[:, 0, 1]-values[:, 1, 0])) > 1e-2
    assert fit.diagnostics['retained_rank'] == 4
    assert fit.diagnostics['free_complement_dimension'] == 0


@pytest.mark.parametrize('singular', [False, True])
def test_realization_is_orbital_covariant_including_singular_moment_support(singular):
    from gradscf.solvers.poles import evaluate_poles
    _, coupling, z, values = _bath()
    if singular:
        coupling = np.column_stack((coupling, np.zeros(len(coupling))))
        values = np.pad(values, ((0, 0), (0, 1), (0, 1)))
    n = coupling.shape[1]
    frame = np.exp(2j*np.pi*np.arange(n)[:, None]*np.arange(n)[None]/n)/np.sqrt(n)
    moment = coupling.conj().T@coupling
    left = _interpolate()(z[::2], values[::2], moment=moment, tolerance=1e-8)
    right = _interpolate()(z[::2], frame.conj().T@values[::2]@frame,
        moment=frame.conj().T@moment@frame, tolerance=1e-8)
    _certificate(right, frame.conj().T@moment@frame)
    assert right.diagnostics['moment_rank'] == 2
    np.testing.assert_allclose(right.poles, left.poles, atol=2e-7, rtol=2e-8)
    query = np.r_[z[1::2], -.35+.06j, .7+.04j]
    np.testing.assert_allclose(evaluate_poles(right, query),
        frame.conj().T@evaluate_poles(left, query)@frame, atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize('bad', ['negative_measure', 'inconsistent_moment'])
def test_nonpositive_original_data_are_rejected_without_causal_projection(bad):
    _, coupling, z, values = _bath()
    moment = coupling.conj().T@coupling
    if bad == 'negative_measure':
        values = -values
    else:
        moment = .2*moment
    with pytest.raises(ValueError, match='PSD|positive|causal'):
        _interpolate()(z, values, moment=moment, tolerance=1e-8)


def test_positive_gram_with_mass_at_infinity_fails_the_nullspace_constraint():
    from gradscf.solvers.poles import causal_gram
    z, values, moment = np.array([1j]), np.zeros((1, 1, 1)), np.ones((1, 1))
    np.testing.assert_array_equal(causal_gram(z, values, moment=moment), np.diag([1., 0.]))
    with pytest.raises(ArithmeticError, match='null|compatible|constraint'):
        _interpolate()(z, values, moment=moment, tolerance=1e-8)


def test_free_completion_retains_the_range_null_cross_block():
    fit = _interpolate()(np.array([1j]), np.array([[[-.5j]]]),
                         moment=np.ones((1, 1)), tolerance=1e-10)
    _certificate(fit, np.ones((1, 1)))
    np.testing.assert_allclose(fit.poles, [-1., 1.], atol=2e-12, rtol=0)
    np.testing.assert_allclose(fit.residues[:, 0, 0], [.5, .5], atol=2e-12, rtol=0)
    assert fit.diagnostics['free_complement_dimension'] == 1


def test_controlled_rank_approximation_is_certified_against_original_data():
    from gradscf.solvers.poles import evaluate_poles
    poles, weight = np.array([-1.1, .8, 2.3]), np.array([.3, .7, 1e-11])
    z = 1j*np.geomspace(.04, 15., 12)
    values = np.sum(weight/(z[:, None]-poles), axis=1)[:, None, None]
    moment = np.array([[weight.sum()]])
    fit = _interpolate()(z[::2], values[::2], moment=moment,
                         tolerance=1e-5, rank_tolerance=1e-7)
    _certificate(fit, moment)
    assert fit.diagnostics['retained_rank'] == 2
    assert fit.diagnostics['raw_hamiltonian_antihermitian_relative_error'] > fit.diagnostics['roundoff_relative_tolerance']
    assert fit.diagnostics['hx_relative_error'] <= 1e-5
    np.testing.assert_allclose(evaluate_poles(fit, z[1::2]), values[1::2], atol=2e-8, rtol=2e-8)


def test_zero_moment_has_only_the_zero_dynamic_function_and_free_physical_spectrum():
    from gradscf.solvers.poles import auxiliary_spectrum, evaluate_poles
    z = 1j*np.array([.1, .4, 1.])
    fit = _interpolate()(z, np.zeros((3, 2, 2)), moment=np.zeros((2, 2)), tolerance=1e-8)
    assert fit.converged and fit.diagnostics['moment_rank'] == 0
    np.testing.assert_array_equal(evaluate_poles(fit, z), np.zeros((3, 2, 2)))
    energy, weights = auxiliary_spectrum(np.diag([-.4, .3]), fit)
    np.testing.assert_allclose(energy, [-.4, .3], atol=1e-14)
    np.testing.assert_allclose(weights, 1., atol=1e-14)
    with pytest.raises(ValueError, match='moment|support|PSD'):
        _interpolate()(z, np.tile(np.eye(2), (3, 1, 1)), moment=np.zeros((2, 2)), tolerance=1e-8)


def test_rank_truncation_cannot_hide_failure_of_original_matrix_certificates():
    _, coupling, z, values = _bath()
    with pytest.raises((ValueError, ArithmeticError), match='rank|support|constraint|interpol|moment'):
        _interpolate()(z, values, moment=coupling.conj().T@coupling,
                       tolerance=1e-10, rank_tolerance=.9)
