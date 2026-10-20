"""Matrix Nevanlinna kernels checked against independent finite bath Grams."""
import importlib

import numpy as np
import pytest


def _bath():
    poles = np.array([-1.37, -.19, .63, 1.92])
    coupling = np.array([[.44, .12+.17j], [.05-.21j, .32],
                         [.27+.09j, -.13+.14j], [.16-.08j, .23+.11j]])
    z = np.array([.03+.07j, -.11+.23j, .08+.61j, .12+1.4j, -.03+3.7j])
    values = np.einsum('li,wl,lj->wij', coupling.conj(),
                      1/(z[:, None]-poles), coupling)
    return poles, coupling, z, values


def _function():
    module = importlib.import_module('gradscf.solvers.poles')
    function = getattr(module, 'causal_gram', None)
    assert callable(function), 'Missing shared causal_gram capability'
    return function


@pytest.mark.parametrize('anchored', [False, True])
def test_causal_gram_equals_literal_bath_feature_gram(anchored):
    function = _function()
    poles, coupling, z, values = _bath()
    blocks = [coupling/(point.conjugate()-poles)[:, None] for point in z]
    if anchored:
        blocks.insert(0, coupling)
    feature = np.concatenate(blocks, axis=1)
    expected = feature.conj().T@feature
    actual = function(z, values, moment=coupling.conj().T@coupling if anchored else None)
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(actual, actual.conj().T, atol=2e-14)
    assert np.linalg.eigvalsh(actual).min() > -2e-14


def test_causal_gram_reports_noncausal_data_and_inconsistent_moment_without_projection():
    function = _function()
    _, coupling, z, values = _bath()
    negative = function(z, -values)
    assert np.linalg.eigvalsh(negative).min() < -.1
    inconsistent = function(z, values, moment=.2*coupling.conj().T@coupling)
    assert np.linalg.eigvalsh(inconsistent).min() < -.01


def test_complex_gram_and_moment_anchor_are_orbital_unitary_covariant():
    function = _function()
    _, coupling, z, values = _bath()
    unitary = np.array([[1., 1j], [1j, 1.]])/np.sqrt(2.)
    moment = coupling.conj().T@coupling
    left = function(z, values, moment=moment)
    right = function(z, unitary.conj().T@values@unitary,
                     moment=unitary.conj().T@moment@unitary)
    block_rotation = np.kron(np.eye(len(z)+1), unitary)
    np.testing.assert_allclose(right, block_rotation.conj().T@left@block_rotation,
                               atol=3e-14, rtol=3e-14)


def test_causal_gram_rejects_invalid_domain_and_nonhermitian_moment():
    function = _function()
    _, coupling, z, values = _bath()
    with pytest.raises(ValueError, match='upper|imaginary'):
        function(-z, values)
    with pytest.raises(ValueError, match='shape|matrix|moment'):
        function(z, values, moment=np.eye(3))
    moment = coupling.conj().T@coupling
    moment[0, 1] += .1j
    with pytest.raises(ValueError, match='Hermitian'):
        function(z, values, moment=moment)
