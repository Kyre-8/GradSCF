"""Independent real-W broadening, without changing the legacy CD default."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import LinearSolverConfig
from gradscf.solvers.linear import factor_shifted

from test_self_consistency import _model
from test_shifted_screening import _block


@pytest.mark.parametrize('omega,eta_w', [(-.9, .006), (.9, .006), (1.75, 0.)])
@pytest.mark.parametrize('cached', [False, True])
def test_independent_screening_eta_matches_nonpaired_complex_auxiliary_oracle(omega, eta_w, cached):
    """The advanced negative branch is evaluated as a matrix, not conjugated."""
    query = importlib.import_module('gradscf.gw.pbc.query')
    block, nk, eta_g = _block(), 2, .003
    block['screening_eta'] = eta_w
    delta, signs, gram = (np.asarray(block[name]) for name in ('differences', 'signs', 'gram'))
    matrix = -jnp.diag(jnp.asarray(delta))+2/nk*jnp.asarray(signs)[:, None]*block['gram']
    if cached:
        block['shifted_factor'] = factor_shifted(matrix, symmetric=False)
    energy = np.array([-.5, .5])
    expected = np.zeros((2, 2), complex)
    expected_derivative = np.zeros_like(expected)
    for m, em in enumerate(energy):
        amplitude = float((em > 0) and (em < omega)) if omega > 0 else -float((em < 0) and (em > omega))
        if amplitude:
            frequency = omega-em
            denominator = frequency+delta+1j*eta_w*np.sign(frequency)
            projected = np.asarray(block['projected'])[:, m]
            # The division-free original equation also covers a zero-width
            # W bare pole while eta_G remains nonzero.
            operator = np.diag(nk*denominator/(2*signs))-gram
            solution = np.linalg.solve(operator, projected)
            tangent = np.linalg.solve(operator, -nk/(2*signs[:, None])*solution)
            expected += amplitude*projected.conj().T@solution/nk
            expected_derivative += amplitude*projected.conj().T@tangent/nk
    actual = lambda value: query._sigma_residue_full(value, jnp.asarray(energy), [block],
        0., eta_g, nk, LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13))
    np.testing.assert_allclose(actual(jnp.asarray(omega)), expected, atol=3e-12, rtol=2e-11)
    derivative = jax.jvp(actual, (jnp.asarray(omega),), (jnp.asarray(1.),))[1]
    np.testing.assert_allclose(derivative, expected_derivative, atol=3e-11, rtol=2e-10)


def test_paired_negative_retarded_matrix_is_positive_advanced_adjoint():
    """For a paired spectrum W_R(-w)=W_R(w)^dagger; elementwise conj fails."""
    query = importlib.import_module('gradscf.gw.pbc.query')
    vertex = np.array([[.2+.07j, -.09+.04j], [.03-.11j, .16+.02j]])
    transition = np.concatenate((vertex, vertex), axis=1)
    external = np.array([[.4+.2j, .12-.1j], [.09+.16j, .3-.07j]])
    delta, signs = np.array([-1.1, -1.4, 1.1, 1.4]), np.array([1., 1., -1., -1.])
    gram, projected = transition.conj().T@transition, transition.conj().T@external
    eta_w, frequency = .007, .4
    def auxiliary(z):
        pi = (transition*(2*signs/(z+delta))[None])@transition.conj().T
        wc = np.linalg.solve(np.eye(2)-pi, pi)
        return external.conj().T@wc@external
    positive = auxiliary(frequency+1j*eta_w)
    negative_retarded = auxiliary(-frequency+1j*eta_w)
    np.testing.assert_allclose(negative_retarded, positive.conj().T, atol=2e-14, rtol=0)
    assert np.max(abs(negative_retarded-positive.conj())) > 1e-5
    block = dict(gram=jnp.asarray(gram), differences=jnp.asarray(delta), signs=jnp.asarray(signs),
        projected=jnp.broadcast_to(jnp.asarray(projected)[:, None], (4, 2, 2)),
        q_weight=1., screening_eta=eta_w)
    config = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13)
    # The CD residue chooses the advanced branch at negative frequency.
    result = query._sigma_residue_full(jnp.asarray(-.9), jnp.array([-.5, .5]),
                                      [block], 0., .003, 1, config)
    np.testing.assert_allclose(result, -positive, atol=2e-13, rtol=2e-11)


def test_source_screening_eta_preserves_imaginary_cache_and_legacy_default():
    source = importlib.import_module('gradscf.gw.pbc.source')
    model = _model(scale=.4)
    legacy = source.build_correlation(model, model.energy, nw=4, eta=.003)
    explicit_none = source.build_correlation(model, model.energy, nw=4, eta=.003, screening_eta=None)
    broadened = source.build_correlation(model, model.energy, nw=4, eta=.003, screening_eta=.006)
    zero = source.build_correlation(model, model.energy, nw=4, eta=.003, screening_eta=0.)
    for name in ('positive', 'negative', 'static'):
        for old, new, exact in zip(legacy[name], broadened[name], zero[name]):
            np.testing.assert_allclose(new, old, atol=2e-14, rtol=0)
            np.testing.assert_allclose(exact, old, atol=2e-14, rtol=0)
    np.testing.assert_allclose(source.correlation_at(legacy, 0, jnp.asarray(.9)),
        source.correlation_at(explicit_none, 0, jnp.asarray(.9)), atol=2e-14, rtol=0)
    assert all(block['screening_eta'] == .006 for block in broadened['blocks'])
    assert all(block['screening_eta'] == 0. for block in zero['blocks'])
    assert np.max(abs(np.asarray(source.correlation_at(legacy, 0, jnp.asarray(.9))
                               -source.correlation_at(broadened, 0, jnp.asarray(.9))))) > 1e-10


@pytest.mark.parametrize('screening_eta', [-.001, np.inf, np.nan])
def test_source_rejects_invalid_independent_screening_eta(screening_eta):
    source = importlib.import_module('gradscf.gw.pbc.source')
    model = _model()
    with pytest.raises(ValueError, match='screening_eta'):
        source.build_correlation(model, model.energy, nw=2, screening_eta=screening_eta)
