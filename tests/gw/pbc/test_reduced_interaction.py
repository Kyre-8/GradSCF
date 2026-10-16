"""Finite-part low-D correlation kernels and independent signed-v oracles."""
import importlib
import importlib.util
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.periodic.ao import reciprocal_grid
from gradscf.integrals.periodic.coulomb import ReducedBoundary, coulomb_kernel
from gradscf.gw.screened import solve_transition_screening
from gradscf.solvers import LinearSolverConfig


def api():
    return importlib.import_module('gradscf.gw.pbc.interaction')


def inputs(dimension):
    lattice = jnp.diag(jnp.array([5., 16., 16.] if dimension == 1 else [5., 6., 18.]))
    mesh = (3, 5, 5)
    vectors = reciprocal_grid(lattice, mesh)
    volume = jnp.linalg.det(lattice)
    return SimpleNamespace(lattice=lattice, gvectors=vectors,
        weights=jnp.full(len(vectors), volume/len(vectors)),
        boundary=ReducedBoundary(lattice, dimension, mesh) if dimension < 3 else None)


def test_interaction_api_is_available():
    assert importlib.util.find_spec('gradscf.gw.pbc.interaction') is not None
    assert callable(api().correlation_kernel) and callable(api().finite_head)


def test_three_dimensional_branch_preserves_original_values_and_ad():
    data = inputs(3)
    vectors = jnp.array([[0., 0., 0.], [.2, -.3, .4], [.7, .1, -.2]])
    def actual(scale):
        return jnp.sum(api().correlation_kernel(data, vectors*scale))
    expected = lambda scale: jnp.sum(coulomb_kernel(vectors*scale))
    np.testing.assert_array_equal(api().correlation_kernel(data, vectors), coulomb_kernel(vectors))
    np.testing.assert_allclose(jax.jit(jax.value_and_grad(actual))(1.),
                               jax.jit(jax.value_and_grad(expected))(1.), atol=0., rtol=0.)
    np.testing.assert_allclose(jax.jvp(actual, (1.,), (.4,)),
                               jax.jvp(expected, (1.,), (.4,)), atol=0., rtol=0.)
    np.testing.assert_allclose(api().finite_head(data), 0., atol=0.)


@pytest.mark.parametrize('dimension', [1, 2])
def test_actual_low_dimensional_grids_keep_body_and_raw_finite_head(dimension):
    data = inputs(dimension)
    raw = coulomb_kernel(data.gvectors, boundary=data.boundary)
    assert float(raw[0]) < 0. and np.all(np.asarray(raw[1:]) >= 0.)
    actual = api().correlation_kernel(data, data.gvectors)
    assert float(actual[0]) == 0.
    np.testing.assert_array_equal(actual[1:], raw[1:])
    np.testing.assert_allclose(api().finite_head(data), raw[0]/jnp.linalg.det(data.lattice), atol=1e-13)
    shifted = data.gvectors+jnp.array([.13, 0., 0.])
    np.testing.assert_array_equal(api().correlation_kernel(data, shifted),
                                  coulomb_kernel(shifted, boundary=data.boundary))


@pytest.mark.parametrize('bad', [-1e-15, -1., np.nan, np.inf])
def test_nonhead_negative_or_nonfinite_values_are_never_clipped(monkeypatch, bad):
    data = inputs(1)
    values = jnp.ones(len(data.gvectors)).at[0].set(-2.).at[1].set(bad)
    monkeypatch.setattr(api(), 'coulomb_kernel', lambda vectors, **kwargs: values)
    with pytest.raises(ValueError, match='negative|finite'):
        api().correlation_kernel(data, data.gvectors)


def test_only_an_exact_zero_negative_head_is_removed(monkeypatch):
    data = inputs(1)
    values = jnp.ones(len(data.gvectors)).at[0].set(-2.)
    monkeypatch.setattr(api(), 'coulomb_kernel', lambda vectors, **kwargs: values)
    with pytest.raises(ValueError, match='negative'):
        api().correlation_kernel(data, data.gvectors+jnp.array([1e-13, 0., 0.]))
    positive = values.at[0].set(2.)
    monkeypatch.setattr(api(), 'coulomb_kernel', lambda vectors, **kwargs: positive)
    np.testing.assert_array_equal(api().correlation_kernel(data, data.gvectors), positive)
    np.testing.assert_allclose(api().finite_head(data), 2./jnp.linalg.det(data.lattice), atol=1e-15)


def test_low_dimensional_scope_is_eager_and_volume_is_the_fft_volume():
    data = inputs(1)
    with pytest.raises(NotImplementedError, match='eager|AD'):
        jax.jit(lambda scale: api().correlation_kernel(data, scale*data.gvectors))(1.)
    bad = SimpleNamespace(**vars(data),)  # retain the original boundary/metric
    bad.weights = 2*data.weights
    with pytest.raises(ValueError, match='volume|weights'):
        api().finite_head(bad)


def orbital_pairs(data, *, nonorthogonal=False):
    """Four arbitrary complex grid orbitals, orthonormal in the same FFT metric."""
    rng = np.random.default_rng(20261016)
    ao = rng.normal(size=(len(data.gvectors), 5))+1j*rng.normal(size=(len(data.gvectors), 5))
    weight, volume = float(data.weights[0]), float(jnp.linalg.det(data.lattice))
    overlap = weight*ao.conj().T@ao
    coefficients = np.linalg.inv(np.linalg.cholesky(overlap)).conj().T[:, :4]
    if nonorthogonal:
        coefficients[:, 2] += .1*coefficients[:, 0]
    gram = coefficients.conj().T@overlap@coefficients
    values = ao@coefficients
    pairs = values.conj()[:, :, None]*values[:, None, :]
    transformed = np.fft.fftn(pairs.reshape(*data.boundary.mesh, 4, 4), axes=(0, 1, 2))
    rho = transformed.reshape(len(pairs), 16)*np.sqrt(volume)/len(pairs)
    # This is the raw Fourier head before any Coulomb weighting/removal.
    np.testing.assert_allclose(rho[0].reshape(4, 4)*np.sqrt(volume), gram, atol=3e-15)
    return rho, gram


def screened_pair_matrices(data, rho):
    raw = np.asarray(coulomb_kernel(data.gvectors, boundary=data.boundary))
    body = np.asarray(api().correlation_kernel(data, data.gvectors))
    energies = np.array([-1., -.6, .2, .7])
    occupation = np.array([1., 1., 0., 0.])
    pairs = [(i, a) for i in range(2) for a in range(2, 4)]
    pairs += [(a, i) for i, a in pairs.copy()]
    p, q = np.array(pairs).T
    indices = p*4+q
    weights = 2*(occupation[p]-occupation[q])/(energies[p]-energies[q])
    transitions = rho[:, indices]
    # Independent full auxiliary signed-v Dyson equation, without sqrt(v).
    chi = (transitions*weights)@transitions.conj().T
    bare = np.diag(raw)
    full = np.linalg.solve(np.eye(len(raw))-bare@chi, bare)
    expected = rho.conj().T@(full-bare)@rho
    factors, targets = np.sqrt(body)[:, None]*transitions, np.sqrt(body)[:, None]*rho
    projection = factors.conj().T@targets
    solved = solve_transition_screening(factors.conj().T@factors, weights, projection,
        config=LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13, max_dense=32))
    assert bool(solved.converged)
    actual = projection.conj().T@np.asarray(solved.solution)
    return actual, expected, rho.conj().T@bare@rho, targets.conj().T@targets


@pytest.mark.parametrize('dimension', [1, 2])
def test_correlation_matches_full_signed_v_and_bare_head_is_retained(dimension):
    data = inputs(dimension)
    rho, gram = orbital_pairs(data)
    np.testing.assert_allclose(gram, np.eye(4), atol=3e-15)
    np.testing.assert_allclose(gram[:2, 2:], 0., atol=3e-15)
    np.testing.assert_allclose(gram[2:, :2], 0., atol=3e-15)
    actual, expected, bare, positive_bare = screened_pair_matrices(data, rho)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-9)
    head = float(api().finite_head(data))
    identity = np.eye(4).reshape(-1)
    np.testing.assert_allclose(bare-positive_bare, head*np.outer(identity, identity), atol=2e-13)
    vv, cc = [i*4+j for i in range(2) for j in range(2)], [a*4+b for a in range(2, 4) for b in range(2, 4)]
    direct = (bare-positive_bare)[np.ix_(vv, cc)].reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    np.testing.assert_allclose(-direct, -head*np.eye(4), atol=2e-13)
    exchange = -np.diag(bare-positive_bare).reshape(4, 4)[:, :2].sum(axis=1)
    np.testing.assert_allclose(exchange, -head*np.array([1., 1., 0., 0.]), atol=2e-13)


def test_discarding_a_nonzero_transition_head_would_change_screening():
    # Caller checks must inspect the same-grid raw OV/VO head, not silently
    # discard it. This counterexample deliberately violates that precondition.
    data = inputs(1)
    rho, gram = orbital_pairs(data, nonorthogonal=True)
    assert abs(gram[0, 2]) > .09
    actual, expected, _, _ = screened_pair_matrices(data, rho)
    assert np.max(abs(actual-expected)) > 1e-6
