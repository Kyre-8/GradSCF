"""Independent small dense checks for the high-dimensional A03 model."""
import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def demonstration():
    path = Path(__file__).resolve().parents[2] / "examples/solvers/degenerate_vjp.py"
    assert path.is_file(), "The matrix-free A03 VJP demonstration is not implemented"
    spec = importlib.util.spec_from_file_location("a03_degenerate_vjp", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_structured_operator_matches_known_spectrum_and_diagonal(demonstration):
    model = demonstration.DegenerateModel(32, rank=4, seed=1703)
    parameters = jnp.array([.01, -.015, .005, .02])
    op = model.operator(parameters, internal_gap=0., external_gap=.7)
    matrix = np.asarray(op.apply(jnp.eye(model.dimension)))
    np.testing.assert_allclose(matrix, matrix.T, atol=2e-14, rtol=0)
    np.testing.assert_allclose(np.diag(matrix), op.diagonal, atol=2e-14, rtol=0)
    zero = model.operator(jnp.zeros(4), internal_gap=0., external_gap=.7)
    np.testing.assert_allclose(np.linalg.eigvalsh(zero.apply(jnp.eye(32))),
                               model.spectrum(0., .7), atol=2e-14, rtol=0)
    rng = np.random.default_rng(19)
    vectors = jnp.asarray(rng.normal(size=(32, 3)))
    np.testing.assert_allclose(model.rotate(model.rotate(vectors), transpose=True),
                               vectors, atol=2e-14, rtol=0)


@pytest.mark.parametrize("internal_gap", [0., 1e-7])
def test_analytic_projector_vjp_matches_independent_dense_differences(demonstration, internal_gap):
    model = demonstration.DegenerateModel(32, rank=4, seed=1703)
    expected = model.analytic_reference(internal_gap, .7)
    direction = np.array([.3, -.2, .4, .1])
    direction /= np.linalg.norm(direction)
    probes, weights = np.asarray(model.probes), np.asarray(model.weights)

    def oracle(t):
        op = model.operator(jnp.asarray(t*direction), internal_gap=internal_gap,
                            external_gap=.7)
        matrix = np.asarray(op.apply(jnp.eye(model.dimension)))
        _, vectors = np.linalg.eigh(matrix)
        amplitudes = vectors[:, :model.rank].T @ probes
        return np.sum(weights*(amplitudes.T @ amplitudes))

    for step in (1e-4, 3e-5):
        difference = (oracle(step)-oracle(-step))/(2*step)
        np.testing.assert_allclose(difference, np.asarray(expected["vjp"])@direction,
                                   atol=3e-8, rtol=0)


@pytest.mark.parametrize("method", ["dense", "davidson"])
def test_shared_subspace_vjp_uses_complete_exact_cluster(demonstration, method):
    model = demonstration.DegenerateModel(32, rank=4, seed=1703)
    solve = demonstration.make_solve(model, method=method, max_subspace=24,
                                      eig_tol=1e-11, response_tol=1e-11,
                                      restart=24, maxiter=100)
    zero = jnp.zeros(4)
    observables = lambda p: demonstration.observables(model, solve(p, 0., .7))
    values, pullback = jax.vjp(observables, zero)
    actual = pullback(jnp.array([1., 0.]))[0]
    reference = model.analytic_reference(0., .7)
    np.testing.assert_allclose(values[0], reference["loss"], atol=2e-9, rtol=0)
    np.testing.assert_allclose(actual, reference["vjp"], atol=2e-8, rtol=0)
    direction = jnp.array([.3, -.2, .4, .1])
    tangent = jax.jvp(observables, (zero,), (direction,))[1]
    np.testing.assert_allclose(tangent[0], actual@direction, atol=2e-10, rtol=0)
