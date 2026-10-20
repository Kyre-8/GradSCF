"""Periodic matrix QPs retain the common branch certificate and continuation."""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers.nonlinear import ScalarRootConfig


query = importlib.import_module('gradscf.gw.pbc.query')


def test_negative_dyson_slope_is_not_promoted_by_a_small_matrix_residual():
    energy = jnp.array([-.4, -.4])
    zero = jnp.zeros((2, 2), dtype=jnp.complex128)
    result = query._solve_subspace_qp(energy, zero, zero,
        lambda omega: 1.2*omega*jnp.eye(2), occupied=True,
        config=ScalarRootConfig(method='hybrid', ftol=1e-10, xtol=1e-10, maxiter=120))
    np.testing.assert_allclose(result['roots'], 2., atol=2e-9)
    assert np.max(np.abs(result['residual'])) < 1e-10
    np.testing.assert_allclose(result['weight'], -5., atol=2e-12)
    assert not np.asarray(result['branch_valid']).any()
    assert not np.asarray(result['converged']).any()
    np.testing.assert_array_equal(result['candidate_count'], [0, 0])


@pytest.mark.parametrize('target', [-1., 1.])
def test_explicit_matrix_continuation_preserves_requested_positive_slope_branch(target):
    energy = jnp.array([.2])
    zero = jnp.zeros((1, 1))
    residual = lambda omega: omega*(omega**2-1)
    sigma = lambda omega: (omega-energy[0]-residual(omega))*jnp.eye(1)
    result = query._solve_subspace_qp(energy, zero, zero, sigma,
        occupied=False, initial_guess=jnp.array([target]),
        config=ScalarRootConfig(method='hybrid', ftol=1e-10, xtol=1e-10, maxiter=80))
    np.testing.assert_allclose(result['roots'], target, atol=1e-10)
    np.testing.assert_allclose(result['weight'], .5, atol=2e-10)
    assert bool(result['converged'][0]) and bool(result['branch_valid'][0])
    assert int(result['candidate_count'][0]) >= 1
