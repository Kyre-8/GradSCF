"""Signed finite-eta CD cached solves against the original transition action."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import LinearSolverConfig
from gradscf.solvers.linear.shifted import factor_shifted, solve_shifted
from gradscf.gw.pbc.product_basis import lehmann_weights
from gradscf.gw.screened import solve_transition_screening


def _block():
    rng = np.random.default_rng(62)
    vertex = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    gram = jnp.asarray(.03*vertex.conj().T@vertex)
    projected = jnp.asarray(.1*(rng.normal(size=(4, 2, 2))+1j*rng.normal(size=(4, 2, 2))))
    return dict(gram=gram, differences=jnp.array([-1.25, 1.25, -1.25, 1.25]),
        signs=jnp.array([1., -1., 1., -1.]), projected=projected, q_weight=.5)


@pytest.mark.parametrize('frequency,eta', [(1.1, .003), (-.7, .003), (1.25, 0.), (-1.25, 0.)])
def test_cached_signed_transition_matches_complex_matrix_and_frequency_jvp(frequency, eta):
    block, nk = _block(), 2
    gram, delta, signs = (block[name] for name in ('gram', 'differences', 'signs'))
    rhs = block['projected'][:, 0]
    matrix = -jnp.diag(delta)-1j*eta*jnp.diag(signs)+2/nk*signs[:, None]*gram
    state = factor_shifted(matrix, symmetric=False)
    def cached(value):
        z = value+1j*eta*jnp.where(value >= 0, 1., -1.)
        return solve_shifted(state, -2/nk*signs[:, None]*rhs, z)
    def exact(value):
        if eta == 0 and abs(abs(frequency)-1.25) < 1e-12:
            # The original weight quotient is undefined at a bare pole. Its
            # division-free linear limit still has a finite checked solution.
            inverse = nk*(value+delta)/(2*signs)
            return jnp.linalg.solve(jnp.diag(inverse)-gram, rhs)
        return solve_transition_screening(gram, lehmann_weights(value, delta, signs, nk, eta=eta),
            rhs, config=LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13)).solution
    value = jnp.asarray(frequency)
    np.testing.assert_allclose(cached(value), exact(value), atol=3e-12, rtol=2e-11)
    np.testing.assert_allclose(jax.jvp(cached, (value,), (jnp.ones_like(value),))[1],
        jax.jvp(exact, (value,), (jnp.ones_like(value),))[1], atol=3e-11, rtol=2e-10)


@pytest.mark.parametrize('omega,eta', [(-.9, .003), (.9, .003), (1.75, 0.)])
def test_cd_residue_uses_cached_factor_without_changing_full_matrix(monkeypatch, omega, eta):
    query = importlib.import_module('gradscf.gw.pbc.query')
    block, nk = _block(), 2
    energy, ef = jnp.array([-.5, .5]), 0.
    config = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13)
    expected = query._sigma_residue_full(jnp.asarray(omega), energy, [block], ef, eta, nk, config)
    matrix = (-jnp.diag(block['differences'])-1j*eta*jnp.diag(block['signs'])
              +2/nk*block['signs'][:, None]*block['gram'])
    cached_block = dict(block, shifted_factor=factor_shifted(matrix, symmetric=False))
    def forbidden(*args, **kwargs):
        raise AssertionError('A cached q block must not repeat transition factorization')
    monkeypatch.setattr(query, 'solve_transition_screening', forbidden)
    actual = query._sigma_residue_full(jnp.asarray(omega), energy, [cached_block], ef, eta, nk, config)
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=2e-11)
