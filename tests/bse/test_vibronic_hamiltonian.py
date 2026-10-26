"""Finite vibrational-space oracles, using the shared eigensolver."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import factorial

from gradscf.bse import ep_coupling as ep
from gradscf.solvers import (
    EigenSolverConfig,
    LinearSolverConfig,
    solve_hermitian,
    solve_complex,
)


def lines(h, model, d, cutoff):
    matrix, moments = ep.vibronic_hamiltonian(h, model, d, max_quanta=cutoff)
    result = solve_hermitian(
        matrix,
        config=EigenSolverConfig(
            method="dense",
            nroots=matrix.shape[0],
            max_dense=matrix.shape[0],
            atol=1e-10,
        ),
    )
    assert np.all(result.converged)
    return result.values, result.vectors.T @ moments


def test_displaced_oscillator_matches_poisson_progression():
    e, w, g = 0.4, 0.02, 0.024
    model = ep.PhononModel(jnp.array([w]), jnp.array([[[g]]]))
    energies, moments = lines(jnp.array([e]), model, jnp.array([[1.0, 0.0, 0.0]]), 32)
    n = np.arange(12)
    s = (g / w) ** 2
    np.testing.assert_allclose(energies[:12], e - g * g / w + n * w, atol=1e-10)
    np.testing.assert_allclose(
        moments[:12, 0] ** 2, np.exp(-s) * s**n / factorial(n), atol=2e-9
    )
    np.testing.assert_allclose(jnp.sum(moments[:, 0] ** 2), 1.0, atol=1e-12)


def test_two_mode_progression_and_zero_coupling():
    w = jnp.array([0.02, 0.031])
    g = jnp.array([[[0.01]], [[0.008]]])
    model = ep.PhononModel(w, g)
    energies, moments = lines(jnp.array([0.4]), model, jnp.array([[1.0, 0.0, 0.0]]), 12)
    s = np.asarray((g[:, 0, 0] / w) ** 2)
    np.testing.assert_allclose(
        energies[0], 0.4 - jnp.sum(g[:, 0, 0] ** 2 / w), atol=1e-11
    )
    np.testing.assert_allclose(moments[0, 0] ** 2, np.exp(-s.sum()), atol=1e-10)
    _, bare = lines(
        jnp.array([0.4]),
        replace(model, couplings=g * 0),
        jnp.array([[1.0, 0.0, 0.0]]),
        3,
    )
    np.testing.assert_allclose(bare[0, 0] ** 2, 1.0)
    np.testing.assert_allclose(jnp.sum(bare[1:] ** 2), 0.0)


def test_quadratic_boundary_keeps_virtual_intermediate_states():
    model = ep.PhononModel(
        jnp.array([0.02]), jnp.zeros((1, 1, 1)), jnp.array([[[0.006]]])
    )
    matrix, _ = ep.vibronic_hamiltonian(
        jnp.array([0.4]), model, jnp.zeros((1, 3)), max_quanta=1
    )
    np.testing.assert_allclose(jnp.diag(matrix), [0.403, 0.429], atol=1e-14)


def test_cross_mode_quadratic_and_complex_frame_covariance():
    h = jnp.array([[0.4, 0.01j], [-0.01j, 0.5]])
    g = jnp.array([[[0.01, 0.002j], [-0.002j, -0.02]], [[0.003, 0.004], [0.004, 0.01]]])
    quadratic = jnp.zeros((2, 2, 2, 2)).at[0, 1].set(jnp.eye(2) * 0.006)
    quadratic = quadratic.at[1, 0].set(jnp.eye(2) * 0.006)
    model = ep.PhononModel(jnp.array([0.02, 0.03]), g, quadratic)
    d = jnp.ones((2, 3), dtype=complex)
    matrix, moments = ep.vibronic_hamiltonian(h, model, d, max_quanta=2)
    np.testing.assert_allclose(matrix, matrix.conj().T)
    # Basis order is vacuum, (0,1), (1,0), ... .
    np.testing.assert_allclose(matrix[2:4, 4:6], jnp.eye(2) * 0.006)
    u = jnp.array([[1.0, 1j], [1j, 1.0]]) / jnp.sqrt(2.0)
    transformed = replace(
        model,
        couplings=jnp.einsum("is,lij,jt->lst", u.conj(), g, u),
        quadratic=jnp.einsum("is,lmij,jt->lmst", u.conj(), quadratic, u),
    )
    rotated, rd = ep.vibronic_hamiltonian(
        u.conj().T @ h @ u, transformed, u.conj().T @ d, max_quanta=2
    )
    frame = jnp.kron(jnp.eye(6), u)
    np.testing.assert_allclose(rotated, frame.conj().T @ matrix @ frame, atol=1e-14)
    np.testing.assert_allclose(rd, frame.conj().T @ moments, atol=1e-14)


def test_matrix_assembly_is_jittable_and_differentiable_at_zero_coupling():
    model = ep.PhononModel(jnp.array([0.02]), jnp.zeros((1, 1, 1)))

    def objective(g):
        matrix, _ = ep.vibronic_hamiltonian(
            jnp.array([0.4]),
            replace(model, couplings=jnp.reshape(g, (1, 1, 1))),
            jnp.zeros((1, 3)),
            max_quanta=2,
        )
        return jnp.sum(matrix**2)

    value, grad = jax.jit(jax.value_and_grad(objective))(0.01)
    np.testing.assert_allclose(grad, 0.12)
    np.testing.assert_allclose(jax.grad(jax.grad(objective))(0.0), 12.0)
    assert np.isfinite(value)


def test_symmetry_blocks_preserve_vacuum_optical_response():
    w = jnp.array([0.02, 0.031, 0.035])
    g = jnp.zeros((3, 3, 3)).at[0].set(jnp.diag(jnp.array([0.01, 0.015, 0.02])))
    g = g.at[1].set(jnp.diag(jnp.array([0.008, 0.01, 0.013])))
    g = g.at[2, 0, 1].set(0.005).at[2, 1, 0].set(0.005)
    h, d = jnp.array([0.4, 0.46, 0.5]), jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )

    def response(h, model, d):
        matrix, moments = ep.vibronic_hamiltonian(h, model, d, max_quanta=4)
        result = solve_complex(
            (0.405 + 0.002j) * jnp.eye(matrix.shape[0]) - matrix,
            moments,
            config=LinearSolverConfig(method="direct", rtol=1e-11),
        )
        assert bool(result.converged)
        return moments.T @ result.solution

    full = response(h, ep.PhononModel(w, g), d)
    blocks = response(h[:2], ep.PhononModel(w, g[:, :2, :2]), d[:2]) + response(
        h[2:], ep.PhononModel(w[:2], g[:2, 2:, 2:]), d[2:]
    )
    np.testing.assert_allclose(full, blocks, rtol=1e-12, atol=1e-12)


def test_vibronic_resolvent_gradient_matches_finite_difference():
    model = ep.PhononModel(jnp.array([0.02, 0.031]), jnp.array([[[0.01]], [[0.008]]]))

    def loss(scale):
        matrix, moments = ep.vibronic_hamiltonian(
            jnp.array([0.4]),
            replace(model, couplings=model.couplings * scale),
            jnp.array([[1.0, 0.0, 0.0]]),
            max_quanta=4,
        )
        result = solve_complex(
            (0.405 + 0.002j) * jnp.eye(matrix.shape[0]) - matrix,
            moments,
            config=LinearSolverConfig(method="direct", rtol=1e-11),
        )
        return -(moments.T @ result.solution)[0, 0].imag

    value, gradient = jax.jit(jax.value_and_grad(loss))(1.0)
    difference = (loss(1.0 + 1e-6) - loss(1.0 - 1e-6)) / (2e-6)
    np.testing.assert_allclose(gradient, difference, rtol=1e-7)
    assert np.isfinite(value)


@pytest.mark.parametrize("cutoff", [-1, True, 1.5])
def test_invalid_cutoff_rejected(cutoff):
    model = ep.PhononModel(jnp.array([0.02]), jnp.zeros((1, 1, 1)))
    with pytest.raises((ValueError, TypeError)):
        ep.vibronic_hamiltonian(
            jnp.array([0.4]), model, jnp.zeros((1, 3)), max_quanta=cutoff
        )
