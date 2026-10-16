"""Exact complex transition compression against independent dense dielectric algebra.

CPU float64/complex128. This tests the reduced action, not periodic-GW driver
integration, auxiliary-rank truncation, q streaming, or material convergence.
"""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import LinearSolverConfig


def kernels():
    module = importlib.import_module("gradscf.gw.screened")
    assert hasattr(module, "solve_transition_screening"), "Complex transition screening action is not implemented"
    assert hasattr(module, "transition_head_wing"), "Compressed head/wing contraction is not implemented"
    return module


def model():
    rng = np.random.default_rng(827)
    factors = .07*(rng.normal(size=(23, 5))+1j*rng.normal(size=(23, 5)))
    # A repeated column is retained exactly; no inverse Gram or rank truncation.
    factors[:, -1] = factors[:, 0]
    rhs = .1*(rng.normal(size=(23, 7))+1j*rng.normal(size=(23, 7)))
    qij = 1e-4*(rng.normal(size=5)+1j*rng.normal(size=5))
    return factors, rhs, qij


def weights(axis, omega=.41, eta=.03, nk=1):
    gap = np.array([.7, 1.1, 1.4, 1.8, .7])
    if axis == "imaginary":
        return -4*gap/(omega**2+gap**2)/nk
    # Preserve rho_response_real's asymmetric 2i*eta convention exactly.
    return 2*(1/(omega-gap+2j*eta)-1/(omega+gap))/nk


@pytest.mark.parametrize("axis", ["imaginary", "retarded"])
@pytest.mark.parametrize("method", ["direct", "gmres"])
@pytest.mark.parametrize("zero", [False, True])
def test_reduced_action_matches_dense_with_dependent_columns(axis, method, zero):
    module = kernels()
    factors, rhs, _ = model()
    value = weights(axis).astype(complex)
    if zero:
        value[[1, 3]] = 0.
    gram, projected = factors.conj().T@factors, factors.conj().T@rhs
    config = LinearSolverConfig(method=method, rtol=1e-12, atol=1e-13, restart=12, maxiter=20)
    result = module.solve_transition_screening(jnp.asarray(gram), jnp.asarray(value), jnp.asarray(projected), config=config)
    assert bool(result.converged)
    dielectric = np.eye(len(factors))-(factors*value)@factors.conj().T
    actual = factors@np.asarray(result.solution)
    expected = np.linalg.solve(dielectric, rhs)-rhs
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=2e-11)
    np.testing.assert_allclose(np.sum(projected.conj()*result.solution, axis=0),
                               np.sum(rhs.conj()*expected, axis=0), atol=3e-12, rtol=2e-11)
    # The diagnostic belongs to the actual shared reduced solve.
    residual = (np.eye(len(value))-value[:, None]*gram)@np.asarray(result.solution)-value[:, None]*projected
    assert np.linalg.norm(residual) <= 2e-11
    assert np.linalg.norm(dielectric@(rhs+actual)-rhs) <= 2e-11


def test_zero_weights_return_zero_screening_without_weight_inverse():
    module = kernels()
    factors, rhs, _ = model()
    result = module.solve_transition_screening(factors.conj().T@factors, np.zeros(5), factors.conj().T@rhs)
    assert bool(result.converged)
    np.testing.assert_array_equal(result.solution, np.zeros((5, 7)))


def dense_head_wing(factors, value, qij, diagonal, q_abs, volume, nk):
    inverse = np.linalg.inv(np.eye(len(factors))-(factors*value)@factors.conj().T)
    factor = np.sqrt(4*np.pi)/np.linalg.norm(q_abs)
    eps00 = 1-factor**2*np.sum(value*qij.conj()*qij)
    wing = -factor*(factors@(value*qij.conj()))
    schur = eps00-wing.conj()@inverse@wing
    inverse00, inversewing = 1/schur, -(inverse@wing)/schur
    head_scale = 2/np.pi*(6*np.pi**2/volume/nk)**(1/3)
    wing_scale = np.sqrt(volume/(4*np.pi**3))*(6*np.pi**2/volume/nk)**(2/3)
    return head_scale*(inverse00-1), 2*wing_scale*np.real(diagonal.conj().T@inversewing), schur


@pytest.mark.parametrize("axis", ["imaginary", "retarded"])
@pytest.mark.parametrize("nk", [1, 3])
def test_head_wing_keeps_retarded_conjugation_and_nk_factors(axis, nk):
    module = kernels()
    factors, rhs, qij = model()
    value = weights(axis, nk=nk)
    gram, projected = factors.conj().T@factors, factors.conj().T@rhs
    q_abs, volume = np.array([1e-3, 0., 0.]), 93.
    actual = module.transition_head_wing(gram, value, qij, projected, q_abs, volume, nkpts=nk)
    expected = dense_head_wing(factors, value, qij, rhs, q_abs, volume, nk)
    assert bool(actual.converged)
    for observed, wanted in zip((actual.head, actual.wings, actual.schur), expected):
        np.testing.assert_allclose(observed, wanted, atol=2e-12, rtol=2e-11)


@pytest.mark.parametrize("axis", ["imaginary", "retarded"])
def test_transition_phase_gauge_and_output_window(axis):
    module = kernels()
    factors, rhs, qij = model()
    value = weights(axis)
    gram, projected = factors.conj().T@factors, factors.conj().T@rhs
    baseline = module.solve_transition_screening(gram, value, projected)
    phase = np.exp(1j*np.array([.2, -.8, .4, .7, -.3]))
    rotated = module.solve_transition_screening(phase.conj()[:, None]*gram*phase[None, :],
        value, phase.conj()[:, None]*projected)
    np.testing.assert_allclose(rotated.solution, phase.conj()[:, None]*baseline.solution, atol=2e-12)
    selected = np.array([1, 4, 6])
    window = module.solve_transition_screening(gram, value, projected[:, selected])
    np.testing.assert_allclose(window.solution, np.asarray(baseline.solution)[:, selected], atol=2e-12)
    arguments = (np.array([1e-3, 0., 0.]), 93.)
    head = module.transition_head_wing(gram, value, qij, projected, *arguments)
    changed = module.transition_head_wing(phase.conj()[:, None]*gram*phase[None, :], value,
        phase*qij, phase.conj()[:, None]*projected, *arguments)
    np.testing.assert_allclose(changed.head, head.head, atol=2e-12)
    np.testing.assert_allclose(changed.wings, head.wings, atol=2e-12)


@pytest.mark.parametrize("eta", [.03, .003, .0003])
def test_retarded_action_near_a_screened_pole_preserves_finite_eta(eta):
    module = kernels()
    factors = np.array([[.2], [.1j], [.15]], dtype=complex)
    gram = factors.conj().T@factors
    omega = np.sqrt(1+4*gram[0, 0].real)
    value = np.array([2*(1/(omega-1+2j*eta)-1/(omega+1))])
    rhs = np.array([[.3+.1j], [.2], [-.1j]])
    result = module.solve_transition_screening(gram, value, factors.conj().T@rhs)
    expected = np.linalg.solve(np.eye(3)-(factors*value)@factors.conj().T, rhs)-rhs
    assert bool(result.converged)
    np.testing.assert_allclose(factors@result.solution, expected, atol=2e-10, rtol=2e-10)


def test_shared_complex_solver_receives_only_transition_dimension(monkeypatch):
    module = kernels()
    original = module.solve_complex
    seen = []
    def checked(operator, rhs, **kwargs):
        seen.append((operator.shape, np.shape(rhs)))
        return original(operator, rhs, **kwargs)
    monkeypatch.setattr(module, "solve_complex", checked)
    factors, rhs, _ = model()
    result = module.solve_transition_screening(factors.conj().T@factors, weights("retarded"), factors.conj().T@rhs)
    assert bool(result.converged)
    assert seen == [((5, 5), (5, 7))]


def test_zero_body_transition_factors_do_not_drop_nonzero_q0_head():
    module = kernels()
    _, rhs, qij = model()
    factors = np.zeros((23, 5), dtype=complex)
    value = weights("retarded", nk=3)
    q_abs, volume = np.array([1e-3, 0., 0.]), 93.
    actual = module.transition_head_wing(np.zeros((5, 5)), value, qij, np.zeros((5, 7)), q_abs, volume, nkpts=3)
    expected = dense_head_wing(factors, value, qij, rhs, q_abs, volume, 3)
    assert bool(actual.converged)
    assert abs(actual.head) > 1e-4
    np.testing.assert_allclose(actual.head, expected[0], atol=2e-12, rtol=2e-11)
    np.testing.assert_array_equal(actual.wings, np.zeros(7))


@pytest.mark.parametrize("nk", [1, 3])
def test_retarded_compression_matches_existing_q0_residue_convention(nk):
    module = kernels()
    from gradscf.gw.pbc.q0 import q0_residue_correction
    factors, rhs, qij = model()
    occupied = np.array([-.5])
    virtual = occupied[0]+np.array([.7, 1.1, 1.4, 1.8, .7])
    channels = tuple(((jnp.asarray(occupied), jnp.asarray(virtual)),
                      jnp.asarray(factors[:, None, :]), 2.) for _ in range(nk))
    dipoles = tuple(jnp.asarray(qij[None, :]) for _ in range(nk))
    q_abs, volume = np.array([1e-3, 0., 0.]), 93.
    repeated = np.tile(factors, (1, nk))
    actual = module.transition_head_wing(repeated.conj().T@repeated,
        np.tile(weights("retarded", nk=nk), nk), np.tile(qij, nk),
        repeated.conj().T@rhs[:, 2], q_abs, volume, nkpts=nk)
    expected = q0_residue_correction(.41, jnp.asarray(rhs[:, 2]), channels,
        dipoles, jnp.asarray(q_abs), volume, .03, nkpts=nk)
    assert bool(actual.converged)
    np.testing.assert_allclose(actual.head+actual.wings, expected, atol=2e-12, rtol=2e-11)
