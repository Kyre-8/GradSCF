"""Explicit CIS response reuses shared solver controls without legacy ambiguity."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import ci
from gradscf.solvers import EigenResponseConfig, LinearSolverConfig


def problem(t):
    # One occupied orbital, two virtuals, zero two-electron interaction.
    # The two excitation energies are the eigenvalues of [[1,t],[t,3]].
    return jnp.array([[-1., 0., 0.], [0., 0., t], [0., t, 2.]]), jnp.zeros((3,)*4)


def test_cis_explicit_direct_response_retains_second_derivative():
    policy = EigenResponseConfig(linear_config=LinearSolverConfig(method="direct", max_dense=2, rtol=1e-12))
    config = ci.CIConfig(nroots=1, solver="dense", conv_tol=1e-12)
    energy = lambda t: ci.solve_cis(*problem(t), nocc=1, config=config, response=policy).excitation_energies[0]
    expected = lambda t: 2.-jnp.sqrt(1.+t*t)
    np.testing.assert_allclose(jax.jit(jax.grad(jax.grad(energy)))(.13),
                               jax.grad(jax.grad(expected))(.13), atol=1e-11, rtol=0)


def test_cis_response_capacity_is_enforced_only_when_response_is_needed():
    policy = EigenResponseConfig(linear_config=LinearSolverConfig(method="direct", max_dense=1, rtol=1e-12))
    config = ci.CIConfig(nroots=1, solver="dense", conv_tol=1e-12)
    energy = lambda t: ci.solve_cis(*problem(t), nocc=1, config=config, response=policy).excitation_energies[0]
    assert np.isfinite(energy(.1))
    assert np.isfinite(jax.jit(jax.grad(energy))(.1))
    with pytest.raises(ValueError, match="max_dense"):
        jax.jit(jax.grad(jax.grad(energy)))(.1)


def test_cis_explicit_eigenpair_target_marks_amplitude_response():
    policy = EigenResponseConfig(target="eigenpairs",
        linear_config=LinearSolverConfig(method="direct", max_dense=2, rtol=1e-12))
    config = ci.CIConfig(nroots=1, solver="dense", conv_tol=1e-12)
    result = ci.solve_cis(*problem(.1), nocc=1, config=config, response=policy)
    assert result.amplitude_response
    coefficient = lambda t: ci.solve_cis(*problem(t), nocc=1, config=config, response=policy).amplitudes[0, 0, 1]**2
    expected = lambda t: .5*(1.-1./jnp.sqrt(1.+t*t))
    np.testing.assert_allclose(jax.jit(jax.grad(coefficient))(.1), jax.grad(expected)(.1), atol=1e-11, rtol=0)


@pytest.mark.parametrize("legacy", [dict(gradient_mode="implicit_eigenvector"),
                                    dict(adjoint_tol=1e-9), dict(adjoint_max_cycle=50)])
def test_cis_explicit_response_rejects_nondefault_legacy_controls(legacy):
    with pytest.raises(ValueError, match="legacy"):
        ci.solve_cis(*problem(.1), nocc=1, config=ci.CIConfig(solver="dense", **legacy),
                     response=EigenResponseConfig())


def test_cis_individual_root_result_rejects_subspace_target():
    with pytest.raises(ValueError, match="subspace"):
        ci.solve_cis(*problem(.1), nocc=1, response=EigenResponseConfig(target="subspace"))


def test_cis_response_policy_has_shared_type():
    with pytest.raises(TypeError, match="EigenResponseConfig"):
        ci.solve_cis(*problem(.1), nocc=1, response={})
