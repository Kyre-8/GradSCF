"""A QP root exposes the same implicit equation to JVP, VJP and mixed AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw.qp import solve_qp_batch
from test_gw_qp import _context


def root(parameters, method="secant"):
    shared, stacked = _context()
    result, _ = solve_qp_batch(
        jnp.array([-.5+parameters[0]]), jnp.array([parameters[1]]), shared,
        {**stacked, "wmn_p": (1+parameters[2])*stacked["wmn_p"]},
        occupied=jnp.array([True]), tol=1e-11, method=method)
    return result[0]


@pytest.mark.parametrize("method", ["secant", "newton", "hybrid"])
def test_qp_jvp_vjp_and_reconverged_differences(method):
    parameters = jnp.array([.002, -.001, .04])
    direction = jnp.array([.2, -.3, .4])
    value = lambda p: root(p, method)
    primal, tangent = jax.jit(lambda p: jax.jvp(value, (p,), (direction,)))(parameters)
    reverse = jax.jit(jax.grad(value))(parameters)
    np.testing.assert_allclose(tangent, reverse@direction, atol=1e-11, rtol=0)
    for step in (1e-3, 3e-4, 1e-4):
        finite = (value(parameters+step*direction)-value(parameters-step*direction))/(2*step)
        np.testing.assert_allclose(tangent, finite, atol=2e-8, rtol=0)


def test_qp_hessian_vector_uses_the_moving_root():
    parameters = jnp.array([.002, -.001, .04])
    direction = jnp.array([.2, -.3, .4])
    gradient = jax.jit(jax.grad(root))
    hvp = jax.jit(lambda p: jax.jvp(jax.grad(root), (p,), (direction,))[1])(parameters)
    assert np.linalg.norm(np.asarray(hvp)) > 1e-6
    for step in (1e-3, 3e-4, 1e-4):
        finite = (gradient(parameters+step*direction)-gradient(parameters-step*direction))/(2*step)
        np.testing.assert_allclose(hvp, finite, atol=2e-8, rtol=0)


def test_qp_real_parameter_response_with_complex_conjugate_vertices():
    shared, stacked = _context()
    shared = {**shared, "conjugate": True,
              "channels": ((shared["mo_energy"], jnp.array([[[.13+.05j]]]), 2.),)}
    vertices = jnp.array([[[.1+.2j, .07-.04j]]])
    def run(scale):
        moving = {**stacked, "b_pm": scale*vertices, "b_mp": scale*vertices}
        return solve_qp_batch(jnp.array([-.6]), jnp.zeros(1), shared, moving,
                              occupied=jnp.array([True]), tol=1e-11)
    assert bool(jnp.all(run(1.)[1]))
    energy = lambda scale: run(scale)[0].sum()
    tangent = jax.jit(lambda t: jax.jvp(energy, (t,), (jnp.ones_like(t),))[1])(jnp.array(1.))
    reverse = jax.jit(jax.grad(energy))(1.)
    np.testing.assert_allclose(tangent, reverse, atol=1e-11, rtol=0)
    step = 1e-4
    np.testing.assert_allclose(tangent, (energy(1+step)-energy(1-step))/(2*step), atol=2e-8, rtol=0)


def test_explicit_qp_seed_preserves_the_root_response_not_the_seed_tangent():
    shared, stacked = _context()
    def energy(seed, reference):
        result, _ = solve_qp_batch(jnp.atleast_1d(reference), jnp.zeros(1), shared, stacked,
            occupied=jnp.array([True]), initial_guess=jnp.atleast_1d(seed), tol=1e-11,
            method="newton")
        return result.sum()
    value, (dseed, dphysical) = jax.jit(jax.value_and_grad(energy, argnums=(0, 1)))(-.61, -.6)
    np.testing.assert_allclose(dseed, 0., atol=0., rtol=0)
    np.testing.assert_allclose(value, energy(-.59, -.6), atol=1e-11, rtol=0)
    step = 1e-4
    finite = (energy(-.61, -.6+step)-energy(-.61, -.6-step))/(2*step)
    np.testing.assert_allclose(dphysical, finite, atol=2e-8, rtol=0)


def test_g0w0_exposes_qp_precision_without_changing_screening():
    from gradscf.gw import g0w0_cd_restricted
    from test_gw_differentiability import _h2_inputs
    inputs = _h2_inputs()
    result = g0w0_cd_restricted(**inputs, nw=40, qp_solver="newton", qp_tol=1e-11,
                               qp_initial=inputs["mo_energy"])
    assert bool(result.converged)
    assert np.max(np.abs(result.qp_residual)) < 1e-11
    np.testing.assert_array_equal(result.screening_energy, inputs["mo_energy"])
    for invalid in (0., -1e-6, float("nan")):
        with pytest.raises(ValueError, match="qp_tol"):
            g0w0_cd_restricted(**inputs, qp_tol=invalid)
