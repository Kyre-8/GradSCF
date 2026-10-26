"""QP solver status and AD regressions (CPU, float64; energies in Ha)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers.nonlinear import ScalarRootConfig, solve_scalar_roots

from gradscf.gw.qp import (
    solve_qp_batch,
    solve_qp_orbital,
)


def _context():
    energy = jnp.array([-0.5, 0.5])
    shared = dict(
        mo_energy=energy,
        channels=((energy, jnp.zeros((1, 1, 1)), 2.0),),
        ef=jnp.array(0.0),
        eta=jnp.array(1e-3),
        freqs=jnp.array([1.0]),
        wts=jnp.array([1.0]),
        conjugate=False,
    )
    stacked = dict(
        wmn_p=jnp.array([[[-0.02, -0.01]]]),
        b_pm=jnp.zeros((1, 1, 2)),
        b_mp=jnp.zeros((1, 1, 2)),
    )
    return shared, stacked


@pytest.mark.parametrize("method", ["secant", "newton", "hybrid"])
@pytest.mark.parametrize("mode", ["implicit", "unrolled"])
def test_jit_preserves_nonconvergence_with_custom_iteration_limit(mode,method):
    shared, stacked = _context()

    def solve(e):
        return solve_qp_batch(
            e, jnp.zeros(1), shared, stacked, occupied=jnp.array([True]),
            maxiter=1, tol=1e-12, diff_mode=mode, method=method,
        )

    eager = solve(jnp.array([-0.5]))
    compiled = jax.jit(solve)(jnp.array([-0.5]))
    assert not bool(eager[1][0])
    np.testing.assert_array_equal(compiled[1], eager[1])
    np.testing.assert_allclose(compiled[0], eager[0], rtol=0, atol=1e-14)


def test_orbital_wrapper_preserves_nonconvergence_under_jit():
    shared, stacked = _context()
    ctx = {**shared, **{key: value[0] for key, value in stacked.items()}}
    solve = jax.jit(lambda e: solve_qp_orbital(
        e, 0.0, ctx, occupied=True, maxiter=1, tol=1e-12, diff_mode="unrolled"
    ))
    _, done = solve(-0.5)
    assert not bool(done)


@pytest.mark.parametrize("method", ["secant", "newton", "hybrid"])
def test_context_only_implicit_gradient_matches_finite_difference(method):
    shared, stacked = _context()

    def energy(scale):
        return solve_qp_batch(
            jnp.array([-0.5]), jnp.zeros(1), shared,
            {**stacked, "wmn_p": scale * stacked["wmn_p"]},
            occupied=jnp.array([True]), tol=1e-10, method=method,
        )[0][0]

    grad = jax.jit(jax.grad(energy))(1.0)
    step = 1e-4
    fd = (energy(1.0 + step) - energy(1.0 - step)) / (2 * step)
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("scan", [False, True])
def test_small_step_without_small_residual_is_not_converged(scan):
    # A jump can yield a tiny secant step although no root exists.
    fn = lambda x: jnp.where(x > 0.0, 1e12, 1.0)
    out = solve_scalar_roots(fn, jnp.array([1.0]), x1=jnp.array([0.0]), config=ScalarRootConfig(maxiter=3), scan=scan)
    root, done = out.roots, out.converged
    assert not bool(done[0])
    assert np.isfinite(root).all()


def test_unconverged_implicit_gradient_raises():
    shared, stacked = _context()

    def energy(e):
        return solve_qp_batch(
            e, jnp.zeros(1), shared, stacked, occupied=jnp.array([True]),
            maxiter=1, tol=1e-12,
        )[0].sum()

    # JAX wraps host callback exceptions under jit; the diagnostic must survive.
    with pytest.raises(Exception, match="QP implicit differentiation.*not converged"):
        jax.jit(jax.grad(energy))(jnp.array([-0.5])).block_until_ready()


def test_singular_implicit_gradient_raises_under_jit():
    shared, stacked = _context()
    poles = jnp.array([-0.51, 0.5])
    shared = {**shared, "mo_energy": poles, "channels": ((poles, jnp.zeros((1, 1, 1)), 2.0),)}
    eta = shared["eta"]
    # At omega == poles[0], this one-frequency self-energy has dSigma/dw=1.
    weight = -jnp.pi * (1.0 - eta**2)**2 / (1.0 + eta**2)
    stacked = {**stacked, "wmn_p": jnp.array([[[weight, 0.0]]])}

    def energy(e):
        return solve_qp_batch(
            e, jnp.array([-0.01]), shared, stacked, occupied=jnp.array([True])
        )[0].sum()

    # A root can be reportable even if its implicit derivative is singular.
    np.testing.assert_allclose(energy(jnp.array([-0.5])), -0.51, rtol=0, atol=1e-8)
    np.testing.assert_allclose(jax.jit(energy)(jnp.array([-0.5])), -0.51, rtol=0, atol=1e-8)
    with pytest.raises(Exception, match="QP implicit differentiation is singular"):
        jax.jit(jax.grad(energy))(jnp.array([-0.5])).block_until_ready()
