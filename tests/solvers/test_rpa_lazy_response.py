"""RPA vector response is attached lazily without changing its metric gauge."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import EigenSolverConfig, LinearSolverConfig, solve_rpa
from gradscf.solvers.eigen import structured_rpa


def model():
    coupling = jnp.array([.03, -.02, .04, .01])
    a = jnp.diag(jnp.array([.5, .9, 1.3, 1.8])) + jnp.outer(coupling, coupling)
    b = jnp.eye(4) * .025 + .3 * jnp.outer(coupling, coupling)
    config = EigenSolverConfig(method="davidson", nroots=2, max_subspace=4,
        atol=1e-11, gradient_mode="implicit_eigenvector")
    linear = LinearSolverConfig(method="direct", rtol=1e-11, max_dense=8)
    return a, b, config, linear


def test_forward_does_not_call_vector_response_linear_solve(monkeypatch):
    a, b, config, linear = model()
    calls = []
    original = structured_rpa.solve_linear

    def observed(*args, **kwargs):
        calls.append(None)  # Do not retain tracers from the staged computation.
        return original(*args, **kwargs)

    monkeypatch.setattr(structured_rpa, "solve_linear", observed)
    result = jax.jit(lambda a, b: solve_rpa(
        a, b, config=config, response_config=linear))(a, b)
    assert np.asarray(result.converged & result.response_valid).all()
    assert np.isfinite(result.x).all() and np.isfinite(result.y).all()
    assert not calls, "Forward executed a vector-response linear solve."
    tangent = jax.jit(lambda t: jax.jvp(lambda s: solve_rpa(
        a + s * (b + .02), b, config=config, response_config=linear).x[0, 1],
        (t,), (jnp.ones_like(t),))[1])(0.)
    assert calls, "AD did not use the shared response linear solver."
    assert np.isfinite(tangent) and abs(float(tangent)) > 1e-9


@pytest.mark.parametrize("method", ["direct", "gmres"])
def test_nested_vector_response_fd_and_adjoint(method):
    a, b, config, linear = model()
    linear = replace(linear, method=method, restart=8, maxiter=50)
    da = jnp.array([[.06, .08, -.03, 0.], [.08, -.02, .04, 0.],
                    [-.03, .04, .1, .02], [0., 0., .02, -.04]])
    probe = jnp.array([.2, -.1, .3, .4])

    def observable(parameters):
        out = solve_rpa(a + parameters[0] * b + parameters[1] * da,
            b + parameters[0] * .03 * a + parameters[1] * .05 * da,
            config=config, response_config=linear)
        return jnp.concatenate((out.values, ((out.x + out.y).T @ probe)**2))

    parameters = jnp.array([.1, -.07])
    direction = jnp.array([.3, -.7])
    cotangent = jnp.array([.2, -.3, .7, -.4])
    inner = jax.jit(observable)
    raw = jax.jit(lambda p: jax.jvp(observable, (p,), (direction,))[1])(parameters)
    forward = jax.jit(lambda p: jax.jvp(inner, (p,), (direction,))[1])(parameters)
    reverse = jax.jit(jax.grad(lambda p: jnp.vdot(cotangent, inner(p))))(parameters)
    finite = (inner(parameters + 1e-4 * direction) - inner(parameters - 1e-4 * direction)) / 2e-4
    assert np.isfinite(forward).all() and np.isfinite(reverse).all()
    np.testing.assert_allclose(forward, raw, atol=2e-11, rtol=0)
    np.testing.assert_allclose(forward, finite, atol=2e-8, rtol=0)
    np.testing.assert_allclose(jnp.vdot(reverse, direction), jnp.vdot(cotangent, forward),
                               atol=2e-10, rtol=0)


@pytest.mark.parametrize("failure", ["asymmetric", "nonfinite", "unstable", "near_degenerate"])
def test_invalid_roots_reject_vector_jvp_and_vjp(failure):
    a, b, config, linear = model()
    if failure == "asymmetric":
        a = a.at[0, 1].add(.02)
    elif failure == "nonfinite":
        a = a.at[0, 0].set(jnp.nan)
    elif failure == "unstable":
        a = a.at[-1, -1].set(-.3)
    else:
        a = jnp.diag(jnp.array([.5, .5 + 1e-10, 1.3, 1.8]))
        b = jnp.zeros_like(a)

    result = solve_rpa(a, b, config=config, response_config=linear)
    assert not np.any(result.response_valid)

    def loss(t):
        out = solve_rpa(a * t, b * t, config=config, response_config=linear)
        return jnp.sum((out.x + out.y)**2)

    # Scaling would give zero vector response for an admissible isolated root.
    # An invalid root must not quietly turn that zero physical tangent into AD=0.
    inner = jax.jit(loss)
    forward = jax.jit(lambda t: jax.jvp(inner, (t,), (jnp.ones_like(t),))[1])(1.)
    reverse = jax.jit(jax.grad(inner))(1.)
    assert not np.isfinite(forward) and not np.isfinite(reverse)


def test_failed_response_solver_keeps_forward_but_rejects_ad():
    a, b, config, linear = model()
    failed = replace(linear, method="gmres", restart=1, maxiter=1, rtol=1e-14, atol=1e-14)
    out = solve_rpa(a, b, config=config, response_config=failed)
    assert np.asarray(out.converged & out.response_valid).all()
    perturb = jnp.ones_like(a) * .04 + jnp.diag(jnp.array([.1, -.2, .3, -.4]))
    probe = jnp.array([.2, -.1, .3, .4])

    def loss(t):
        value = solve_rpa(a + t * perturb, b, config=config, response_config=failed)
        return jnp.sum(((value.x + value.y).T @ probe)**2)

    inner = jax.jit(loss)
    forward = jax.jit(lambda t: jax.jvp(inner, (t,), (jnp.ones_like(t),))[1])(0.)
    reverse = jax.jit(jax.grad(inner))(0.)
    assert not np.isfinite(forward) and not np.isfinite(reverse)


def test_direct_response_retains_two_n_capacity_contract():
    a, b, config, linear = model()
    with pytest.raises(ValueError, match="max_dense"):
        solve_rpa(a, b, config=config, response_config=replace(linear, max_dense=7))
    tangent = jax.jvp(lambda t: solve_rpa(a + t * b, b, config=config,
        response_config=linear).x, (0.,), (1.,))[1]
    assert np.isfinite(tangent).all()
