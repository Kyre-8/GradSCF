"""Isolated left/right spectral response, independent of EOM contractions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import (
    LinearOperator, LinearSolverConfig, NonHermitianSolverConfig,
    solve_nonhermitian,
)


def family(t):
    frame = jnp.array([[1.0, .3, -.2], [.2, 1.0, .4], [-.1, .2, 1.0]])
    change = jnp.array([[.1, -.2, .3], [.2, -.1, .1], [.3, .2, -.2]])
    frame = frame + t * change
    energies = jnp.array([.4 + .2*t + .1*t*t, 1.3 - .3*t, 2.5 + .1*t])
    return frame @ jnp.diag(energies) @ jnp.linalg.inv(frame)


def solved(t, *, method="direct", max_dense=4, operator=False, max_memory=None):
    a = family(t)
    source = (LinearOperator(a.shape, a.dtype, lambda v: a @ v)
              if operator else a)
    return solve_nonhermitian(source,
        config=NonHermitianSolverConfig(nroots=2, atol=1e-11),
        vector_response=True,
        response_config=LinearSolverConfig(method=method, rtol=1e-11,
                                          restart=4, max_dense=max_dense),
        max_memory=max_memory)


def observable(t, **kwargs):
    result = solved(t, **kwargs)
    # A residue is invariant under either eigenpair's arbitrary sign/scale.
    c = jnp.array([.7, -.2, .5])
    b = jnp.array([.3, .8, -.1])
    weights = (c @ result.right_vectors) * (result.left_vectors.T @ b)
    return jnp.concatenate((result.values, weights))


@pytest.mark.parametrize("method", ["direct", "gmres"])
@pytest.mark.parametrize("operator", [False, True])
def test_residue_jvp_vjp_and_reconverged_difference(method, operator):
    function = lambda t: observable(t, method=method, operator=operator)
    value, tangent = jax.jit(lambda t: jax.jvp(function, (t,), (jnp.ones_like(t),)))(.1)
    h = 1e-5
    difference = (function(.1+h)-function(.1-h))/(2*h)
    np.testing.assert_allclose(tangent, difference, atol=2e-9, rtol=2e-8)
    cotangent = jnp.array([.2, -.3, .7, -.5])
    reverse = jax.jit(jax.grad(lambda t: cotangent @ function(t)))(.1)
    np.testing.assert_allclose(reverse, cotangent @ tangent, atol=2e-11)
    assert np.all(np.isfinite(value))


def test_vectors_obey_differentiated_equations_and_normalization():
    def arrays(t):
        out = solved(t)
        return out.values, out.right_vectors, out.left_vectors
    (w, r, l), (dw, dr, dl) = jax.jvp(arrays, (.1,), (1.,))
    a, da = jax.jvp(family, (.1,), (1.,))
    np.testing.assert_allclose(da @ r+a @ dr-dr*w-r*dw, 0., atol=2e-11)
    np.testing.assert_allclose(da.T @ l+a.T @ dl-dl*w-l*dw, 0., atol=2e-11)
    np.testing.assert_allclose(jnp.sum(r*dr, axis=0), 0., atol=2e-11)
    np.testing.assert_allclose(dl.T @ r+l.T @ dr, 0., atol=2e-11)
    assert np.linalg.norm(dr) > .05 and np.linalg.norm(dl) > .05


def test_nested_jit_and_second_derivative_retain_live_pairs():
    inner = jax.jit(lambda t: observable(t))
    cotangent = jnp.array([.2, -.3, .7, -.5])
    energy = lambda t: cotangent @ inner(t)
    derivative = jax.jit(jax.grad(energy))
    hessian = jax.jit(jax.grad(derivative))(.1)
    h = 2e-4
    np.testing.assert_allclose(hessian,
        (derivative(.1+h)-derivative(.1-h))/(2*h), atol=2e-8, rtol=2e-7)
    forward = jax.jit(lambda t: jax.jvp(derivative, (t,), (jnp.ones_like(t),))[1])(.1)
    np.testing.assert_allclose(forward, hessian, atol=2e-10)


@pytest.mark.parametrize('budget', [None, 4.])
def test_per_root_capacity_not_joint_root_count(budget):
    # n=3, two selected roots: right response is two independent 4x4 blocks.
    assert np.all(np.isfinite(jax.jvp(lambda t: observable(t, max_memory=budget), (.1,), (1.,))[1]))
    with pytest.raises(ValueError, match="max_dense"):
        jax.jvp(lambda t: observable(t, max_dense=3, max_memory=budget), (.1,), (1.,))


@pytest.mark.parametrize("matrix", [
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 3.]],
    [[1., 1., 0.], [0., 1., 0.], [0., 0., 3.]],
    [[0., -1., 0.], [1., 0., 0.], [0., 0., 3.]],
])
@pytest.mark.parametrize('budget', [None, 4.])
def test_invalid_pairs_reject_vector_derivatives(matrix, budget):
    a = jnp.asarray(matrix)
    def properties(t):
        result = solve_nonhermitian(a*t,
            config=NonHermitianSolverConfig(nroots=1), vector_response=True,
            response_config=LinearSolverConfig(method="direct", max_dense=4),
            max_memory=budget)
        return jnp.sum(result.right_vectors)+jnp.sum(result.left_vectors)
    assert not np.isfinite(jax.jvp(properties, (1.,), (1.,))[1])
    assert not np.isfinite(jax.grad(properties)(1.))


def test_opt_in_preserves_default_outputs_and_stopped_vector_contract():
    cfg = NonHermitianSolverConfig(nroots=2, atol=1e-11)
    old = solve_nonhermitian(family(.1), config=cfg)
    new = solved(.1)
    for name in ("values", "right_vectors", "left_vectors", "residual_norms",
                 "left_residual_norms", "condition_numbers"):
        np.testing.assert_allclose(getattr(new, name), getattr(old, name), atol=1e-13)
    default = lambda t: solve_nonhermitian(family(t), config=cfg).right_vectors
    np.testing.assert_array_equal(jax.jvp(default, (.1,), (1.,))[1], 0.)


def test_response_configuration_validation():
    with pytest.raises(TypeError, match="response_config"):
        solve_nonhermitian(family(.1), vector_response=True, response_config="direct")
    with pytest.raises(ValueError, match="vector_response"):
        solve_nonhermitian(family(.1), response_config=LinearSolverConfig())
