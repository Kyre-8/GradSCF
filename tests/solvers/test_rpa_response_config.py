"""RPA response uses explicit shared linear-solver controls and capacities."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.solvers import EigenSolverConfig, LinearSolverConfig, solve_rpa


@pytest.mark.parametrize("forward", ["dense", "davidson"])
@pytest.mark.parametrize("response", ["direct", "gmres"])
def test_rpa_response_config_preserves_roots_jvp_vjp_and_fd(forward, response):
    rng = np.random.default_rng(19)
    mixing = jnp.asarray(rng.normal(size=(6, 2))*.04)
    a = jnp.diag(jnp.linspace(.4, 2., 6))+mixing@mixing.T
    b = .3*mixing@mixing.T
    config = EigenSolverConfig(method=forward, nroots=2, max_subspace=6,
                                atol=1e-11, gradient_mode="implicit_eigenvector")
    linear = LinearSolverConfig(method=response, rtol=1e-11, restart=12, max_dense=12)
    probe = jnp.linspace(-.3, .4, 6)
    def loss(t, settings):
        out = solve_rpa(a+t*b, b+t*.01*a, config=config, response_config=settings)
        return jnp.sum(out.values*(1+((out.x+out.y).T@probe)**2))
    value, reverse = jax.jit(jax.value_and_grad(lambda t: loss(t, linear)))(0.)
    tangent = jax.jit(lambda t: jax.jvp(lambda x: loss(x, linear), (t,), (jnp.ones_like(t),))[1])(jnp.array(0.))
    np.testing.assert_allclose(value, loss(0., None), atol=1e-10, rtol=0)
    np.testing.assert_allclose(tangent, reverse, atol=1e-10, rtol=0)
    finite = (loss(1e-4, linear)-loss(-1e-4, linear))/2e-4
    np.testing.assert_allclose(tangent, finite, atol=2e-8, rtol=0)


@pytest.mark.parametrize("forward,dimension", [("dense", 6), ("davidson", 12)])
def test_rpa_response_direct_capacity_uses_its_actual_linear_dimension(forward, dimension):
    a = jnp.diag(jnp.linspace(.4, 2., 6))+jnp.full((6, 6), .01)
    b = jnp.eye(6)*.03
    config = EigenSolverConfig(method=forward, nroots=2, max_subspace=6,
                                gradient_mode="implicit_eigenvector")
    linear = LinearSolverConfig(method="direct", max_dense=dimension-1)
    def loss(t):
        out = solve_rpa(a+t*b, b, config=config, response_config=linear)
        return jnp.sum((out.x+out.y)**2)
    with pytest.raises(ValueError, match="max_dense"):
        jax.jvp(loss, (0.,), (1.,))


def test_rpa_rejects_an_invalid_response_configuration():
    from gradscf.solvers import solve_stable_rpa
    for solve in (solve_rpa, solve_stable_rpa):
        with pytest.raises(TypeError, match="response_config"):
            solve(jnp.eye(2), jnp.zeros((2, 2)), response_config="direct")
