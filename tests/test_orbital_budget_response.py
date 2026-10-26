"""Original scalar/metric/finite-iterate contracts with the same SCF budget."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf import orbital_optimization as orbital
from gradscf.scf.autodiff import attach_scf_backward
from test_orbital_memory_budget import _inputs


def test_budgeted_root_retains_joint_jvp_vjp_hvp_and_initial_independence():
    point, direction = jnp.array([8., 2.]), jnp.array([.3, -.2])
    def loss(p, initial, maximum):
        state = attach_scf_backward({'scale': p[0], 'nested': (p[1],)},
            solution=jax.lax.stop_gradient(jnp.sqrt(p[0]/p[1]))+0.*initial,
            residual=lambda x, args: args['nested'][0]*x*x-args['scale'],
            max_memory=maximum, reserved_bytes=4096, operator_min_bytes=512)
        return state**3
    def bundle(maximum):
        f = lambda p: loss(p, 1., maximum)
        return jax.jvp(jax.value_and_grad(f), (point,), (direction,))
    actual, expected = jax.jit(lambda: bundle(8.))(), jax.jit(lambda: bundle(None))()
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-10)
    _, pull = jax.vjp(lambda p: loss(p, 1., 8.), point)
    np.testing.assert_allclose(pull(jnp.asarray(1.))[0], actual[0][1], atol=2e-11)
    gradient = jax.jit(jax.grad(lambda p: loss(p, 1., None)))
    step = 2e-4
    finite = (gradient(point+step*direction)-gradient(point-step*direction))/(2*step)
    np.testing.assert_allclose(actual[1][1], finite, atol=3e-9, rtol=3e-8)
    assert jax.grad(lambda initial: loss(point, initial, 8.))(1.) == 0.


@pytest.mark.parametrize('mode', ['implicit', 'unrolled'])
def test_budgeted_existing_overlap_response_keeps_metric_anchor(mode, monkeypatch):
    import test_scf_orbital_gradients as old
    original = old.minimize_uks_from_integrals
    monkeypatch.setattr(old, 'minimize_uks_from_integrals',
        lambda **kwargs: original(**kwargs, max_memory=32., reserved_bytes=4096))
    old.test_overlap_response_includes_metric_constraint(mode)


def test_budgeted_existing_truncated_explicit_and_failed_implicit_modes(monkeypatch):
    import test_scf_orbital_gradients as old
    original = old.minimize_uks_from_integrals
    monkeypatch.setattr(old, 'minimize_uks_from_integrals',
        lambda **kwargs: original(**kwargs, max_memory=32., reserved_bytes=4096))
    old.test_unrolled_differentiates_truncated_solver_and_implicit_rejects_it()


def test_budgeted_empty_stationarity_space_does_not_attach_a_root(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Empty stationarity space requested a forward/root solve')
    monkeypatch.setattr(orbital, 'solve_orbitals', forbidden)
    monkeypatch.setattr(orbital, 'attach_scf_backward', forbidden)
    options = _inputs()
    options['mo_occ'] = np.zeros((2, 2))
    def value(t):
        result = orbital.minimize_uks_from_integrals(**{**options,
            'hcore': options['hcore']+t*jnp.eye(2)}, max_memory=8., reserved_bytes=4096)
        return result.total_energy, result.density_matrix
    (energy, density), tangent = jax.jvp(value, (.1,), (1.,))
    np.testing.assert_allclose((energy, jnp.sum(density), tangent[0], jnp.sum(tangent[1])),
                               0., atol=1e-13)
