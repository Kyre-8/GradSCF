"""Force supervision: analytic signs, mixed derivatives, and optimizer use."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from gradscf.model.neural_xc import training


def _harmonic_energy(params, coordinates):
    return 0.5 * params["stiffness"] * jnp.sum(coordinates**2)


def _mlp_energy(params, coordinates):
    hidden = jnp.tanh(coordinates @ params["hidden"]["weight"] + params["hidden"]["bias"])
    return jnp.sum(hidden @ params["output"])


def _mlp_case():
    params = {
        "hidden": {
            "weight": jnp.array([[0.2, -0.1], [0.3, 0.4], [-0.2, 0.5]]),
            "bias": jnp.array([0.1, -0.2]),
        },
        "output": jnp.array([0.7, -0.4]),
    }
    coordinates = jnp.array([[0.2, -0.3, 0.1], [-0.4, 0.2, 0.6]])
    target = jnp.array([[0.1, 0.2, -0.1], [-0.2, 0.05, 0.3]])
    return params, coordinates, target


def test_energy_and_forces_sign_shape_and_jit_pytree():
    params = {"stiffness": jnp.array(2.0)}
    coordinates = jnp.array([[1.0, 2.0, -1.0], [0.0, -2.0, 3.0]])
    evaluate = jax.jit(lambda p, r: training.energy_and_forces(_harmonic_energy, p, r))
    result = evaluate(params, coordinates)
    assert isinstance(result, training.EnergyAndForces)
    assert result.energy.shape == ()
    assert result.forces.shape == coordinates.shape
    assert len(jax.tree_util.tree_leaves(result)) == 2
    np.testing.assert_allclose(result.energy, 19.0, atol=1e-14)
    np.testing.assert_allclose(result.forces, -2.0 * coordinates, atol=1e-14)


def test_force_loss_normalization_and_nested_parameter_gradient():
    coordinates = jnp.array([[1.0, -2.0, 0.0], [0.0, 1.0, -3.0]])
    target = jnp.zeros_like(coordinates)
    params = {"stiffness": jnp.array(2.0)}
    loss_grad = jax.jit(training.make_force_loss_and_grad(_harmonic_energy))
    loss, grad = loss_grad(params, coordinates, target)
    # Sum R^2 = 15, six force components: L = k^2 * 15 / (2 * 6).
    np.testing.assert_allclose(loss, 5.0, atol=1e-14)
    np.testing.assert_allclose(grad["stiffness"], 5.0, atol=1e-14)
    curvature = jax.grad(lambda k: loss_grad({"stiffness": k}, coordinates, target)[1]["stiffness"])(params["stiffness"])
    np.testing.assert_allclose(curvature, 2.5, atol=1e-14)


def test_mlp_force_loss_parameter_gradient_matches_finite_difference():
    params, coordinates, target = _mlp_case()
    loss_fn = lambda p: training.force_matching_loss(_mlp_energy, p, coordinates, target)
    actual = jax.grad(loss_fn)(params)
    leaves, structure = jax.tree_util.tree_flatten(params)
    actual_leaves = jax.tree_util.tree_leaves(actual)
    step = 1e-5
    for leaf_index, leaf in enumerate(leaves):
        expected = np.zeros(leaf.shape)
        for index in np.ndindex(leaf.shape):
            plus, minus = list(leaves), list(leaves)
            plus[leaf_index] = leaf.at[index].add(step)
            minus[leaf_index] = leaf.at[index].add(-step)
            expected[index] = (
                loss_fn(jax.tree_util.tree_unflatten(structure, plus))
                - loss_fn(jax.tree_util.tree_unflatten(structure, minus))
            ) / (2.0 * step)
        np.testing.assert_allclose(actual_leaves[leaf_index], expected, rtol=2e-7, atol=2e-10)


def test_one_optax_update_reduces_mlp_force_loss():
    params, coordinates, target = _mlp_case()
    optimizer = optax.sgd(learning_rate=0.05)
    state = optimizer.init(params)
    loss_grad = jax.jit(training.make_force_loss_and_grad(_mlp_energy))
    before, grad = loss_grad(params, coordinates, target)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grad))
    assert float(optax.global_norm(grad)) > 1e-8
    updates, _ = optimizer.update(grad, state, params)
    after, _ = loss_grad(optax.apply_updates(params, updates), coordinates, target)
    assert float(after) < float(before)


@pytest.mark.parametrize("coordinates", [jnp.zeros((3,)), jnp.zeros((2, 2)), jnp.zeros((0, 3))])
def test_rejects_invalid_coordinate_shape(coordinates):
    with pytest.raises(ValueError, match="coordinates"):
        training.energy_and_forces(_harmonic_energy, {"stiffness": 1.0}, coordinates)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.complex128])
def test_rejects_non_real_float_coordinates(dtype):
    with pytest.raises(TypeError, match="coordinates"):
        training.energy_and_forces(_harmonic_energy, {"stiffness": 1.0}, jnp.ones((2, 3), dtype=dtype))


def test_rejects_broadcastable_target_force_shape():
    with pytest.raises(ValueError, match="target_forces"):
        training.force_matching_loss(_harmonic_energy, {"stiffness": 1.0}, jnp.ones((2, 3)), jnp.ones((3,)))


def test_rejects_integer_force_targets():
    with pytest.raises(TypeError, match="target_forces"):
        training.force_matching_loss(_harmonic_energy, {"stiffness": 1.0}, jnp.ones((2, 3)), jnp.ones((2, 3), dtype=jnp.int32))


def test_energy_callback_must_return_scalar():
    with pytest.raises(TypeError, match="scalar"):
        training.energy_and_forces(lambda p, r: jnp.sum(r**2, axis=1), {}, jnp.ones((2, 3)))
