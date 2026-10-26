"""External global XC keeps cross-grid coupling with streamed density inputs."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft
from gradscf.integrals.basis import CartesianAO, CartesianBasis
from gradscf.integrals.grids.ao import evaluate_cartesian_ao
from gradscf.integrals.grids.blocks import AOGrid


def _molecule(displacement, *, streamed):
    basis = CartesianBasis(aos=(
        CartesianAO(jnp.asarray([-.3, .1, .2]) + displacement * jnp.asarray([.2, -.1, .3]),
                    (0, 0, 0), jnp.asarray([.7]), jnp.asarray([1.])),
        CartesianAO(jnp.asarray([.4, -.2, .1]), (1, 0, 0),
                    jnp.asarray([.9]), jnp.asarray([.8])),
    ), precompute_eri_groups=False)
    coords = jnp.asarray([[0., 0., 0.], [.2, -.1, .3], [-.4, .5, -.2],
                          [.8, -.7, .1], [.1, .3, -.4], [-.2, -.1, .6], [.5, .2, .1]])
    weights = jnp.asarray([.2, .4, .1, .3, .15, .25, .35])
    if streamed:
        provider = AOGrid.from_basis(basis, coords, weights, block_size=3)
        ao, derivative, laplacian = jnp.zeros((0, 2)), jnp.zeros((4, 0, 2)), None
    else:
        provider = None
        values = evaluate_cartesian_ao(basis, coords, deriv=2)
        ao, derivative, laplacian = values[0], values[:4], values[4]
    return SimpleNamespace(ao=ao, ao_deriv1=derivative, ao_laplacian=laplacian,
                           grid_ao=provider, overlap_matrix=jnp.eye(2),
                           grid=SimpleNamespace(weights=weights, coords=coords))


def _functional():
    def inputs(state):
        return dict(
            local=(state.rho, state.grad_rho, state.tau, state.laplacian_rho),
            spin=(state.rho_spin, state.grad_rho_spin, state.tau_spin,
                  state.laplacian_rho_spin),
            weights=state.weights,
        )

    def energy(params, features):
        rho, gradient, tau, laplacian = features["local"]
        rho_spin, gradient_spin, tau_spin, laplacian_spin = features["spin"]
        local = jnp.stack((rho, jnp.sum(gradient**2, axis=-1), tau, laplacian), axis=-1)
        # Four externally constructed layers; the last scalar is pooled only
        # after ALL original grid points have been assembled as small features.
        value = local
        for weight, bias in params["layers"][:-1]:
            value = jnp.tanh(value @ weight + bias)
        weight, bias = params["layers"][-1]
        pooled = jnp.dot(features["weights"], rho * (value @ weight + bias)[:, 0])
        spin_term = jnp.dot(features["weights"], .03 * rho_spin[0] * rho_spin[1]
            + .002 * jnp.sum(gradient_spin[0] * gradient_spin[1], axis=-1)
            + .004 * tau_spin[0] * tau_spin[1]
            + .001 * laplacian_spin[0] * laplacian_spin[1])
        # pooled**2 couples different blocks: summing independent block
        # callbacks would silently change both the energy and its Hessian.
        return params["scale"] * (pooled + .17 * pooled**2 + spin_term)

    layers = tuple((jnp.linspace(-.2, .3, ni * no).reshape(ni, no),
                    jnp.linspace(.01, .03, no))
                   for ni, no in ((4, 5), (5, 4), (4, 3), (3, 1)))
    return dft.custom_xc(input_fn=inputs, energy_fn=energy, name="streamed_global_mlp"), {
        "scale": jnp.asarray(.3), "layers": layers}


@pytest.mark.parametrize("spin", [False, True], ids=["restricted", "unrestricted"])
def test_global_external_network_streamed_energy_potential_kernel_and_mixed_ad(spin):
    functional, params = _functional()
    density = jnp.asarray([[1., .12], [.12, .7]])
    tangent = jnp.asarray([[.2, -.07], [-.07, -.1]])
    if spin:
        density = jnp.stack((.6 * density, .4 * density))
        tangent = jnp.stack((tangent, -.3 * tangent))
    results = []
    for streamed in (False, True):
        molecule = _molecule(jnp.asarray(.04), streamed=streamed)
        features = functional.inputs_for_density(molecule, density)
        assert features["local"][0].shape == (7,)
        assert features["spin"][0].shape == (2, 7)

        def observables(parameters):
            return (functional.energy_for_density(parameters, molecule, density),
                    functional.potential(parameters, molecule, density),
                    functional.kernel_action(parameters, molecule, density, tangent))

        actual = jax.jit(observables)(params)
        # Gradients reach every network layer through the full potential and
        # response-kernel graph, not just the scalar XC energy.
        mixed = jax.jit(jax.grad(lambda p: jnp.sum(observables(p)[1]**2)
                                 + jnp.sum(observables(p)[2]**2)))(params)

        def geometry_response(displacement):
            state = _molecule(displacement, streamed=streamed)
            kernel = functional.kernel_action(params, state, density, tangent)
            return jnp.sum(kernel**2)

        geometry = jax.jit(jax.value_and_grad(geometry_response))(jnp.asarray(.04))
        results.append((actual, mixed, geometry))
    for value, expected in zip(jax.tree.leaves(results[1]), jax.tree.leaves(results[0])):
        assert np.isfinite(np.asarray(value)).all()
        np.testing.assert_allclose(value, expected, atol=3e-10, rtol=3e-10)
    for weight_gradient, bias_gradient in results[1][1]["layers"]:
        assert float(jnp.linalg.norm(weight_gradient)) > 1e-9
        assert float(jnp.linalg.norm(bias_gradient)) > 1e-9
    assert abs(float(results[1][2][1])) > 1e-9


def test_external_explicit_ao_requests_materialize_values_instead_of_empty_placeholders():
    functional = dft.custom_xc(
        input_fn=lambda state: (state.ao, state.ao_deriv1),
        energy_fn=lambda params, arrays: params * jnp.sum(arrays[0]**2))
    density = jnp.asarray([[1., .12], [.12, .7]])
    expected = functional.inputs_for_density(_molecule(jnp.asarray(.04), streamed=False), density)
    actual = functional.inputs_for_density(_molecule(jnp.asarray(.04), streamed=True), density)
    assert actual[0].shape == (7, 2)
    assert actual[1].shape == (4, 7, 2)
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("kind", ["LDA", "GGA", "MGGA", "MGGA_LAPL"])
def test_streamed_grid_component_potential_and_geometry_ad_match_dense(kind):
    from gradscf.scf.differentiable import _build_vxc_matrix_from_components

    def potential(displacement, streamed):
        molecule = _molecule(displacement, streamed=streamed)
        return _build_vxc_matrix_from_components(
            molecule=molecule,
            # Supplied integration weights may differ from the provider's
            # defaults.  The same explicit weights must be used in both paths.
            weights=1.1 * molecule.grid.weights,
            v_rho=jnp.linspace(.03, .09, 7),
            v_grad=jnp.linspace(-.04, .06, 21).reshape(7, 3),
            v_tau=jnp.linspace(.01, .04, 7),
            v_lapl=jnp.linspace(-.02, .01, 7), xc_kind=kind)

    results = []
    for streamed in (False, True):
        function = lambda displacement: jnp.sum(potential(displacement, streamed)**2)
        results.append(jax.jit(lambda displacement: (
            potential(displacement, streamed),
            jax.grad(function)(displacement),
            jax.grad(jax.grad(function))(displacement),
        ))(jnp.asarray(.04)))
    for value, expected in zip(results[1], results[0]):
        assert np.isfinite(np.asarray(value)).all()
        np.testing.assert_allclose(value, expected, atol=3e-11, rtol=3e-11)


def test_replacing_grid_provider_invalidates_trial_neural_grid_payload():
    from gradscf.scf.differentiable import _replace_molecule

    molecule = _molecule(jnp.asarray(.04), streamed=True)
    molecule.neural_xc_grid_payload = {"old_density_features": jnp.ones(7)}
    provider = _molecule(jnp.asarray(.05), streamed=True).grid_ao
    updated = _replace_molecule(molecule, grid_ao=provider)
    assert updated.grid_ao is provider
    assert updated.neural_xc_grid_payload is None
    assert molecule.neural_xc_grid_payload is not None
