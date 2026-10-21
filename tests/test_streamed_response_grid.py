"""Streamed AO grids preserve local and globally coupled TD response.

Seven fixed quadrature points and three Cartesian Gaussians isolate the grid
representation; these are operator regressions, not molecular spectra.
"""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.basis import CartesianAO, CartesianBasis
from gradscf.integrals.grids.ao import evaluate_cartesian_ao
from gradscf.integrals.grids.blocks import AOGrid
from gradscf.tddft import response
from gradscf.tddft.unrestricted import _unrestricted_grid_xc_response_action
from gradscf.scf.molecules import QuadratureGrid, RestrictedMolecule, UnrestrictedMolecule
from gradscf.tools.device import put_molecule_on_device
from gradscf.tddft._semilocal_response import SemilocalResponseFunctional
from gradscf.tddft._unrestricted_semilocal_response import (
    UnrestrictedSemilocalResponseFunctional,
    build_unrestricted_semilocal_response_action,
)
from gradscf.tools.features import (
    grid_features_with_spin_gradients_for_molecule,
    restricted_grid_response_variables,
)


class _Reference(SimpleNamespace):
    """Weak-referenceable view for the existing semilocal kernel cache."""


def _reference(parameter, *, streamed, unrestricted=False):
    center = jnp.asarray([-.2, .1, -.1]) + parameter * jnp.asarray([.1, -.2, .3])
    basis = CartesianBasis(aos=tuple(
        CartesianAO(center=center + offset, angular=(0, 0, 0),
                    exponents=jnp.asarray([exponent]), coefficients=jnp.asarray([1.]))
        for offset, exponent in ((jnp.zeros(3), .8),
                                 (jnp.asarray([.7, -.2, .3]), 1.1),
                                 (jnp.asarray([-.3, .6, .2]), .6))
    ), precompute_eri_groups=False)
    coords = jnp.asarray([[0., 0., 0.], [.2, -.1, .3], [-.4, .5, -.2],
                          [.8, -.7, .1], [.1, .3, -.4], [-.2, -.1, .6],
                          [.5, .2, .1]]) + parameter * jnp.asarray([.03, -.02, .01])
    weights = jnp.asarray([.2, .4, .1, .3, .15, .25, .35]) * jnp.exp(.2 * parameter)
    if streamed:
        provider = AOGrid.from_basis(basis, coords, weights, block_size=3)
        ao, dao, lap = jnp.zeros((0, 3)), jnp.zeros((4, 0, 3)), None
    else:
        values = evaluate_cartesian_ao(basis, coords, deriv=2)
        ao, dao, lap, provider = values[0], values[:4], values[4], None
    alpha = jnp.asarray([[.7, .08, -.02], [.08, .4, .03], [-.02, .03, .2]])
    beta = .63 * alpha if unrestricted else alpha
    return _Reference(
        ao=ao, ao_deriv1=dao, ao_laplacian=lap, grid_ao=provider,
        grid=SimpleNamespace(coords=coords, weights=weights),
        rdm1=jnp.stack((alpha, beta)), xc_density_floor=1e-12,
        dispersion_energy=None,
    )


def _orbitals():
    occupied = jnp.asarray([[.8], [.3], [-.2]])
    virtual = jnp.asarray([[.1, -.3], [.7, .2], [.2, .9]])
    return occupied, virtual


@pytest.mark.parametrize("unrestricted", [False, True])
def test_streamed_small_density_features_match_dense(unrestricted):
    reference = _reference(.04, streamed=False, unrestricted=unrestricted)
    streamed = _reference(.04, streamed=True, unrestricted=unrestricted)
    assert streamed.ao.shape == (0, 3)
    functions = (
        grid_features_with_spin_gradients_for_molecule,
        lambda mol: restricted_grid_response_variables(mol, feature_kind="MGGA_LAPL"),
    )
    for function in functions:
        for actual, expected in zip(jax.tree.leaves(function(streamed)),
                                    jax.tree.leaves(function(reference)), strict=True):
            np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def _restricted_action(parameter, values, *, streamed, block_size=3):
    molecule = _reference(parameter, streamed=streamed)
    occupied, virtual = _orbitals()
    functional = SemilocalResponseFunctional("pbe")
    hessian = functional.grid_response_tensor(molecule)
    action, _ = response._restricted_grid_xc_response(
        molecule, occupied, virtual, hessian * molecule.grid.weights[None, None],
        feature_kind="GGA", dtype=jnp.float64, block_size=block_size,
        compute_diagonal=False,
    )
    return action(values)


def _unrestricted_action(parameter, values, *, streamed, hvp=None, block_size=3):
    molecule = _reference(parameter, streamed=streamed, unrestricted=True)
    occupied, virtual = _orbitals()
    functional = UnrestrictedSemilocalResponseFunctional("pbe")
    action = build_unrestricted_semilocal_response_action(
        molecule, occupied, virtual, .9 * occupied, virtual[:, ::-1],
        functional.spin_grid_response_hvp if hvp is None else hvp,
        feature_kind="GGA", dtype=jnp.float64, block_size=block_size,
    )
    return action(values, -.4 * values)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_streamed_pbe_response_preserves_geometry_jvp_and_vjp(unrestricted):
    values = jnp.asarray([[[.2, -.3]], [[-.1, .4]]])
    operation = _unrestricted_action if unrestricted else _restricted_action
    actual = lambda p: operation(p, values, streamed=True)
    expected = lambda p: operation(p, values, streamed=False)
    point, direction = jnp.asarray(.04), jnp.asarray(.13)
    for left, right in zip(jax.tree.leaves(jax.jit(actual)(point)),
                           jax.tree.leaves(expected(point)), strict=True):
        np.testing.assert_allclose(left, right, atol=3e-11, rtol=3e-11)
    for left, right in zip(jax.tree.leaves(jax.jvp(actual, (point,), (direction,))[1]),
                           jax.tree.leaves(jax.jvp(expected, (point,), (direction,))[1]),
                           strict=True):
        np.testing.assert_allclose(left, right, atol=3e-10, rtol=3e-10)
    loss = lambda function, p: sum(jnp.sum(value**2) for value in jax.tree.leaves(function(p)))
    np.testing.assert_allclose(jax.grad(lambda p: loss(actual, p))(point),
                               jax.grad(lambda p: loss(expected, p))(point),
                               atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_streamed_custom_hvp_retains_complete_cross_grid_coupling(unrestricted):
    values = jnp.asarray([[[.2, -.3]], [[-.1, .4]]])

    def global_hvp(molecule, tangent):
        assert tangent.shape[-1] == 7, "An opaque HVP must receive the complete grid"
        rho, _, _, _ = restricted_grid_response_variables(molecule, feature_kind="LDA")
        return (.11 + rho[None, None]) * tangent + .17 * jnp.sum(tangent, axis=-1, keepdims=True)

    def spin_global_hvp(molecule, alpha, beta):
        assert alpha.shape[-1] == beta.shape[-1] == 7
        return global_hvp(molecule, alpha + .3 * beta), global_hvp(molecule, beta + .3 * alpha)

    def operation(parameter, *, streamed):
        if unrestricted:
            return _unrestricted_action(parameter, values, streamed=streamed, hvp=spin_global_hvp)
        molecule = _reference(parameter, streamed=streamed)
        occupied, virtual = _orbitals()
        action = response._restricted_grid_xc_response_hvp(
            molecule, occupied, virtual, global_hvp, feature_kind="GGA", dtype=jnp.float64,
        )
        return action(values)

    actual = lambda p: operation(p, streamed=True)
    expected = lambda p: operation(p, streamed=False)
    point = jnp.asarray(.04)
    for left, right in zip(jax.tree.leaves(jax.jit(actual)(point)),
                           jax.tree.leaves(expected(point)), strict=True):
        np.testing.assert_allclose(left, right, atol=2e-11, rtol=2e-11)
    loss = lambda function, p: sum(jnp.sum(value**2) for value in jax.tree.leaves(function(p)))
    np.testing.assert_allclose(jax.grad(lambda p: loss(actual, p))(point),
                               jax.grad(lambda p: loss(expected, p))(point),
                               atol=3e-10, rtol=3e-10)


def test_streamed_spin_density_kernel_fallback_matches_dense():
    class DensityKernel:
        response_feature_kind = "LDA"

        def spin_local_kernel(self, alpha, beta):
            return .2 + .1 * alpha, .03 + .07 * (alpha + beta), .15 + .12 * beta

    occupied, virtual = _orbitals()
    values = jnp.asarray([[[.2, -.3]], [[-.1, .4]]])
    outputs = []
    for streamed in (False, True):
        molecule = _reference(.04, streamed=streamed, unrestricted=True)
        action = _unrestricted_grid_xc_response_action(
            molecule, DensityKernel(), occupied, virtual, .9 * occupied, virtual[:, ::-1],
            dtype=jnp.float64, block_size=3,
        )
        outputs.append(action(values, -.4 * values))
    for actual, expected in zip(outputs[1], outputs[0], strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def _molecular_reference(view, *, unrestricted):
    molecule_type = UnrestrictedMolecule if unrestricted else RestrictedMolecule
    return molecule_type(
        ao=view.ao, ao_deriv1=view.ao_deriv1, grid_ao=view.grid_ao,
        grid=QuadratureGrid(view.grid.weights, view.grid.coords),
        dipole_integrals=jnp.zeros((3, 3, 3)), rep_tensor=jnp.zeros((0, 0, 0, 0)),
        mo_coeff=jnp.stack((jnp.eye(3), jnp.eye(3))),
        mo_occ=jnp.asarray([[1., 0., 0.], [1., 0., 0.]]),
        mo_energy=jnp.asarray([[-1., .4, .8], [-.9, .5, 1.]]),
        rdm1=view.rdm1, h1e=jnp.eye(3), nuclear_repulsion=0.,
    )


@pytest.mark.parametrize("unrestricted", [False, True])
def test_streamed_reference_density_preserves_existing_spin_shape(unrestricted):
    outputs = []
    for streamed in (False, True):
        view = _reference(.04, streamed=streamed, unrestricted=unrestricted)
        outputs.append(_molecular_reference(view, unrestricted=unrestricted).density())
    assert outputs[1].shape == ((7,) if unrestricted else (7, 2))
    np.testing.assert_allclose(outputs[1], outputs[0], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_device_transfer_includes_streamed_grid_provider(unrestricted):
    view = _reference(.04, streamed=True, unrestricted=unrestricted)
    provider = view.grid_ao
    # Host-backed provider leaves distinguish a real transfer even on CPU.
    view.grid_ao = replace(provider, **{name: np.asarray(getattr(provider, name))
        for name in ("coords", "weights", "centers", "exponents", "coefficients")})
    molecule = _molecular_reference(view, unrestricted=unrestricted)
    device = jax.devices("cpu")[0]
    transferred = put_molecule_on_device(molecule, device)
    assert transferred.grid_ao.block_size == provider.block_size
    for leaf in jax.tree.leaves(transferred.grid_ao):
        assert isinstance(leaf, jax.Array)
        assert leaf.devices() == {device}


@pytest.mark.parametrize("unrestricted", [False, True])
def test_response_grid_block_respects_ao_provider_workspace(monkeypatch, unrestricted):
    block = AOGrid.block

    def bounded_block(provider, start, size, *, deriv=0):
        assert size <= provider.block_size, "Response must respect the AO workspace block budget"
        return block(provider, start, size, deriv=deriv)

    monkeypatch.setattr(AOGrid, "block", bounded_block)
    values = jnp.asarray([[[.2, -.3]], [[-.1, .4]]])
    operation = _unrestricted_action if unrestricted else _restricted_action
    actual = operation(jnp.asarray(.04), values, streamed=True, block_size=99)
    expected = operation(jnp.asarray(.04), values, streamed=False, block_size=3)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(left, right, atol=3e-11, rtol=3e-11)


def test_streamed_legacy_neural_scf_projection_and_parameter_vjp_match_dense():
    # Legacy neural_xc derives complete-grid features before AO projection.
    # Its optional local-HFX/fxx/PT2 caches have independent data requirements;
    # this regression covers the ordinary semilocal network with no HFX/PT2.
    from gradscf.model import neural_xc

    functional = neural_xc.make_functional(
        semilocal_xc="pbe", hidden_dims=(4,), include_hfx_channel=False,
        include_pt2_channel=False, input_feature_mode="enhanced",
    )
    references = [_molecular_reference(_reference(.04, streamed=streamed), unrestricted=False)
                  for streamed in (False, True)]
    params = functional.init_from_molecule(jax.random.PRNGKey(37), references[0])

    def projection(parameters, molecule):
        return functional.scf_xc_fock_terms(
            parameters, molecule, weights=molecule.grid.weights,
            functional_dtype=jnp.float64, vxc_clip=20.,
        )

    actual, expected = projection(params, references[1]), projection(params, references[0])
    for left, right in zip(actual, expected, strict=True):
        assert np.all(np.isfinite(np.asarray(left)))
        np.testing.assert_allclose(left, right, atol=3e-10, rtol=3e-10)

    def loss(parameters, molecule):
        matrix, _, _, energy = projection(parameters, molecule)
        return jnp.sum(matrix**2) + .01 * energy

    actual_grad = jax.grad(lambda p: loss(p, references[1]))(params)
    expected_grad = jax.grad(lambda p: loss(p, references[0]))(params)
    for left, right in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad), strict=True):
        assert np.all(np.isfinite(np.asarray(left)))
        np.testing.assert_allclose(left, right, atol=2e-9, rtol=2e-9)
