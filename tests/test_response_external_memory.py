"""Opaque external networks retain global density AD and shared SCF modes."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.scf.molecules import QuadratureGrid, RestrictedMolecule
from gradscf.tddft import response


def _parameters():
    layers = []
    for inputs, outputs in ((4, 5), (5, 4), (4, 3), (3, 1)):
        weight = jnp.linspace(-.18, .23, inputs * outputs).reshape(inputs, outputs)
        bias = jnp.linspace(.02, .05, outputs)
        layers.append((weight, bias))
    return {"layers": tuple(layers), "scale": jnp.asarray(.3)}


def _four_layer_mlp(layers, inputs):
    value = inputs
    for weight, bias in layers[:-1]:
        value = jnp.tanh(value @ weight + bias)
    weight, bias = layers[-1]
    return (value @ weight + bias)[..., 0]


def _network_energy(params, density, ao, weights, coupled):
    density = .5 * (density + density.T)
    # Local fields come from current AO density without a grid-by-AO-pair tensor.
    rho = jnp.sum((ao @ density) * ao, axis=-1)
    inputs = jnp.stack((rho, rho**2, jnp.sin(rho), jnp.cos(rho)), axis=-1)
    pooled = jnp.dot(weights, _four_layer_mlp(params["layers"], inputs))
    energy = pooled
    if coupled:
        # The global model also takes the complete AO matrix directly, and
        # couples every grid point through the square of the global pooling.
        energy = energy + .23 * pooled**2 + .17 * _four_layer_mlp(
            params["layers"], density.reshape(-1))
    return params["scale"] * energy


def _functional(coupled, expected_grids=None):
    def inputs(state):
        if expected_grids is not None:
            assert state.ao.shape[0] == expected_grids, "Opaque energy callbacks require the full grid"
        return state.total_density_matrix, state.ao, state.weights

    return dft.custom_xc(
        input_fn=inputs,
        energy_fn=lambda params, data: _network_energy(params, *data, coupled),
        name="four_layer_global" if coupled else "four_layer_local",
    )


@pytest.fixture(scope="module")
def external_reference():
    # An algebraic two-AO problem isolates the representation change. Only the
    # ERIs come from this native H2 basis; no molecular result is claimed here.
    topology, parameters = integrals.prepare_basis(
        "H 0 0 0; H 0 0 1.4", "sto-3g", cart=False)
    plan = integrals.make_plan(topology)
    eri = plan.evaluate("eri", parameters)
    ao = jnp.asarray([[.8, .2], [.3, .7], [-.2, .6], [.4, -.3],
                      [.5, .1], [-.1, .4], [.2, .9]])
    reference = RestrictedMolecule(
        ao=ao, ao_deriv1=jnp.stack((ao, .2 * ao, -.1 * ao, .3 * ao)),
        ao_laplacian=.14 * ao,
        grid=QuadratureGrid(jnp.linspace(.1, .3, len(ao)), jnp.zeros((len(ao), 3))),
        dipole_integrals=jnp.zeros((3, 2, 2)), rep_tensor=jnp.zeros((0, 0, 0, 0)),
        h1e=jnp.asarray([[-3., .08], [.08, 3.]]), overlap_matrix=jnp.eye(2),
        mo_coeff=jnp.stack((jnp.eye(2),) * 2), mo_occ=jnp.asarray([[1., 0.]] * 2),
        mo_energy=jnp.asarray([[-3., 3.]] * 2),
        rdm1=jnp.asarray([[[1., 0.], [0., 0.]]] * 2),
        nuclear_repulsion=0., nocc=1, direct_basis=NativeDirectBasis(plan, parameters))
    return reference, eri


def _dense_td_action(reference, eri, energy, params, density, values, full_td):
    no = reference.nocc
    coeff = reference.mo_coeff[0]
    occupied, virtual = coeff[:, :no], coeff[:, no:]
    gaps = reference.mo_energy[0, no:][None] - reference.mo_energy[0, :no, None]
    hessian = jax.hessian(lambda dm: energy(params, dm.reshape(density.shape)))(density.reshape(-1))

    def coupling(x, bottom):
        left, right = (occupied, virtual) if bottom else (virtual, occupied)
        transition = 2 * jnp.einsum("nia,pi,qa->npq" if bottom else "nia,pa,qi->npq",
                                    x, left, right)
        potential = (jnp.einsum("pqrs,nrs->npq", eri, transition)
                     + (transition.reshape(len(x), -1) @ hessian.T).reshape(transition.shape))
        return jnp.einsum("npq,qi,pa->nia", potential, occupied, virtual)

    x = values[:, :1].reshape(-1, 1, 1)
    upper = gaps[None] * x + coupling(x, False)
    if not full_td:
        return upper.reshape(values.shape)
    y = values[:, 1:].reshape(-1, 1, 1)
    upper = upper + coupling(y, True)
    lower = coupling(x, True) + gaps[None] * y + coupling(y, False)
    return jnp.concatenate((upper.reshape(-1, 1), -lower.reshape(-1, 1)), axis=1)


def _check_parameter_and_density_response(actual, oracle, params, density, values):
    np.testing.assert_allclose(jax.jit(actual)(params, density, values),
                               oracle(params, density, values), atol=2e-11, rtol=2e-11)
    direction = jax.tree.map(lambda value: jnp.ones_like(value) * .07, params)
    density_direction = jnp.asarray([[.2, -.1], [.3, .05]])
    tangent = lambda operation: jax.jvp(
        lambda p, dm: operation(p, dm, values),
        (params, density), (direction, density_direction))[1]
    np.testing.assert_allclose(jax.jit(tangent, static_argnums=0)(actual), tangent(oracle),
                               atol=3e-10, rtol=3e-10)
    cotangent = jnp.arange(values.size, dtype=values.dtype).reshape(values.shape) / 7 + .2
    gradients = []
    for operation in (actual, oracle):
        loss = lambda p, dm: jnp.sum(operation(p, dm, values) * cotangent)
        gradients.append(jax.jit(jax.grad(loss, argnums=(0, 1)))(params, density))
    for value, expected in zip(jax.tree.leaves(gradients[0]),
                                jax.tree.leaves(gradients[1]), strict=True):
        np.testing.assert_allclose(value, expected, atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("full_td", [False, True])
def test_external_four_layer_ao_network_uses_full_density_hvp(
        external_reference, monkeypatch, coupled, full_td):
    reference, eri = external_reference
    params = _parameters()
    model = _functional(coupled, expected_grids=7)
    density = jnp.asarray([[1.3, .12], [.12, .7]])
    values = jnp.asarray([[.2, -.3], [-.4, .1], [.7, .2]])
    if not full_td:
        values = values[:, :1]

    def forbidden(*args, **kwargs):
        raise AssertionError("An opaque AO network must not enter local grid projections")

    monkeypatch.setattr(response, "_restricted_grid_xc_response", forbidden)
    monkeypatch.setattr(response, "_restricted_grid_xc_response_hvp", forbidden)

    def actual(p, dm, x):
        state = replace(reference, rdm1=jnp.stack((.5 * dm, .5 * dm)))
        options = dict(integral_cache="none", direct_workers=1, grid_block_size=1)
        if full_td:
            action = response.build_restricted_tdhf_operator(
                state, model, xc_params=p, response_kernel_options=options)
        else:
            action, _, _ = response.build_restricted_tda_operator(
                state, model, xc_params=p, response_kernel_options=options)
        return action(x)

    energy = lambda p, dm: _network_energy(p, dm, reference.ao, reference.grid.weights, coupled)
    oracle = lambda p, dm, x: _dense_td_action(reference, eri, energy, p, dm, x, full_td)
    _check_parameter_and_density_response(actual, oracle, params, density, values)


def test_external_global_grid_network_hvp_is_called_on_the_complete_grid(
        external_reference, monkeypatch):
    reference, eri = external_reference
    reference = replace(reference, grid=replace(reference.grid, weights=jnp.ones(7)))
    params = _parameters()
    density = jnp.asarray([[1.3, .12], [.12, .7]])
    values = jnp.asarray([[.2], [-.4], [.7]])

    def grid_energy(p, rho):
        features = jnp.stack((rho, rho**2, jnp.sin(rho), jnp.cos(rho)), axis=-1)
        pooled = jnp.sum(_four_layer_mlp(p["layers"], features))
        return p["scale"] * (pooled + .23 * pooled**2)

    class GlobalHVP:
        exact_exchange_fraction = 0.
        response_feature_kind = "LDA"

        def __init__(self, p):
            self.params = p

        def grid_response_hvp(self, state, tangent):
            assert tangent.shape[-1] == 7, "A global HVP must receive every grid point"
            rho = jnp.einsum("gp,pq,gq->g", state.ao, state.rdm1.sum(axis=0), state.ao)
            return jax.vmap(lambda t: jax.jvp(
                jax.grad(lambda r: grid_energy(self.params, r)), (rho,), (t,))[1])(
                    tangent[:, 0])[:, None]

    def forbidden(*args, **kwargs):
        raise AssertionError("A global HVP must not use a local Hessian tensor")

    monkeypatch.setattr(response, "_restricted_grid_xc_response", forbidden)

    def actual(p, dm, x):
        state = replace(reference, rdm1=jnp.stack((.5 * dm, .5 * dm)))
        action, _, _ = response.build_restricted_tda_operator(
            state, GlobalHVP(p), response_kernel_options=dict(
                integral_cache="none", direct_workers=1, grid_block_size=1))
        return action(x)

    energy = lambda p, dm: grid_energy(p, jnp.einsum(
        "gp,pq,gq->g", reference.ao, .5 * (dm + dm.T), reference.ao))
    oracle = lambda p, dm, x: _dense_td_action(reference, eri, energy, p, dm, x, False)
    _check_parameter_and_density_response(actual, oracle, params, density, values)


@pytest.mark.parametrize("mode,backward", [
    ("fixed_density", "implicit"), ("self_consistent", "explicit"),
    ("self_consistent", "implicit"),
])
def test_external_network_three_training_modes_reuse_direct_jk(external_reference, mode, backward):
    from gradscf.model.neural_xc.training import (
        MolecularTrainingConfig, MolecularTrainingDatum, molecular_loss,
    )

    direct, eri = external_reference
    dense = replace(direct, direct_basis=None, rep_tensor=eri)
    model, params = _functional(True, expected_grids=7), _parameters()
    config = MolecularTrainingConfig(
        mode=mode, scf_gradient_mode=backward, e0_total_mse_weight=1.,
        scf_max_cycle=60, scf_damping=0., scf_conv_tol_density=1e-10,
        scf_conv_tol_energy=1e-12, scf_eigenvalue_jitter=0.,
        scf_implicit_diff_tolerance=1e-10, scf_implicit_diff_max_iter=40,
        scf_require_converged=True)

    def loss(state, scale):
        p = dict(params, scale=scale)
        datum = MolecularTrainingDatum(state, target_e0_total_h=jnp.asarray(-5.))
        return molecular_loss(p, model, datum, training_config=config)[0]

    actual = jax.jit(jax.value_and_grad(lambda scale: loss(direct, scale)))(params["scale"])
    expected = jax.jit(jax.value_and_grad(lambda scale: loss(dense, scale)))(params["scale"])
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-6)
    assert np.isfinite(np.asarray(actual)).all() and abs(float(actual[1])) > 1e-8
    step = 1e-4
    finite_difference = (loss(dense, params["scale"] + step)
                         - loss(dense, params["scale"] - step)) / (2 * step)
    np.testing.assert_allclose(actual[1], finite_difference, atol=2e-8, rtol=2e-5)


def test_external_network_scf_facade_does_not_force_direct_reference_to_pair(monkeypatch):
    from gradscf import gto
    from gradscf.scf.inputs.types import RKSIntegralInputs

    molecule = gto.M(atom="H 0 0 0; H 0 0 .8", basis="sto-3g", cart=True)
    model, params = _functional(True), _parameters()
    options = dict(xc=model, params=params, integral_backend="native", grids_level=0,
                   conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-8, max_cycle=60)
    expected = dft.RKS(molecule, jk_backend="full", **options).run()
    assert expected.converged

    def forbidden(*args, **kwargs):
        raise AssertionError("External direct SCF must not force an AO-pair ERI cache")

    monkeypatch.setattr(RKSIntegralInputs, "response_eri_pair_matrix", forbidden)
    calls = []
    original_jk = NativeDirectBasis.get_jk

    def selected_jk(provider, density, **kwargs):
        calls.append((kwargs.get("with_j", True), kwargs.get("with_k", True)))
        return original_jk(provider, density, **kwargs)

    monkeypatch.setattr(NativeDirectBasis, "get_jk", selected_jk)
    actual = dft.RKS(molecule, jk_backend="direct", **options).run()
    assert actual.converged
    state = actual.to_reference()
    assert state.direct_basis is not None
    assert state.eri_pair_matrix is None and state.rep_tensor.size == 0
    np.testing.assert_allclose(actual.e_tot, expected.e_tot, atol=2e-9, rtol=0)
    np.testing.assert_allclose(actual.make_rdm1(), expected.make_rdm1(), atol=2e-8, rtol=0)
    assert calls and calls[-1] == (True, False), "The external finalizer only needs Coulomb"


@pytest.mark.parametrize("provider_kind", ["native", "generic"])
@pytest.mark.parametrize("consumer", ["scf_j", "scf_functional_default", "training_j", "training_k"])
def test_external_consumers_select_required_jk_operators(
        external_reference, monkeypatch, provider_kind, consumer):
    from gradscf.scf.differentiable import DifferentiableSCF
    from gradscf.model.neural_xc.training import targets

    reference, _ = external_reference
    density = jnp.asarray([[1.1, .2], [.2, .7]])
    calls = []

    def native_jk(provider, dm, *, with_j=True, with_k=True, **kwargs):
        calls.append((with_j, with_k))
        return (2 * dm if with_j else jnp.zeros_like(dm),
                3 * dm if with_k else jnp.zeros_like(dm))

    class GenericProvider:
        def get_jk(self, dm):
            calls.append(None)
            return 2 * dm, 3 * dm

    monkeypatch.setattr(NativeDirectBasis, "get_jk", native_jk)
    native = reference.direct_basis
    provider = native if provider_kind == "native" else GenericProvider()
    state = replace(reference, direct_basis=provider,
                    rdm1=jnp.stack((.5 * density, .5 * density)))
    if consumer in {"scf_j", "scf_functional_default"}:
        context = SimpleNamespace(df_factors=None, direct_basis=provider, rep_tensor=None)
        if consumer == "scf_functional_default":
            context.xc_functional = dft.custom_xc(
                input_fn=lambda state: state.total_density_matrix,
                energy_fn=lambda params, dm: jnp.sum(dm * 0.))
            actual, exchange = DifferentiableSCF()._restricted_response_jk(context, density)
        else:
            actual, exchange = DifferentiableSCF()._restricted_response_jk(context, density, with_k=False)
        np.testing.assert_array_equal(exchange, jnp.zeros_like(density))
    elif consumer == "training_j":
        actual = targets._coulomb_potential_from_molecule(state, density)
    else:
        actual = targets._exchange_potential_from_molecule(state, density)
    selected = (False, True) if consumer == "training_k" else (True, False)
    assert calls == ([None] if provider_kind == "generic" else [selected])
    np.testing.assert_allclose(actual, (3 if consumer == "training_k" else 2) * density,
                               atol=1e-14, rtol=0)


@pytest.mark.parametrize("beta_occupied", [False, True])
@pytest.mark.parametrize("mode,backward", [
    ("fixed_density", "implicit"), ("self_consistent", "explicit"),
    ("self_consistent", "implicit"),
])
def test_external_unrestricted_direct_training_preserves_three_modes(
        external_reference, beta_occupied, mode, backward):
    from dataclasses import fields
    from gradscf.scf.molecules import UnrestrictedMolecule
    from gradscf.model.neural_xc.training import (
        MolecularTrainingConfig, MolecularTrainingDatum, molecular_loss,
    )

    reference, eri = external_reference
    values = {field.name: getattr(reference, field.name)
              for field in fields(UnrestrictedMolecule) if hasattr(reference, field.name)}
    occupations = jnp.asarray([[1., 0.], [float(beta_occupied), 0.]])
    values.update(mo_occ=occupations, rdm1=jax.vmap(jnp.diag)(occupations),
                  nocc_alpha=1, nocc_beta=int(beta_occupied),
                  direct_basis=reference.direct_basis)
    direct = UnrestrictedMolecule(**values)
    dense = replace(direct, direct_basis=None, rep_tensor=eri)

    def energy(p, inputs):
        spin_density, ao, weights = inputs
        return (_network_energy(p, spin_density.sum(axis=0), ao, weights, True)
                + .07 * p["scale"] * jnp.sum(spin_density[0] * spin_density[1]))

    model = dft.custom_xc(input_fn=lambda state: (state.density_matrix, state.ao, state.weights),
                          energy_fn=energy)
    params = _parameters()
    config = MolecularTrainingConfig(
        mode=mode, scf_gradient_mode=backward, e0_total_mse_weight=1.,
        scf_max_cycle=60, scf_damping=0., scf_conv_tol_density=1e-10,
        scf_conv_tol_energy=1e-12, scf_eigenvalue_jitter=0.,
        scf_implicit_diff_tolerance=1e-10, scf_implicit_diff_max_iter=40,
        scf_require_converged=True)

    def loss(state, scale):
        datum = MolecularTrainingDatum(state, target_e0_total_h=jnp.asarray(-5.))
        return molecular_loss(dict(params, scale=scale), model, datum, training_config=config)[0]

    actual = jax.jit(jax.value_and_grad(lambda scale: loss(direct, scale)))(params["scale"])
    expected = jax.jit(jax.value_and_grad(lambda scale: loss(dense, scale)))(params["scale"])
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-6)
    assert np.isfinite(np.asarray(actual)).all() and abs(float(actual[1])) > 1e-8
    step = 1e-4
    finite_difference = (loss(dense, params["scale"] + step)
                         - loss(dense, params["scale"] - step)) / (2 * step)
    np.testing.assert_allclose(actual[1], finite_difference, atol=2e-8, rtol=2e-5)
