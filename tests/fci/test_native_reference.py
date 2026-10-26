"""Exact native CAS preparation: CPU float64, energies in Hartree."""
from dataclasses import replace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import fci


@pytest.fixture(scope="module", params=("rhf", "rohf"))
def native_source(request):
    from gradscf import gto, integrals, scf
    from gradscf.integrals.backends.native.jk import NativeDirectBasis

    open_shell = request.param == "rohf"
    molecule = gto.M(atom="Li 0 0 0" if open_shell else "Li 0 0 0; H 0 0 1.6",
                     basis="sto-3g", spin=int(open_shell), cart=True)
    factory = scf.ROHF if open_shell else scf.RHF
    options = {} if open_shell else dict(jk_backend="direct", direct_scf_tol=0.)
    source = factory(molecule, integral_backend="native", execution_device="cpu",
                     max_memory=128., **options,
                     conv_tol=1e-13, conv_tol_grad=1e-10,
                     conv_tol_density=1e-11, max_cycle=200).run()
    assert source.converged
    if open_shell:
        # ROKS currently solves a full-integral SCF. Adapt the representation
        # of that converged common frame exactly; this does not claim that
        # the ROHF SCF solver itself executes a direct J/K path.
        topology, parameters = integrals.prepare_basis(source._spec(), molecule.basis, cart=True)
        provider = NativeDirectBasis(integrals.make_plan(topology, backend="native"), parameters)
        source._scf_inputs = replace(source._scf_inputs, direct_basis=provider,
                                     eri=jnp.zeros((0, 0, 0, 0), dtype=jnp.float64))
    inputs = source._scf_inputs
    assert inputs.direct_basis is not None and inputs.df_factors is None
    assert inputs.eri is None or inputs.eri.size == 0
    assert getattr(inputs, "eri_pair_matrix", None) is None
    # A tiny full AO tensor is constructed before the prohibition below and
    # exists solely as an independent oracle in the same spatial MO frame.
    basis = inputs.direct_basis
    eri = basis.plan.evaluate("eri", basis.parameters)
    coefficients = source.mo_coeff
    h = coefficients.T @ inputs.hcore @ coefficients
    g = jnp.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, coefficients, coefficients,
                   coefficients, coefficients, precision="highest")
    return source, h, g


def _forbid_full_eri(source, monkeypatch):
    from gradscf.integrals.plan import IntegralPlan
    from gradscf.scf.inputs.types import RKSIntegralInputs

    def forbidden(*args, **kwargs):
        pytest.fail("Native FCI must not materialize full AO ERIs or an AO-pair matrix")

    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == "eri":
            forbidden()
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, "evaluate", checked)
    monkeypatch.setattr(RKSIntegralInputs, "response_eri_pair_matrix", forbidden)
    monkeypatch.setattr(type(source._scf_inputs.direct_basis),
                        "response_eri_pair_matrix", forbidden)
    monkeypatch.setattr(source, "to_reference", forbidden)


@pytest.mark.parametrize("freeze", (False, True), ids=("no-core", "frozen-core"))
def test_native_active_reference_matches_dense_same_frame(native_source, freeze, monkeypatch):
    from gradscf.fci import reference

    source, h, g = native_source
    core = (0,) if freeze else ()
    active = (source.mo_coeff.shape[1] - 1, 2, 1) if freeze else (2, 0, 1)
    expected = fci.fold_core(h, g, core=core, active=active,
                             ecore=source._scf_inputs.nuclear_repulsion)
    _forbid_full_eri(source, monkeypatch)
    transform = reference.transform_integrals
    seen = []

    def selected(hcore, coefficients, **kwargs):
        seen.append((coefficients.shape, kwargs.get("direct_basis"),
                     kwargs.get("max_memory")))
        return transform(hcore, coefficients, **kwargs)

    monkeypatch.setattr(reference, "transform_integrals", selected)
    actual = reference.active_reference(source, core, active, max_memory=23.5)
    assert seen == [((source.mo_coeff.shape[0], len(active)),
                     source._scf_inputs.direct_basis, 23.5)]
    assert actual[1].shape == (len(active),) * 4
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, atol=3e-12, rtol=2e-11)


@pytest.mark.parametrize("freeze", (False, True), ids=("no-core", "frozen-core"))
def test_native_molecular_fci_matches_dense_reference(native_source, freeze, monkeypatch):
    source, h, g = native_source
    core = (0,) if freeze else ()
    active = (source.mo_coeff.shape[1] - 1, 2, 1) if freeze else (2, 0, 1)
    reference = fci.FCIReference(h, g, source.mol.nelectron,
                                source._scf_inputs.nuclear_repulsion)
    expected = fci.FCI(reference, core=core, active=active, solver="dense").run()
    _forbid_full_eri(source, monkeypatch)
    actual = fci.FCI(source, core=core, active=active, solver="dense").run()
    assert actual.converged
    assert actual.space.nelec == expected.space.nelec
    np.testing.assert_allclose(actual.e_tot, expected.e_tot, atol=2e-11, rtol=0)
    np.testing.assert_allclose(actual.make_rdm1(), expected.make_rdm1(), atol=2e-10, rtol=0)


@pytest.mark.parametrize("budget", (None, 23.5), ids=("inherited", "override"))
def test_native_fci_inherits_or_overrides_transform_budget(native_source, monkeypatch, budget):
    api = importlib.import_module("gradscf.fci.api")
    source, _, _ = native_source
    active_reference = api.active_reference
    seen = []

    def prepare(source, core, active, **kwargs):
        seen.append(kwargs.get("max_memory"))
        return active_reference(source, core, active, **kwargs)

    monkeypatch.setattr(api, "active_reference", prepare)
    _forbid_full_eri(source, monkeypatch)
    actual = fci.FCI(source, core=1, active=(2, 1), solver="dense",
                     max_memory=budget).run()
    assert actual.converged
    assert seen == [source.max_memory if budget is None else budget]
    actual.max_memory = 22.
    with pytest.raises(RuntimeError, match="changed"):
        actual.make_rdm1()


def test_native_fci_budget_precedes_core_jk_and_eri(native_source, monkeypatch):
    from gradscf.ao2mo import native

    source, _, _ = native_source
    _forbid_full_eri(source, monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("An insufficient FCI budget reached an integral evaluation")

    monkeypatch.setattr(type(source._scf_inputs.direct_basis), "get_jk", forbidden)
    monkeypatch.setattr(native, "evaluate_eri_slab", forbidden)
    with pytest.raises(ValueError, match="(?i)(budget|memory|resident|minimum)"):
        fci.FCI(source, core=1, active=(2, 1), max_memory=.0001).run()


def test_legacy_jax_direct_budget_precedes_packed_eri_materialization(monkeypatch):
    from gradscf import gto, scf
    from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis
    from gradscf.scf.inputs.types import RKSIntegralInputs

    molecule = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", cart=True)
    source = scf.RHF(molecule, integral_backend="jax", execution_device="cpu",
                     jk_backend="direct", direct_scf_tol=0., max_memory=128.,
                     conv_tol=1e-12, conv_tol_grad=1e-10,
                     conv_tol_density=1e-11).run()
    assert source.converged
    assert not isinstance(source._scf_inputs.direct_basis,
                          (NativeDirectBasis, ProjectedNativeDirectBasis))

    def forbidden(*args, **kwargs):
        pytest.fail("An insufficient FCI budget reached legacy packed AO ERI materialization")

    monkeypatch.setattr(RKSIntegralInputs, "response_eri_pair_matrix", forbidden)
    with pytest.raises(ValueError, match="(?i)(budget|memory|resident|minimum)"):
        fci.FCI(source, max_memory=.0001).run()


def test_explicit_reference_budget_precedes_core_folding(monkeypatch):
    from gradscf.fci import reference

    h = jnp.diag(jnp.array([-1.3, -.4, .2, .7]))
    g = jnp.zeros((4, 4, 4, 4), dtype=jnp.float64)
    source = fci.FCIReference(h, g, (2, 2), .5)

    def forbidden(*args, **kwargs):
        pytest.fail("An insufficient explicit FCI budget reached core-folding allocations")

    monkeypatch.setattr(reference, "fold_core", forbidden)
    with pytest.raises(ValueError, match="(?i)(budget|memory|resident|minimum)"):
        fci.FCI(source, core=(0,), active=(2, 1), max_memory=.0001).run()


@pytest.mark.parametrize("core,active,match", [
    ((0,), (0, 1), "disjoint"),
    ((0, 1, 2), (3, 4), "[Ee]lectron"),
    ((), (0,), "[Ee]lectron"),
])
def test_native_fci_static_selection_rejected_before_integrals(
        native_source, monkeypatch, core, active, match):
    from gradscf.ao2mo import native

    source, _, _ = native_source
    _forbid_full_eri(source, monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("An invalid active/core selection reached an integral evaluation")

    monkeypatch.setattr(type(source._scf_inputs.direct_basis), "get_jk", forbidden)
    monkeypatch.setattr(native, "evaluate_eri_slab", forbidden)
    with pytest.raises(ValueError, match=match):
        fci.FCI(source, core=core, active=active).run()


@pytest.mark.parametrize("invalid", ("complex", "nonorthogonal", "spin-frames"))
def test_native_fci_preserves_common_real_orthonormal_source_guard(
        native_source, monkeypatch, invalid):
    source, _, _ = native_source
    _forbid_full_eri(source, monkeypatch)
    coefficients = source.mo_coeff
    if invalid == "complex":
        coefficients = coefficients.astype(jnp.complex128) + .01j
    elif invalid == "nonorthogonal":
        coefficients = coefficients * 1.1
    else:
        coefficients = jnp.stack((coefficients, coefficients))
    monkeypatch.setattr(source, "mo_coeff", coefficients)
    error = NotImplementedError if invalid == "spin-frames" else ValueError
    match = "common spatial" if invalid == "spin-frames" else "real S-orthonormal"
    with pytest.raises(error, match=match):
        fci.FCI(source, core=1, active=(2, 1)).run()


def test_native_reference_empty_active_block_uses_only_core_jk(native_source, monkeypatch):
    from gradscf.ao2mo import native
    from gradscf.fci import reference

    source, h, g = native_source
    expected = fci.fold_core(h, g, core=(0,), active=(),
                             ecore=source._scf_inputs.nuclear_repulsion)
    _forbid_full_eri(source, monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("An empty active space must not evaluate a native ERI slab")

    monkeypatch.setattr(native, "evaluate_eri_slab", forbidden)
    actual = reference.active_reference(source, (0,), (), max_memory=4.)
    assert actual[0].shape == (0, 0) and actual[1].shape == (0, 0, 0, 0)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, atol=3e-12, rtol=2e-11)


@pytest.mark.parametrize("projected", (False, True), ids=("native", "projected-native"))
def test_pure_native_active_reference_keeps_coordinate_coefficient_and_projection_response(
        monkeypatch, projected):
    from gradscf import integrals
    from gradscf.fci import reference
    from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis
    from gradscf.integrals.plan import IntegralPlan

    topology, parameters = integrals.prepare_basis(
        "H 0 0 0; H .2 -.1 .8", basis="3-21g", cart=True)
    plan = integrals.make_plan(topology, backend="native")
    nnuc = len(topology.nuclear_charges)
    coordinates = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
    rng = np.random.default_rng(2619)
    nao = topology.nao - int(projected)
    coefficients = jnp.asarray(rng.normal(scale=.3, size=(nao, 3)))
    point = (coordinates, coefficients)
    if projected:
        transform = jnp.eye(topology.nao)[:, :nao] + .03
        point += (transform,)
    direction = tuple(jnp.asarray(rng.normal(scale=.04, size=value.shape)) for value in point)

    def prepare(r, c, transform=None, *, dense):
        # Rebuild both the provider and all one-electron/nuclear inputs at
        # every geometry. This is a fixed, supplied MO frame, not a cached
        # SCF source derivative or an orbital-relaxation calculation.
        moving = replace(parameters, nuclear_coords=r[:nnuc], centers=r[nnuc:])
        provider = NativeDirectBasis(plan, moving)
        hcore = plan.evaluate("kinetic", moving) + plan.evaluate("nuclear", moving)
        enuc = 1. / jnp.linalg.norm(r[0] - r[1])
        primitive_coefficients = c
        if transform is not None:
            provider = ProjectedNativeDirectBasis(provider, transform)
            hcore = transform.T @ hcore @ transform
            primitive_coefficients = transform @ c
        if dense:
            eri = plan.evaluate("eri", moving)
            g = jnp.einsum("pqrs,pi,qj,rk,sl->ijkl", eri,
                           primitive_coefficients, primitive_coefficients,
                           primitive_coefficients, primitive_coefficients, precision="highest")
            return fci.fold_core(c.T @ hcore @ c, g, core=(0,), active=(2, 1), ecore=enuc)
        return reference._active_from_orbitals(
            hcore, c, core=(0,), active=(2, 1), ecore=enuc,
            direct_basis=provider, max_memory=4.)

    def loss(*values, dense):
        h, g, ecore = prepare(*values, dense=dense)
        return (jnp.sum(jnp.sin(h) + .1*h*h)
                + .3*jnp.sum(jnp.sin(g) + .2*g*g) + .7*ecore)

    argnums = tuple(range(len(point)))
    oracle = lambda *values: loss(*values, dense=True)
    oracle_gradient = jax.grad(oracle, argnums=argnums)
    expected_integrals = prepare(*point, dense=True)
    expected = (jax.value_and_grad(oracle, argnums=argnums)(*point),
                jax.jvp(oracle, point, direction),
                jax.jvp(oracle_gradient, point, direction)[1])
    evaluate = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == "eri":
            pytest.fail("Pure native CAS response must not materialize a full AO ERI tensor")
        return evaluate(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, "evaluate", checked)
    for result, expected_value in zip(prepare(*point, dense=False), expected_integrals):
        np.testing.assert_allclose(result, expected_value, atol=3e-12, rtol=2e-11)
    actual = lambda *values: loss(*values, dense=False)
    gradient = jax.jit(jax.grad(actual, argnums=argnums))
    observed = (jax.jit(jax.value_and_grad(actual, argnums=argnums))(*point),
                jax.jit(lambda values, tangent: jax.jvp(actual, values, tangent))(point, direction),
                jax.jit(lambda values, tangent: jax.jvp(gradient, values, tangent)[1])(point, direction))
    for result, expected_value in zip(jax.tree.leaves(observed), jax.tree.leaves(expected)):
        np.testing.assert_allclose(result, expected_value, atol=3e-9, rtol=3e-8)
    for component in observed[0][1]:
        assert float(jnp.linalg.norm(component)) > 1e-8
    for step in (3e-4, 1e-4):
        plus = gradient(*(value + step*delta for value, delta in zip(point, direction)))
        minus = gradient(*(value - step*delta for value, delta in zip(point, direction)))
        finite_difference = jax.tree.map(lambda a, b: (a-b)/(2*step), plus, minus)
        for result, fd in zip(observed[2], finite_difference):
            np.testing.assert_allclose(result, fd, atol=3e-7, rtol=3e-6)
