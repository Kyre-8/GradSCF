from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals


@pytest.mark.parametrize("nuclear_charge", [0, 1])
def test_prepare_basis_preserves_molecule_spec_nuclear_charges(nuclear_charge):
    from gradscf.data.molecule import parse_molecule_spec

    spec = replace(parse_molecule_spec("He 0 0 0"), charges=jnp.array([nuclear_charge]))
    topology, params = integrals.prepare_basis(spec, {"He": [[0, [1., 1.]]]})
    assert topology.nuclear_charges == (nuclear_charge,)
    expected = -nuclear_charge * 2 * np.sqrt(2 / np.pi)
    for backend in ("native", "jax_reference"):
        result = integrals.make_plan(topology, backend=backend).evaluate("nuclear", params)
        np.testing.assert_allclose(result, [[expected]], atol=1e-12)


@pytest.mark.parametrize("charges", [[1.5], [np.nan], [np.inf], [-1], [2**31], []])
def test_prepare_basis_rejects_unsupported_nuclear_charges(charges):
    from gradscf.data.molecule import parse_molecule_spec

    spec = replace(parse_molecule_spec("He 0 0 0"), charges=jnp.asarray(charges))
    with pytest.raises(ValueError, match="nuclear charges"):
        integrals.prepare_basis(spec, {"He": [[0, [1., 1.]]]})


@pytest.mark.parametrize("backend", ["native", "jax_reference"])
def test_plan_rejects_invalid_dipole_origin_shape(backend):
    topology, params = integrals.prepare_basis("He 0 0 0", {"He": [[0, [1., 1.]]]})
    plan = integrals.make_plan(topology, backend=backend)
    with pytest.raises(ValueError, match="origin must have shape"):
        plan.evaluate("dipole", params, origin=jnp.array([.2]))


@pytest.mark.parametrize("operator", ["overlap", "kinetic", "nuclear", "dipole"])
def test_reference_one_electron_does_not_build_eri_layout(monkeypatch, operator):
    from gradscf.integrals import plan as plan_module

    topology, params = integrals.prepare_basis("He 0 0 0", {"He": [[0, [1., 1.]]]})
    constructor = plan_module.CartesianBasis

    def check_layout(*args, **kwargs):
        basis = constructor(*args, **kwargs)
        assert not basis.quartet_groups
        assert not basis.shell_quartet_groups
        return basis

    monkeypatch.setattr(plan_module, "CartesianBasis", check_layout)
    result = integrals.make_plan(topology, backend="jax_reference").evaluate(operator, params)
    assert np.all(np.isfinite(result))


def test_raw_basis_parameters_have_differentiable_normalization():
    assert hasattr(integrals, "prepare_basis")
    topology, params = integrals.prepare_basis(
        atom="H 0 0 0", basis={"H": [[0, [1.2, .7], [.3, .4]]]}, spin=1,
    )
    plan = integrals.make_plan(topology, backend="jax_reference")

    def value(exponents, coefficients):
        p = replace(params, exponents=(exponents,), coefficients=(coefficients,))
        return plan.evaluate("kinetic", p)[0, 0]

    a, c = params.exponents[0], params.coefficients[0]
    ga, gc = jax.grad(value, argnums=(0, 1))(a, c)
    step = 1e-5
    for idx in range(2):
        da = jnp.zeros_like(a).at[idx].set(step)
        dc = jnp.zeros_like(c).at[idx, 0].set(step)
        np.testing.assert_allclose(ga[idx], (value(a+da,c)-value(a-da,c))/(2*step), atol=1e-7)
        np.testing.assert_allclose(gc[idx,0], (value(a,c+dc)-value(a,c-dc))/(2*step), atol=1e-7)
    np.testing.assert_allclose(plan.evaluate("overlap", params), [[1.]], atol=1e-12)
    np.testing.assert_allclose(jax.jit(value)(a,c), value(a,c), atol=1e-12)


def test_plan_does_not_cache_parameter_values():
    assert hasattr(integrals, "prepare_basis")
    topology, params = integrals.prepare_basis(atom="H 0 0 0", basis={"H": [[0, [1., 1.]]]}, spin=1)
    plan = integrals.make_plan(topology, backend="jax_reference")
    p2 = replace(params, exponents=(params.exponents[0] * 2,))
    np.testing.assert_allclose(plan.evaluate("kinetic", p2), 2*plan.evaluate("kinetic", params), atol=1e-12)


def test_native_capabilities_distinguish_geometry_and_basis_derivatives():
    assert hasattr(integrals, "backend_capabilities")
    caps = integrals.backend_capabilities("native")
    assert caps.supports("eri", derivative_order=0)
    assert not caps.supports("eri", variable="exponents", derivative_order=1)
    assert caps.supports("kinetic", variable="centers", derivative_order=1)
    assert caps.supports("nuclear", variable="nuclear_coords", derivative_order=1)
    assert caps.supports("eri", variable="centers", derivative_order=2)
    assert caps.supports("dipole", variable="origin", derivative_order=2)
    assert not caps.supports("overlap", variable="origin", derivative_order=1)
    assert not caps.supports("eri", variable="centers", derivative_order=3)
    assert not caps.supports("eri", variable="centers", derivative_order=2, layout="s4")
    assert not caps.supports("ecp", variable="centers", derivative_order=2)


def test_reference_eri_parameter_gradient_and_center_gradient():
    topology, params = integrals.prepare_basis(
        atom="H 0 0 0; H 0 0 1.4", unit="Bohr", basis={"H": [[0, [1.2, .7], [.3, .4]]]},
    )
    plan = integrals.make_plan(topology, backend="jax_reference")
    def eri(a):
        p = replace(params, exponents=(a, params.exponents[1]))
        return plan.evaluate("eri", p)[0, 0, 1, 1]
    a = params.exponents[0]
    da = jnp.array([1e-5, 0.])
    np.testing.assert_allclose(jax.grad(eri)(a)[0], (eri(a+da)-eri(a-da))/2e-5, atol=1e-7)
    def ovlp(centers):
        return plan.evaluate("overlap", replace(params, centers=centers))[0, 1]
    delta = jnp.zeros_like(params.centers).at[0, 2].set(1e-5)
    np.testing.assert_allclose(jax.grad(ovlp)(params.centers)[0,2],
                               (ovlp(params.centers+delta)-ovlp(params.centers-delta))/2e-5, atol=1e-7)


def test_plan_native_normalization_and_floating_centers_match_reference():
    topology, params = integrals.prepare_basis(
        atom="O 0 0 0; H 0 .75 .58; H 0 -.75 .58", basis="sto-3g",
    )
    native = integrals.make_plan(topology, backend="native")
    reference = integrals.make_plan(topology, backend="jax_reference")
    for operator in ("overlap", "kinetic", "nuclear", "dipole", "eri"):
        np.testing.assert_allclose(native.evaluate(operator, params), reference.evaluate(operator, params),
                                   atol=2e-7, rtol=0)
    p = replace(params, centers=params.centers.at[0, 0].add(.1))
    np.testing.assert_allclose(native.evaluate("nuclear", p), reference.evaluate("nuclear", p), atol=2e-7, rtol=0)
    np.testing.assert_allclose(jax.jit(lambda p: native.evaluate("overlap", p))(params),
                               native.evaluate("overlap", params), atol=1e-12)
