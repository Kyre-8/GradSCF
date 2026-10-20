"""Contracts for complete XC recipes and their local JAX contributions."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import xc


def _recipe(name):
    assert hasattr(xc, "parse_recipe"), "dft.xc must expose complete XC recipes"
    return xc.parse_recipe(name)


def test_standard_recipe_records_hf_and_semilocal_coefficients():
    recipe = _recipe("PBE0")
    assert recipe.name == "pbe0"
    assert recipe.exact_exchange.full_range == pytest.approx(0.25)
    assert recipe.exact_exchange.short_range == 0.0
    assert recipe.exact_exchange.long_range == 0.0
    assert [(t.name, t.coefficient) for t in recipe.semilocal] == [
        ("gga_x_pbe", 0.75), ("gga_c_pbe", 1.0),
    ]
    assert recipe.references


def test_complete_recipe_keeps_required_nonlocal_terms():
    recipe = _recipe("wb97xd")
    assert recipe.exact_exchange.omega == pytest.approx(0.2)
    assert recipe.exact_exchange.short_range == pytest.approx(0.222036)
    assert recipe.exact_exchange.long_range == 1.0
    assert recipe.dispersion is not None
    assert recipe.semilocal[0].fixed_omega == pytest.approx(0.2)
    with pytest.raises(NotImplementedError, match="dispersion"):
        recipe.require_complete_support()


def test_catalog_distinguishes_recipe_from_installed_local_channel():
    recipe = _recipe("m06-2x")
    assert recipe.exact_exchange.full_range == pytest.approx(0.54)
    assert recipe.kind == "MGGA"
    assert recipe.references
    recipe = _recipe("b2plyp")
    assert recipe.pt2 is not None
    with pytest.raises(NotImplementedError, match="PT2"):
        recipe.require_complete_support()


def test_custom_expression_preserves_signed_coefficients_and_legacy_order():
    recipe = _recipe("0.25*HF + 0.8*gga_x_pbe - 0.05*gga_x_pbe + gga_c_pbe")
    assert recipe.exact_exchange.full_range == pytest.approx(0.25)
    assert sum(t.coefficient for t in recipe.semilocal if t.name == "gga_x_pbe") == pytest.approx(0.75)
    assert [t.name for t in xc.parse_xc("b3lyp")] == [
        "hf", "lda_x", "gga_x_b88", "lda_c_vwn_rpa", "gga_c_lyp",
    ]


def test_custom_expression_keeps_scientific_notation_and_compact_subtraction():
    recipe = _recipe("1e-3*HF + gga_x_pbe-1e-3*gga_x_pbe + gga_c_pbe")
    assert recipe.exact_exchange.full_range == pytest.approx(.001)
    assert sum(term.coefficient for term in recipe.semilocal if term.name == "gga_x_pbe") == pytest.approx(.999)


def test_explicit_rsh_expression_preserves_pyscf_alpha_beta_convention():
    recipe = _recipe("RSH(0.2, 1.0, -0.75)")
    assert recipe.kind == "HF"
    assert recipe.exact_exchange.omega == pytest.approx(0.2)
    assert recipe.exact_exchange.short_range == pytest.approx(0.25)
    assert recipe.exact_exchange.long_range == pytest.approx(1.0)
    with pytest.raises(ValueError, match="omega"):
        _recipe("SR_HF(0.2) + LR_HF(0.3)")


def test_zero_omega_rsh_operator_reduces_to_full_range_hf():
    recipe = _recipe("0.1*HF + RSH(0, 1, -0.75)")
    assert recipe.exact_exchange.full_range == pytest.approx(.35)
    assert recipe.exact_exchange.short_range == 0.
    assert recipe.exact_exchange.long_range == 0.
    assert recipe.exact_exchange.omega is None


def test_local_kernel_factories_share_full_spin_gradient_variables():
    assert hasattr(xc, "point_energy"), "local point energy must live in dft.xc"
    evaluate = importlib.import_module("gradscf.dft.xc.evaluate")
    y = jnp.asarray([0.31, 0.23, 0.12, -0.04, 0.03, -0.02, 0.11, 0.05])
    point = evaluate.point_energy("pbe", "GGA")
    grid = jnp.stack([y, 1.1*y])
    energies, gradients = evaluate.grid_value_and_grad("pbe", "GGA")(grid)
    assert point is evaluate.point_energy("pbe", "GGA")
    np.testing.assert_allclose(energies, jax.vmap(point)(grid), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(gradients, jax.vmap(jax.grad(point))(grid), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(evaluate.grid_hessian("pbe", "GGA")(grid),
                               jax.vmap(jax.hessian(point))(grid), rtol=2e-12, atol=2e-12)


def test_unrestricted_bundle_preserves_nonparallel_sigma_ab():
    assert hasattr(xc, "eval_xc_energy_density_unrestricted")
    rho_a, rho_b = jnp.asarray(0.31), jnp.asarray(0.23)
    grad_a = jnp.asarray([0.12, -0.04, 0.03])
    grad_b = jnp.asarray([-0.02, 0.11, 0.05])
    bundle = xc.RestrictedFeatureBundle(
        rho_a, rho_b, jnp.dot(grad_a, grad_a), jnp.dot(grad_a, grad_b),
        jnp.dot(grad_b, grad_b), jnp.asarray(0.0), jnp.asarray(0.0),
    )
    value = xc.eval_xc_energy_density_unrestricted("pbe", bundle)
    explicit = xc.eval_xc_energy_density_unrestricted_from_density_gradients(
        "pbe", rho_a, rho_b, grad_a, grad_b,
    )
    np.testing.assert_allclose(value, explicit, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("case", ["parallel", "zero_alpha"])
def test_parallel_spin_bundle_keeps_cartesian_transverse_curvature(case):
    ra, rb = jnp.asarray(.31), jnp.asarray(.23)

    def gradients(t):
        if case == "parallel":
            return jnp.asarray([1., 0., 0.]), jnp.asarray([1., t, 0.])
        return jnp.asarray([t, 0., 0.]), jnp.asarray([0., 1., 0.])

    def from_bundle(t):
        ga, gb = gradients(t)
        features = xc.RestrictedFeatureBundle(ra, rb, jnp.dot(ga, ga), jnp.dot(ga, gb),
                                               jnp.dot(gb, gb), jnp.asarray(0.), jnp.asarray(0.))
        return xc.eval_xc_energy_density_unrestricted("pbe", features)

    def from_cartesian(t):
        ga, gb = gradients(t)
        return xc.eval_xc_energy_density_unrestricted_from_density_gradients(
            "pbe", ra, rb, ga, gb,
        )

    actual = jax.grad(jax.grad(from_bundle))(jnp.asarray(0.))
    expected = jax.grad(jax.grad(from_cartesian))(jnp.asarray(0.))
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-12)


def test_restricted_zero_gradient_bundle_keeps_cartesian_curvature():
    rho = jnp.asarray(.54)

    def from_bundle(t):
        features = xc.restricted_feature_bundle_from_rho_grad_tau(rho, jnp.asarray([t, 0., 0.]))
        return xc.eval_xc_energy_density("gga_x_pbe", features)

    def from_cartesian(t):
        half_gradient = jnp.asarray([t/2, 0., 0.])
        return xc.eval_xc_energy_density_unrestricted_from_density_gradients(
            "gga_x_pbe", rho/2, rho/2, half_gradient, half_gradient,
        )

    actual = jax.grad(jax.grad(from_bundle))(jnp.asarray(0.))
    expected = jax.grad(jax.grad(from_cartesian))(jnp.asarray(0.))
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-12)


def test_fixed_omega_recipe_rejects_a_different_runtime_parameter():
    assert hasattr(xc, "point_energy")
    y = jnp.asarray([0.31, 0.23, 0.12, -0.04, 0.03, -0.02, 0.11, 0.05])
    with pytest.raises(ValueError, match="fixed.*omega|omega.*fixed"):
        xc.point_energy("wb97xd", "GGA")(y, omega=0.4)


def test_nonfinite_energy_is_reported_instead_of_replaced_with_zero(monkeypatch):
    assert hasattr(xc, "point_energy")
    backend = importlib.import_module("gradscf.dft.xc._jax_xc")

    class NonfiniteModule:
        @staticmethod
        def lda_x(*, polarized=False):
            return lambda rho_fn, r: jnp.asarray(float("nan"))

    monkeypatch.setattr(backend, "load_jax_xc", lambda: (NonfiniteModule(), "test"))
    bundle = xc.RestrictedFeatureBundle(*[jnp.asarray(x) for x in (.3, .3, 0, 0, 0, 0, 0)])
    with pytest.raises(FloatingPointError, match="non-finite"):
        xc.eval_xc_energy_density("lda_x", bundle)


def test_public_info_and_available_distinguish_local_backend_and_complete_terms(monkeypatch):
    assert hasattr(xc, "info"), "XC library must expose recipe and local availability metadata"
    assert hasattr(xc, "available")
    metadata = xc.info("wb97xd")
    assert metadata["recipe"].dispersion is not None
    assert "dispersion" in metadata["extra_requirements"]
    backend = importlib.import_module("gradscf.dft.xc._jax_xc")
    monkeypatch.setattr(backend, "load_jax_xc", lambda: (_ for _ in ()).throw(backend.MissingJAXXCError("missing")))
    assert not xc.available("pbe")


def test_backend_diagnostics_preserve_underlying_binary_loader_error(monkeypatch):
    backend = importlib.import_module("gradscf.dft.xc._jax_xc")

    def unavailable(name):
        raise OSError("GLIBC_2.34 not found in the active runtime")

    monkeypatch.setattr(backend.importlib, "import_module", unavailable)
    metadata = xc.jax_xc_backend_info()
    assert not metadata["available"]
    assert "GLIBC_2.34" in metadata["reason"]


def test_traditional_object_string_is_its_explicit_specification():
    assert str(xc.TraditionalXCFunctional("pbe")) == "pbe"


def test_unknown_hybrid_component_cannot_be_used_as_complete_scf_recipe():
    # The former local wrapper assigned PBE0_13 the PBE0 local weights and
    # silently lost its HF metadata. A raw channel remains a distinct concept.
    with pytest.raises(ValueError, match="complete hybrid"):
        xc.parse_recipe("hyb_gga_xc_pbe0_13")
    assert xc.resolve_xc_component_name("hyb_gga_xc_pbe0_13") == "hyb_gga_xc_pbe0_13"


@pytest.mark.parametrize("spec", ["scanl", "r2scanl"])
def test_laplacian_functionals_reject_the_tau_only_point_contract(spec):
    assert not xc.available(spec)
    with pytest.raises(NotImplementedError, match="Laplacian"):
        xc.point_energy(spec, "MGGA")(jnp.ones(10))
