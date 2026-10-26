"""Shared element/shell exponent scales retain coefficient-only implicit AD."""
from importlib import import_module
import runpy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.model.nnao import prepare_direct_basis


def layout(atom='C 0 0 0; H 0 0 1.09'):
    return prepare_direct_basis(atom, basis_family='szp663_direct', core_primitives=6)


def experiment(**kwargs):
    cls = vars(import_module("gradscf.model.nnao.training.objective"))['BasisRHF']
    return cls(basis_family='szp663_direct', core_primitives=6, jk_backend='df',
        implicit_tolerance=1e-8, geometry=dict(name='H2', charge=0, spin=0,
            symbols=['H', 'H'], coords_angstrom=[[0., 0., 0.], [0., 0., .74]]), **kwargs)


def test_chno_f_keys_are_unique_canonical_and_exclude_fixed_core():
    basis = layout('F 0 0 0; O 0 0 2; C 0 0 4; H 0 0 6; N 0 0 8; C 0 0 10')
    expected = (('H', 0), ('H', 1), ('C', 0), ('C', 1), ('C', 2),
                ('N', 0), ('N', 1), ('N', 2), ('O', 0), ('O', 1), ('O', 2),
                ('F', 0), ('F', 1), ('F', 2))
    assert getattr(basis, 'exponent_scale_keys', None) == expected
    # The public 442 default keeps H's single-primitive polarization fixed.
    assert prepare_direct_basis('H 0 0 0').exponent_scale_keys == (('H', 0),)


def test_zero_scales_exactly_preserve_basis_without_mutating_original():
    basis = layout(); scaled = basis.with_log_exponent_scales({key: 0. for key in basis.exponent_scale_keys})
    assert scaled is not basis and scaled.topology is basis.topology
    assert scaled.parameters.coefficients is basis.parameters.coefficients
    assert scaled.parameters.centers is basis.parameters.centers
    assert scaled.parameters.nuclear_coords is basis.parameters.nuclear_coords
    for a, b in zip(scaled.parameters.exponents, basis.parameters.exponents):
        np.testing.assert_array_equal(a, b)


def test_shell_scales_preserve_ratios_order_and_fixed_core_for_hcl():
    basis = layout('Cl 0 0 0; H 0 0 1.27; Cl 0 0 3')
    scales = {('Cl', 0): .15, ('Cl', 1): -.2, ('H', 1): .1}
    changed = basis.with_log_exponent_scales(scales)
    for atom, l, slot, original, actual in zip(basis.shell_atoms,
            basis.topology.angular_momenta, basis.slots, basis.parameters.exponents,
            changed.parameters.exponents):
        factor = np.exp(scales.get((basis.symbols[atom], l), 0.)) if slot >= 0 else 1.
        np.testing.assert_allclose(actual, np.asarray(original) * factor, atol=0, rtol=1e-14)
        np.testing.assert_allclose(actual / actual[0], original / original[0], atol=0, rtol=1e-14)
        np.testing.assert_array_equal(np.argsort(actual), np.argsort(original))
        if slot < 0:
            np.testing.assert_array_equal(actual, original)
    assert changed.parameters.coefficients is basis.parameters.coefficients


@pytest.mark.parametrize('scales', [
    {('Ne', 0): .1}, {('C', 3): .1}, {('H', 0): [0.]},
    {('H', 0): complex(.1, .2)}, {('H', 0): np.nan}, {('H', 0): np.inf},
    {('H', 0): -np.inf}, {('H', 0): 1000.}, {('H', 0): -1000.},
])
def test_invalid_or_unknown_scales_are_rejected(scales):
    with pytest.raises(ValueError):
        layout().with_log_exponent_scales(scales)


def test_traced_invalid_scale_fails_closed_without_affecting_fixed_core():
    basis = layout()
    scaled = jax.jit(lambda beta: basis.with_log_exponent_scales({('C', 0): beta}).parameters.exponents)
    actual = scaled(jnp.asarray(1000.))
    for role, slot, l, old, new in zip(basis.roles, basis.slots,
            basis.topology.angular_momenta, basis.parameters.exponents, actual):
        if role == 'valence_s' and slot == 0 and l == 0:
            # This layout has one carbon and one hydrogen: the omitted H scale stays zero.
            if len(old) == 6 and not np.array_equal(new, old):
                assert np.isnan(new).all()
        if role == 'core':
            np.testing.assert_array_equal(new, old)
    assert any(np.isnan(value).any() for value in actual)


def test_scaled_experiment_identity_rejects_unscaled_integral_cache(tmp_path):
    original = experiment(); cache = original.write_integral_cache(tmp_path / 'integrals.npz')
    zero = experiment(log_exponent_scales={('H', 0): 0.}, integral_cache=cache)
    assert zero.integral_signature == original.integral_signature
    with pytest.raises(ValueError, match='Integral cache does not match'):
        experiment(log_exponent_scales={('H', 0): .15}, integral_cache=cache)


def test_scaled_hydrogen_keeps_valid_implicit_coefficient_gradient():
    scaled = experiment(log_exponent_scales={('H', 0): .15, ('H', 1): -.1})
    outputs = scaled.layout.reference_outputs()
    value, gradient, info = scaled.evaluate(outputs)
    assert np.isfinite(value) and np.isfinite(gradient).all() and info['converged']
    direction = jnp.zeros_like(outputs).at[0, 0, 1].set(.3).at[1, 1, 1].set(.2)
    step = 1e-4
    e = lambda h: scaled.evaluate_value(outputs + h * direction)[0]
    fd = (8 * (e(step) - e(-step)) - e(2 * step) + e(-2 * step)) / (12 * step)
    np.testing.assert_allclose(jnp.sum(gradient * direction), fd, atol=2e-6, rtol=2e-5)


def test_experiment_forwards_optional_fixed_metric_factor(monkeypatch):
    from gradscf.integrals.molecular.density_fitting import AuxiliaryPlan
    original = AuxiliaryPlan.factors; sentinel = object(); received = []
    def capture(self, parameters, auxiliary_parameters, **kwargs):
        received.append(kwargs.get('metric_factor'))
        return original(self, parameters, auxiliary_parameters)
    monkeypatch.setattr(AuxiliaryPlan, 'factors', capture)
    experiment(df_metric_factor=sentinel)
    assert received == [sentinel]
