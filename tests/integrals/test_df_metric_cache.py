"""Auxiliary Coulomb whitening reuse at a fixed auxiliary basis and geometry."""
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals import prepare_basis
from gradscf.integrals.molecular.density_fitting import make_auxiliary_plan


@pytest.fixture
def small_plan():
    atom = 'H 0 0 0; H 0 0 .8'
    top, p = prepare_basis(atom, '3-21g', cart=False)
    aux, a = prepare_basis(atom, 'sto-3g', cart=False)
    return make_auxiliary_plan(top, aux), p, a


def cached_metric(plan, p, a, **kwargs):
    assert hasattr(plan, 'metric_factor'), 'Reusable auxiliary metric factor is not implemented.'
    return plan.metric_factor(p, a, **kwargs)


def test_cached_native_factors_only_recompute_three_center_integrals(small_plan, monkeypatch):
    from gradscf.integrals.backends.native import density_fitting as native_compact
    plan, p, a = small_plan
    reference = plan.factors(p, a)
    original = native_compact.evaluate
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs['layout'])
        assert kwargs['layout'] in (2, 3), 'Never request four-center ERIs.'
        return original(*args, **kwargs)

    monkeypatch.setattr(native_compact, 'evaluate', observed)
    metric = cached_metric(plan, p, a)
    assert calls == [2]
    calls.clear()
    actual = plan.factors(p, a, metric_factor=metric)
    assert calls == [3]
    np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-12)


def test_orbital_exponent_changes_reuse_auxiliary_metric(small_plan):
    plan, p, a = small_plan
    metric = cached_metric(plan, p, a)
    changed = replace(p, exponents=tuple(alpha * 1.15 for alpha in p.exponents))
    np.testing.assert_allclose(plan.factors(changed, a, metric_factor=metric),
                               plan.factors(changed, a), atol=1e-12, rtol=1e-12)


def test_contracted_and_primitive_plans_share_the_same_metric(small_plan):
    from gradscf.integrals.basis.contraction import primitive_basis
    plan, p, a = small_plan
    metric = cached_metric(plan, p, a)
    primitive, pp = primitive_basis(plan.orbital, p)
    primitive_plan = make_auxiliary_plan(primitive, plan.auxiliary)
    np.testing.assert_allclose(primitive_plan.factors(pp, a, metric_factor=metric),
                               primitive_plan.factors(pp, a), atol=1e-12, rtol=1e-12)
    assert not metric.matrix.flags.writeable
    with pytest.raises(ValueError, match='lindep'):
        primitive_plan.factors(pp, a, metric_factor=metric, lindep=1e-10)


@pytest.mark.parametrize('change', ['aux_alpha', 'aux_coeff', 'aux_center', 'aux_nuclear',
                                    'orbital_center', 'orbital_nuclear', 'aux_topology'])
def test_changed_auxiliary_or_geometry_is_rejected(small_plan, change):
    plan, p, a = small_plan
    metric = cached_metric(plan, p, a)
    if change == 'aux_alpha':
        a = replace(a, exponents=(a.exponents[0] * 1.01, *a.exponents[1:]))
    elif change == 'aux_coeff':
        a = replace(a, coefficients=(a.coefficients[0].at[0, 0].add(.01), *a.coefficients[1:]))
    elif change == 'aux_center':
        a = replace(a, centers=a.centers.at[0, 0].add(.01))
    elif change == 'aux_nuclear':
        a = replace(a, nuclear_coords=a.nuclear_coords.at[0, 0].add(.01))
    elif change == 'orbital_center':
        p = replace(p, centers=p.centers.at[0, 0].add(.01))
    elif change == 'orbital_nuclear':
        p = replace(p, nuclear_coords=p.nuclear_coords.at[0, 0].add(.01))
    else:
        aux = replace(plan.auxiliary, angular_momenta=(1, *plan.auxiliary.angular_momenta[1:]))
        plan = make_auxiliary_plan(plan.orbital, aux)
    with pytest.raises(ValueError, match='auxiliary|geometry|signature'):
        plan.factors(p, a, metric_factor=metric)


def test_eigenmode_fallback_is_reusable(small_plan, monkeypatch):
    import scipy.linalg
    plan, p, a = small_plan

    def dependent(*args, **kwargs):
        raise np.linalg.LinAlgError('Force the existing eigenmode whitening path.')

    monkeypatch.setattr(scipy.linalg, 'cholesky', dependent)
    metric = cached_metric(plan, p, a)
    np.testing.assert_allclose(plan.factors(p, a, metric_factor=metric),
                               plan.factors(p, a), atol=1e-12, rtol=1e-12)


def test_dependent_metric_drops_only_nonpositive_modes_and_reuses_rank(small_plan, monkeypatch):
    from gradscf.integrals.backends.native import density_fitting as native_compact
    plan, p, a = small_plan
    original = native_compact.evaluate

    def dependent_metric(*args, **kwargs):
        return jnp.ones((2, 2)) if kwargs['layout'] == 2 else original(*args, **kwargs)

    monkeypatch.setattr(native_compact, 'evaluate', dependent_metric)
    metric = cached_metric(plan, p, a)
    assert metric.method == 'eigh' and metric.matrix.shape == (1, 2)
    cached = plan.factors(p, a, metric_factor=metric)
    assert cached.shape[0] == 1
    np.testing.assert_allclose(cached, plan.factors(p, a), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('lindep', [0., -1., np.nan, np.inf])
def test_invalid_lindep_is_rejected_without_integral_evaluation(small_plan, monkeypatch, lindep):
    from gradscf.integrals.backends.native import density_fitting as native_compact
    plan, p, a = small_plan

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid tolerance must be rejected before native evaluation.')

    monkeypatch.setattr(native_compact, 'evaluate', forbidden)
    assert hasattr(plan, 'metric_factor'), 'Reusable auxiliary metric factor is not implemented.'
    with pytest.raises(ValueError, match='lindep'):
        plan.metric_factor(p, a, lindep=lindep)
    with pytest.raises(ValueError, match='lindep'):
        plan.factors(p, a, lindep=lindep)


def test_nonfinite_metric_or_auxiliary_parameters_are_rejected(small_plan, monkeypatch):
    from gradscf.integrals.backends.native import density_fitting as native_compact
    plan, p, a = small_plan
    assert hasattr(plan, 'metric_factor'), 'Reusable auxiliary metric factor is not implemented.'
    bad = replace(a, exponents=(a.exponents[0].at[0].set(jnp.nan), *a.exponents[1:]))
    with pytest.raises(ValueError, match='finite'):
        plan.metric_factor(p, bad)
    monkeypatch.setattr(native_compact, 'evaluate', lambda *args, **kwargs: jnp.full(args[3], jnp.nan))
    with pytest.raises(ValueError, match='finite'):
        plan.metric_factor(p, a)


def test_whitening_overflow_is_rejected_instead_of_returning_infinity():
    from gradscf.integrals.molecular.density_fitting import AuxiliaryMetricFactor
    metric = AuxiliaryMetricFactor('synthetic', 1e-12, 'cholesky', [[1e-150]])
    with pytest.raises(ValueError, match='finite'):
        metric.whiten(np.array([[1e300]]))
