"""RSH direct-SCF wiring and effective exchange, without AO-ERI allocation."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.integrals.plan import IntegralPlan
from gradscf.scf import rsh, rks, uks
from gradscf.scf.inputs import build_rks_integral_inputs, build_uks_integral_inputs


@pytest.fixture
def direct_problem(monkeypatch):
    topology, parameters = integrals.prepare_basis("H 0 0 0; H 0 0 1.6", "sto-3g", cart=True)
    provider = NativeDirectBasis(integrals.make_plan(topology), parameters)
    # Algebraic, permutation-symmetric ERIs isolate the source/coefficients.
    full = jnp.asarray([[.8, .13], [.13, .5]])
    long_range = jnp.asarray([[.3, -.04], [-.04, .2]])
    eri = jnp.einsum("pq,rs->pqrs", full, full)
    eri_lr = jnp.einsum("pq,rs->pqrs", long_range, long_range)
    calls = []

    def get_jk(self, density, *, omega=0., with_j=True, with_k=True,
               screening_threshold=0., workers=1):
        calls.append((omega, with_j, with_k, screening_threshold, workers))
        source = eri if omega == 0. else eri_lr
        return (jnp.einsum("pqrs,...rs->...pq", source, density) if with_j else jnp.zeros_like(density),
                jnp.einsum("prqs,...rs->...pq", source, density) if with_k else jnp.zeros_like(density))

    monkeypatch.setattr(NativeDirectBasis, "get_jk", get_jk)
    return provider, eri, eri_lr, calls


def _oracle(eri, eri_lr, density, coefficients, *, with_j=True, with_k=True):
    _, long_range, short_range = coefficients
    exchange = short_range*eri + (long_range-short_range)*eri_lr
    return (jnp.einsum("pqrs,...rs->...pq", eri, density) if with_j else jnp.zeros_like(density),
            jnp.einsum("prqs,...rs->...pq", exchange, density) if with_k else jnp.zeros_like(density))


def test_rsh_direct_validation_and_metadata_require_no_eri(direct_problem):
    provider, _, _, _ = direct_problem
    preset = rsh.validate_rsh_inputs("lc-wpbe", None, None, None,
                                    jk_backend="direct", direct_basis=provider)
    assert preset.to_range_separated_hybrid_coefficients() == (.4, 1., 0.)
    source = SimpleNamespace(rep_tensor=jnp.zeros((0, 0, 0, 0)), eri_lr=None,
        direct_basis=provider, range_separated_coefficients=(.4, 1., 0.),
        xc_spec="lc-wpbe", dispersion_energy=None)
    assert rsh.response_rsh_coefficients(source) == (.4, 1., 0.)


def test_rsh_full_validation_still_requires_nonempty_matching_full_eri():
    eri = jnp.zeros((2, 2, 2, 2))
    assert rsh.validate_rsh_inputs("lc-wpbe", eri, eri, None) is not None
    for full, long_range in ((None, eri), (eri, None),
                              (jnp.zeros((0, 0, 0, 0)), jnp.zeros((0, 0, 0, 0))),
                              (jnp.zeros((3, 3)), jnp.zeros((3, 3)))):
        with pytest.raises(ValueError, match="(?i)eri|four.index"):
            rsh.validate_rsh_inputs("lc-wpbe", full, long_range, None)


@pytest.mark.parametrize("unrestricted", [False, True], ids=["rks", "uks"])
def test_rsh_native_direct_assembly_never_evaluates_eri(monkeypatch, unrestricted):
    original = IntegralPlan.evaluate

    def without_eri(self, operator, *args, **kwargs):
        if operator == "eri":
            raise AssertionError("Direct RSH assembly must not evaluate full or LR AO ERIs")
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, "evaluate", without_eri)
    config = (uks.UKSConfig if unrestricted else rks.RKSConfig)(
        xc_spec="lc-wpbe", jk_backend="direct", direct_scf_tol=0.)
    build = build_uks_integral_inputs if unrestricted else build_rks_integral_inputs
    controls = {} if unrestricted else {"include_dipole_integrals": False}
    inputs = build(atom="H 0 0 0; H 0 0 1.6", basis="sto-3g", unit="Bohr", cart=True,
        charge=int(unrestricted), spin=int(unrestricted), config=config,
        xc_spec="lc-wpbe", integral_backend="native", grids_level=0, **controls)
    assert type(inputs.direct_basis) is NativeDirectBasis
    assert inputs.eri is None or inputs.eri.size == 0
    assert inputs.eri_lr is None or inputs.eri_lr.size == 0
    if not unrestricted:
        assert inputs.eri_pair_matrix is None


@pytest.mark.parametrize("coefficients", [(.4, 1., 0.), (.3, .8, .6)], ids=["lc-wpbe", "mixed-sr-lr"])
@pytest.mark.parametrize("with_j", [True, False], ids=["jk", "k-only"])
def test_direct_rsh_effective_exchange_preserves_batched_density_ad(direct_problem, coefficients, with_j):
    provider, eri, eri_lr, calls = direct_problem
    density = jnp.linspace(-.3, .6, 12).reshape(3, 2, 2)
    controls = dict(with_j=with_j, screening_threshold=2e-5, workers=3)
    actual = lambda value: rsh._direct_rsh_jk(provider, value, coefficients, **controls)
    oracle = lambda value: _oracle(eri, eri_lr, value, coefficients, with_j=with_j)
    observed = actual(density)
    expected = oracle(density)
    for value, reference in zip(observed, expected, strict=True):
        np.testing.assert_allclose(value, reference, atol=2e-14, rtol=2e-14)
    omega, long_range, short_range = coefficients
    expected_calls = ([] if not (with_j or short_range) else [(0., with_j, short_range != 0., 2e-5, 3)])
    if long_range != short_range:
        expected_calls.append((omega, False, True, 2e-5, 3))
    assert calls == expected_calls
    direction = jnp.cos(density)*.07
    observed_jvp = jax.jvp(actual, (density,), (direction,))[1]
    expected_jvp = jax.jvp(oracle, (density,), (direction,))[1]
    cotangent = (jnp.sin(density), jnp.cos(density))
    observed_vjp = jax.vjp(actual, density)[1](cotangent)[0]
    expected_vjp = jax.vjp(oracle, density)[1](cotangent)[0]
    for value, reference in zip(jax.tree.leaves(observed_jvp), jax.tree.leaves(expected_jvp), strict=True):
        np.testing.assert_allclose(value, reference, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(observed_vjp, expected_vjp, atol=3e-14, rtol=3e-14)


@pytest.mark.parametrize("with_j,with_k", [(True, True), (False, False)], ids=["active", "zero-ops"])
def test_rsh_unknown_direct_provider_is_rejected_before_call(with_j, with_k):
    class LegacyProvider:
        def get_jk(self, density):
            raise AssertionError("Unknown RSH providers must not silently use full-range J/K")

    with pytest.raises((TypeError, NotImplementedError, ValueError), match="(?i)native|canonical|provider"):
        rsh._direct_rsh_jk(LegacyProvider(), jnp.eye(2), (.4, 1., 0.), with_j=with_j, with_k=with_k)


def test_rks_direct_rsh_incremental_builder_accumulates_effective_exchange(direct_problem):
    provider, eri, eri_lr, _ = direct_problem
    config = rks.RKSConfig(xc_spec="lc-wpbe", jk_backend="direct", direct_scf_workers=3)
    builder = rks._make_jk_builder(None, config, direct_basis=provider)
    previous = jnp.asarray([[.7, .1], [.1, .2]])
    density = jnp.asarray([[.6, .17], [.17, .3]])
    old_j, old_k = builder(previous)
    observed = builder(density, density_last=previous, j_last=old_j, k_last=old_k)
    expected = _oracle(eri, eri_lr, density, (.4, 1., 0.))
    for value, reference in zip(observed, expected, strict=True):
        np.testing.assert_allclose(value, reference, atol=3e-14, rtol=3e-14)


def test_uks_direct_rsh_fock_energy_matches_existing_full_lr_source(direct_problem, monkeypatch):
    provider, eri, eri_lr, calls = direct_problem
    da = jnp.asarray([[.7, .1], [.1, .2]])
    db = jnp.asarray([[.3, -.04], [-.04, .15]])
    hcore = jnp.asarray([[-1.7, .08], [.08, .4]])

    def zero_local_xc(**kwargs):
        points = kwargs["weights"].shape[0]
        scalar = jnp.zeros(points)
        gradient = jnp.zeros((points, 3))
        return jnp.asarray(0.), scalar, scalar, gradient, gradient, scalar, scalar

    # This algebraic test isolates the JK source; the same local XC value and
    # potential are used on both sides of the existing raw SCF energy/Fock API.
    monkeypatch.setattr(uks, "_unrestricted_xc_energy_and_potential_on_grid", zero_local_xc)

    def evaluate(density_a, density_b, direct):
        config = uks.UKSConfig(xc_spec="lc-wpbe", jk_backend="direct" if direct else "full")
        return uks._raw_fock_and_energy_for_state(
            density_a=density_a, density_b=density_b, ao=jnp.zeros((1, 2)),
            ao_deriv1=jnp.zeros((4, 1, 2)), weights=jnp.ones(1), h=hcore,
            eri=jnp.zeros((0, 0, 0, 0)) if direct else eri,
            eri_lr=None if direct else eri_lr, direct_basis=provider if direct else None,
            df_factors=None, enuc=jnp.asarray(.2), alpha=jnp.asarray(0.),
            cfg=config, xc_kind="GGA")

    full = evaluate(da, db, False)
    j, _ = _oracle(eri, eri_lr, da+db, (.4, 1., 0.))
    _, ka = _oracle(eri, eri_lr, da, (.4, 1., 0.))
    _, kb = _oracle(eri, eri_lr, db, (.4, 1., 0.))
    expected_energy = (jnp.sum((da+db)*hcore) + .5*jnp.sum((da+db)*j)
                       - .5*(jnp.sum(da*ka)+jnp.sum(db*kb)) + .2)
    np.testing.assert_allclose(full[0], expected_energy, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(full[2], hcore+j-ka, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(full[3], hcore+j-kb, atol=3e-14, rtol=3e-14)
    observed = evaluate(da, db, True)
    for value, expected in zip(observed, full, strict=True):
        np.testing.assert_allclose(value, expected, atol=3e-14, rtol=3e-14)
    assert all(not (omega == .4 and with_j) for omega, with_j, *_ in calls)
    actual_grad = jax.grad(lambda a, b: evaluate(a, b, True)[0], argnums=(0, 1))(da, db)
    expected_grad = jax.grad(lambda a, b: evaluate(a, b, False)[0], argnums=(0, 1))(da, db)
    for value, expected in zip(actual_grad, expected_grad, strict=True):
        np.testing.assert_allclose(value, expected, atol=3e-14, rtol=3e-14)


def _rsh_public_reference(provider, unrestricted):
    from gradscf.scf.molecules import QuadratureGrid, RestrictedMolecule, UnrestrictedMolecule

    common = dict(ao=jnp.ones((1, 2)), grid=QuadratureGrid(jnp.ones(1)),
        dipole_integrals=jnp.zeros((3, 2, 2)), rep_tensor=jnp.zeros((0, 0, 0, 0)),
        mo_coeff=jnp.stack((jnp.eye(2), jnp.eye(2))),
        mo_occ=jnp.asarray([[1., 0.], [0. if unrestricted else 1., 0.]]),
        mo_energy=jnp.asarray([[-1., .5], [-.8, .7]]),
        rdm1=jnp.asarray([[[1., 0.], [0., 0.]], [[0. if unrestricted else 1., 0.], [0., 0.]]]),
        h1e=jnp.eye(2), nuclear_repulsion=0., direct_basis=provider,
        range_separated_coefficients=(.4, 1., 0.), xc_spec="lc-wpbe")
    if unrestricted:
        return UnrestrictedMolecule(**common, nocc_alpha=1, nocc_beta=0)
    return RestrictedMolecule(**common, nocc=1)


@pytest.mark.parametrize("unrestricted", [False, True], ids=["restricted", "unrestricted"])
@pytest.mark.parametrize("driver_name", ["TDA", "TDDFT"])
def test_public_rsh_accepts_canonical_spin_adapter_without_scf(direct_problem, unrestricted, driver_name):
    from gradscf import tdscf
    from gradscf.tddft._semilocal_response import SemilocalResponseFunctional
    from gradscf.tddft._unrestricted_semilocal_response import UnrestrictedSemilocalResponseFunctional

    reference = _rsh_public_reference(direct_problem[0], unrestricted)
    functional = (UnrestrictedSemilocalResponseFunctional if unrestricted
                  else SemilocalResponseFunctional)("lc-wpbe")
    driver = getattr(tdscf, driver_name)(reference, xc_functional=functional, nstates=1)
    assert driver.molecule is reference
    kwargs = (driver._unrestricted_solver_kwargs(reference) if unrestricted
              else driver._restricted_solver_kwargs(reference))
    assert kwargs["xc_functional"] is functional


def test_public_rsh_keeps_unknown_local_and_parameter_override_guards(direct_problem):
    from gradscf import tdscf
    from gradscf.tddft._unrestricted_semilocal_response import UnrestrictedSemilocalResponseFunctional

    reference = _rsh_public_reference(direct_problem[0], True)
    for override in (SimpleNamespace(xc_spec="lc-wpbe"),
                     UnrestrictedSemilocalResponseFunctional("pbe"), "hyb_gga_xc_wb97x_d"):
        with pytest.raises(NotImplementedError, match="Range-separated"):
            _ = tdscf.TDDFT(reference, xc_functional=override).molecule
    with pytest.raises(NotImplementedError, match="parameter overrides"):
        _ = tdscf.TDDFT(reference, xc_functional=UnrestrictedSemilocalResponseFunctional("lc-wpbe"),
                        xc_params={}).molecule
