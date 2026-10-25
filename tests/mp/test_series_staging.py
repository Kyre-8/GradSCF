"""The pure MP series entry reuses a compiled graph without freezing physics."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import mp
from gradscf.mp import series


def model(unrestricted=False, dtype=np.float64):
    """Small stationary reference; no SCF, integral backend or external oracle."""
    n = 4 if unrestricted else 3
    rng = np.random.default_rng(2318)
    factors = rng.normal(size=(3, n, n))*.035
    factors = (factors+factors.transpose(0, 2, 1))*.5
    g = np.einsum("Lpq,Lrs->pqrs", factors, factors).astype(dtype)
    f = np.diag(np.linspace(-1., 1., n)).astype(dtype)
    if unrestricted:
        na, nb = 2, 1
        gaa, gab, gbb = g*.9, g, g*1.1
        va = (np.einsum("pqii->pq", gaa[:, :, :na, :na])
              - np.einsum("piiq->pq", gaa[:, :na, :na, :])
              + np.einsum("pqii->pq", gab[:, :, :nb, :nb]))
        vb = (np.einsum("pqii->pq", gbb[:, :, :nb, :nb])
              - np.einsum("piiq->pq", gbb[:, :nb, :nb, :])
              + np.einsum("iipq->pq", gab[:na, :na]))
        return tuple(map(jnp.asarray, (f-va, f-vb))), tuple(map(jnp.asarray, (gaa, gab, gbb))), (na, nb)
    mean_field = 2*np.einsum("pqii->pq", g[:, :, :1, :1])-np.einsum("piiq->pq", g[:, :1, :1, :])
    return jnp.asarray(f-mean_field), jnp.asarray(g), 1


@pytest.fixture(autouse=True)
def clean_private_cache():
    # Python builder spies must observe tracing, not an executable from a
    # previous test. Do not evict unrelated JAX executables.
    kernel = getattr(series, "_run_series", None)
    if kernel is not None:
        kernel.clear_cache()
    yield
    if kernel is not None:
        kernel.clear_cache()


def test_public_calls_reuse_one_graph_and_live_integrals(monkeypatch):
    h, g, no = model()
    build = series.make_series_space
    calls = []

    def counted(*args, **kwargs):
        calls.append(None)
        return build(*args, **kwargs)

    monkeypatch.setattr(series, "make_series_space", counted)
    cfg = mp.MPConfig(order=4, algorithm="series")
    mean_field = jnp.diag(jnp.linspace(-1., 1., h.shape[0]))-h
    original = mp.run_mp(h, g, nocc=no, config=cfg)
    changed = mp.run_mp(h-.2*mean_field, 1.2*g, nocc=no,
                        nuclear_repulsion=.3, config=replace(cfg, denominator_tol=2e-10))
    repeated = mp.run_mp(h, g, nocc=no, config=cfg)
    assert bool(original.valid & changed.valid & repeated.valid)
    np.testing.assert_allclose(changed.corrections,
                               original.corrections*1.2**np.arange(2, 5), atol=2e-13)
    np.testing.assert_array_equal(original.corrections, repeated.corrections)
    assert not np.isclose(original.correlation_energy, changed.correlation_energy,
                          rtol=1e-5, atol=0.)
    assert len(calls) == 1, "The public series entry rebuilt an unchanged topology"
    assert original.t2 is not None and original.wavefunction_coefficients is None


@pytest.mark.parametrize("field", ["denominator_tol", "canonical_tol"])
def test_tolerance_preserves_strong_and_weak_float32_boundary(field, monkeypatch):
    h = jnp.diag(jnp.array([0., .1], dtype=jnp.float32))
    if field == "canonical_tol":
        h = h.at[0, 1].set(.1).at[1, 0].set(.1)
    g = jnp.zeros((2,)*4, jnp.float32)
    tolerance = .1 if field == "denominator_tol" else .100000003
    build = series.make_series_space
    calls = []

    def counted(*args, **kwargs):
        calls.append(None)
        return build(*args, **kwargs)

    monkeypatch.setattr(series, "make_series_space", counted)
    outcomes = []
    for value in (tolerance, np.float64(tolerance), jnp.asarray(tolerance, jnp.float64)):
        cfg = mp.MPConfig(order=2, algorithm="series", with_t2=False,
                          **{field: value})
        result = mp.run_mp(h, g, nocc=1, config=cfg)
        outcomes.append(bool(result.valid))
        assert getattr(cfg, field) is value
    assert outcomes == [False, True, True]
    assert len(calls) == 2, "Strong float64 tolerance inputs should reuse their graph"


def test_jax_scalar_configuration_controls_are_not_mutated():
    h, g, no = model()
    tol = jnp.asarray(1e-10)
    canonical = jnp.asarray(1e-8)
    keep_t2, keep_waves = jnp.asarray(True), jnp.asarray(False)
    cfg = mp.MPConfig(order=4, algorithm="tensor", denominator_tol=tol,
                      canonical_tol=canonical, with_t2=keep_t2,
                      with_coefficients=keep_waves)
    h_before, g_before = np.asarray(h).copy(), np.asarray(g).copy()
    result = mp.run_mp(h, g, nocc=no, config=cfg)
    assert result.valid and result.t2 is not None
    assert result.wavefunction_coefficients is None
    assert cfg.denominator_tol is tol and cfg.canonical_tol is canonical
    assert cfg.with_t2 is keep_t2 and cfg.with_coefficients is keep_waves
    np.testing.assert_array_equal(h, h_before)
    np.testing.assert_array_equal(g, g_before)
    result = mp.run_mp(h, g, nocc=no,
        config=replace(cfg, with_t2=False, with_coefficients=True))
    assert result.valid and result.t2 is None
    assert result.wavefunction_coefficients is not None


@pytest.mark.parametrize("occupation", [1., True, np.int64(1)])
def test_restricted_occupation_keeps_strict_integer_validation(occupation):
    h, g, _ = model()
    cfg = mp.MPConfig(algorithm="series")
    assert mp.run_mp(h, g, nocc=1, config=cfg).valid
    with pytest.raises(ValueError, match="nocc"):
        mp.run_mp(h, g, nocc=occupation, config=cfg)


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("algorithm", ["series", "tensor"])
def test_arraylike_and_frozen_lists_keep_spin_layout(unrestricted, algorithm):
    h, g, no = model(unrestricted)
    frozen = ([0], [3]) if unrestricted else [2]
    to_list = lambda tree: jax.tree.map(lambda x: x.tolist(), tree)
    cfg = mp.MPConfig(order=4, algorithm=algorithm)
    expected = mp.run_mp(h, g, nocc=no, frozen=frozen, config=cfg)
    occupations = tuple(map(np.int64, no)) if unrestricted else no
    actual = mp.run_mp(to_list(h), to_list(g), nocc=occupations, frozen=frozen, config=cfg)
    assert expected.valid and actual.valid
    np.testing.assert_allclose(actual.corrections, expected.corrections, atol=2e-13)
    assert actual.t2 is not None and actual.wavefunction_coefficients is None


@pytest.mark.parametrize("algorithm", ["series", "tensor"])
def test_nested_jit_integral_gap_jvp_vjp_and_hvp(algorithm):
    h, g, no = model()
    fock = jnp.diag(jnp.linspace(-1., 1., h.shape[0]))
    mean_field = fock-h
    cfg = mp.MPConfig(order=4, algorithm=algorithm, with_t2=False)
    baseline = mp.run_mp(h, g, nocc=no, config=cfg)

    def energy(x):
        hx = (1+.1*(x-1))*fock-x*mean_field
        return mp.run_mp(hx, x*g, nocc=no, config=cfg).correlation_energy

    def bundle(x):
        value, jvp = jax.jvp(energy, (x,), (jnp.ones_like(x),))
        _, pullback = jax.vjp(energy, x)
        vjp = pullback(jnp.ones_like(value))[0]
        hvp = jax.jvp(jax.grad(energy), (x,), (.2*jnp.ones_like(x),))[1]
        return value, jvp, vjp, hvp

    value, jvp, vjp, hvp = jax.jit(jax.jit(bundle))(1.)
    orders = np.arange(2, 5)
    gradient = np.sum((orders-.1*(orders-1))*np.asarray(baseline.corrections))
    curvature = np.sum(orders*(orders-1)*.9**2*np.asarray(baseline.corrections))
    np.testing.assert_allclose(value, baseline.correlation_energy, atol=2e-13)
    np.testing.assert_allclose([jvp, vjp], gradient, atol=2e-12)
    np.testing.assert_allclose(hvp, .2*curvature, atol=2e-12)


def test_mixed_dtype_nuclear_repulsion_is_a_live_real_scalar():
    h = jnp.diag(jnp.array([-1., .5], dtype=jnp.float32))
    g = jnp.zeros((2,)*4, jnp.float32)
    cfg = mp.MPConfig(order=2, algorithm="series")
    energy = lambda enuc: mp.run_mp(h, g, nocc=1, nuclear_repulsion=enuc,
                                    config=cfg).total_energy
    enuc = jnp.asarray(.3, jnp.float64)
    value, derivative = jax.jit(jax.value_and_grad(energy))(enuc)
    assert value.dtype == jnp.float64 and derivative.dtype == enuc.dtype
    np.testing.assert_allclose(value, -2.+enuc, atol=1e-14)
    np.testing.assert_allclose(derivative, 1., atol=0.)
    for invalid in ([.3], .3+0.j):
        with pytest.raises(ValueError, match="real scalar"):
            energy(invalid)


def test_new_capacity_limit_still_precedes_operator_build(monkeypatch):
    h, g, no = model()
    cfg = mp.MPConfig(order=4, algorithm="series")
    assert mp.run_mp(h, g, nocc=no, config=cfg).valid

    def forbidden(*args, **kwargs):
        raise AssertionError("Operator construction preceded the space budget guard")

    monkeypatch.setattr(series.hamiltonian, "build_direct_hamiltonian", forbidden)
    with pytest.raises(ValueError, match="determinant"):
        mp.run_mp(h, g, nocc=no, config=replace(cfg, max_determinants=1))
