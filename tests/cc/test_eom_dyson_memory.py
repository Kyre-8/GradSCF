"""Streamed Dyson contractions keep the upstream endpoint chart and AD."""
from types import SimpleNamespace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.cc.eom.amplitudes import EOMAmplitudeSpace
from gradscf.cc.eom.properties import dyson_orbitals
from gradscf.cc.ground import METHODS


def _inputs(sector, dtype=jnp.float64):
    rng = np.random.default_rng(6531)
    no, nv = 2, 3
    space = EOMAmplitudeSpace(no, nv, sector)
    shapes = ((no, nv), (no, no, nv, nv), (no, nv), (no, no, nv, nv),
              (space.size, 2), (space.size, 2))
    return tuple(jnp.asarray(rng.normal(size=s)*.04, dtype) for s in shapes)


def _evaluate(values, sector, **kwargs):
    t1, t2, l1, l2, right, left = values
    ground = SimpleNamespace(t1=t1, t2=t2, converged=True, method_id=METHODS.index("ccsd"))
    lambdas = SimpleNamespace(l1=l1, l2=l2, converged=True)
    eom = SimpleNamespace(sector=sector, energies=jnp.arange(2.)+.4,
        right_vectors=right, left_vectors=left, ground_valid=True,
        converged=jnp.ones(2, bool), response_valid=jnp.ones(2, bool), vector_response=True)
    return dyson_orbitals(ground, lambdas, eom, **kwargs)


def _pyscf(values, sector):
    from pyscf.cc import eom_rccsd, momgfccsd
    t1, t2, l1, l2, right, left = map(np.asarray, values)
    pack = eom_rccsd.amplitudes_to_vector_ip if sector == "ip" else eom_rccsd.amplitudes_to_vector_ea
    space = SimpleNamespace(amplitudes_to_vector=pack)
    bra = momgfccsd.build_bra_hole if sector == "ip" else momgfccsd.build_bra_part
    ket = momgfccsd.build_ket_hole if sector == "ip" else momgfccsd.build_ket_part
    b = np.stack([bra(None, space, t1, t2, l1, l2, p) for p in range(5)])
    k = np.stack([ket(None, space, t1, t2, p) for p in range(5)])
    if sector == "ea":
        k = -k
    dl, dr = b@right, k@left
    return dl, dr, np.sum(dl*dr, axis=0)


@pytest.mark.parametrize("sector", ["ip", "ea"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_dyson_streams_orbital_rows_with_the_same_pyscf_endpoints(sector, dtype, monkeypatch):
    values = _inputs(sector, dtype)
    expected = _pyscf(values, sector)
    module = importlib.import_module("gradscf.cc.eom.properties")

    def forbidden(*args, **kwargs):
        pytest.fail("Dyson constructed complete orbital-by-EOM endpoint matrices")

    monkeypatch.setattr(module, "_endpoints", forbidden)
    dimension = values[-1].shape[0]
    actual = jax.jit(lambda v: _evaluate(v, sector,
        max_elements=2*dimension, max_memory=1.))(values)
    assert np.all(actual.valid & actual.response_valid)
    tolerance = 2e-6 if dtype == jnp.float32 else 2e-13
    for a, b in zip(actual[:3], expected):
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)
        assert a.dtype == dtype


@pytest.mark.parametrize("sector", ["ip", "ea"])
def test_dyson_t_lambda_and_both_vector_response_matches_independent_fd(sector):
    values = _inputs(sector)
    rng = np.random.default_rng(6532)
    direction = tuple(jnp.asarray(rng.normal(size=a.shape))*.02 for a in values)
    dimension = values[-1].shape[0]

    def function(t):
        v = tuple(a+t*d for a, d in zip(values, direction))
        result = _evaluate(v, sector, max_elements=2*dimension, max_memory=1.)
        return jnp.sum(result.weights)+.3*jnp.sum(result.left**2)+.2*jnp.sum(result.right**2)

    def reference(t):
        dl, dr, weights = _pyscf(tuple(a+t*d for a, d in zip(values, direction)), sector)
        return np.sum(weights)+.3*np.sum(dl**2)+.2*np.sum(dr**2)

    value, gradient, hvp = jax.jit(lambda t: (function(t), jax.grad(function)(t),
        jax.jvp(jax.grad(function), (t,), (jnp.ones_like(t),))[1]))(0.)
    np.testing.assert_allclose(value, reference(0.), atol=2e-13)
    for step in (1e-3, 3e-4):
        minus, middle, plus = reference(-step), reference(0.), reference(step)
        np.testing.assert_allclose(gradient, (plus-minus)/(2*step), atol=2e-9, rtol=2e-6)
        np.testing.assert_allclose(hvp, (plus-2*middle+minus)/step**2, atol=2e-8, rtol=2e-5)


def test_dyson_minimum_checks_precede_endpoint_work(monkeypatch):
    values = _inputs("ip")
    module = importlib.import_module("gradscf.cc.eom.properties")

    def forbidden(*args, **kwargs):
        pytest.fail("Dyson endpoint work preceded budget rejection")

    monkeypatch.setattr(module, "_endpoints", forbidden)
    monkeypatch.setattr(module, "_endpoint", forbidden, raising=False)
    with pytest.raises(ValueError, match="memory|budget|resident|minimum"):
        _evaluate(values, "ip", max_memory=.00001)
    with pytest.raises(ValueError, match="max_elements"):
        _evaluate(values, "ip", max_elements=1)


def test_dyson_keeps_occupied_and_virtual_row_branches_static(monkeypatch):
    module = importlib.import_module("gradscf.cc.eom.properties")
    no, nv = 4, 16
    dimension = EOMAmplitudeSpace(no, nv, "ip").size
    values = tuple(jnp.full(shape, .01) for shape in (
        (no, nv), (no, no, nv, nv), (no, nv), (no, no, nv, nv),
        (dimension, 2), (dimension, 2)))
    original = module._endpoint
    kinds = set()
    def checked(t1, t2, l1, l2, sector, p, kind=None):
        assert kind in {"occupied", "virtual"}, "Dyson vmapped a dynamic row branch"
        kinds.add(kind)
        return original(t1, t2, l1, l2, sector, p, kind)
    monkeypatch.setattr(module, "_endpoint", checked)
    result = jax.jit(lambda x: _evaluate(x, "ip", max_memory=1.))(values)
    assert kinds == {"occupied", "virtual"}
    assert np.all(np.isfinite(result.weights))
    # A panel also fits this smaller budget; no full broadcast is needed.
    small = _evaluate(values, "ip", max_memory=.2)
    np.testing.assert_allclose(small.weights, result.weights, atol=3e-12)


def test_dyson_lowering_does_not_broadcast_full_amplitudes_over_rows():
    no, nv = 16, 128
    dimension = EOMAmplitudeSpace(no, nv, "ea").size
    shapes = ((no, nv), (no, no, nv, nv))*2+((dimension, 2),)*2
    abstract = tuple(jax.ShapeDtypeStruct(shape, jnp.float64) for shape in shapes)
    lowered = jax.jit(lambda x: _evaluate(x, "ea")).lower(abstract).as_text()
    for width in (16, 32):
        assert f"{width}x16x16x128x128xf64" not in lowered, (
            "Dyson replicated full T2/Lambda for each orbital row")


@pytest.mark.parametrize("owner", ["lambda", "eom"])
def test_dyson_charges_retained_response_arrays_before_endpoint_work(owner, monkeypatch):
    module = importlib.import_module("gradscf.cc.eom.properties")
    t1, t2, l1, l2, right, left = _inputs("ip")
    ground = SimpleNamespace(t1=t1, t2=t2, converged=True, method_id=METHODS.index("ccsd"))
    lambdas = SimpleNamespace(l1=l1, l2=l2, converged=True)
    eom = SimpleNamespace(sector="ip", energies=jnp.arange(2.)+.4,
        right_vectors=right, left_vectors=left, ground_valid=True,
        converged=jnp.ones(2, bool), response_valid=jnp.ones(2, bool), vector_response=True)
    # A separately retained packed Lambda vector or guard-root array must not
    # disappear from the ledger merely because the endpoint formula ignores it.
    setattr(lambdas if owner == "lambda" else eom,
            "adjoint" if owner == "lambda" else "guard_residual_norms", jnp.zeros(30_000))
    calls = []
    original = module._endpoint
    def endpoint(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(module, "_endpoint", endpoint)
    with pytest.raises(ValueError, match="memory|budget|resident|minimum"):
        dyson_orbitals(ground, lambdas, eom, max_memory=.2)
    assert not calls


def test_dyson_facade_charges_reference_and_known_scf_owners_once(monkeypatch):
    from dataclasses import dataclass
    from gradscf import cc

    @dataclass
    class Reference:
        h1: object
        eri: object

    h, g, s, density, ao_cache, params = (jnp.zeros((n,)) for n in (10, 20, 30, 40, 50, 60))
    ground_state = SimpleNamespace(t1=h, t2=g)
    lambda_state = SimpleNamespace(l1=h, l2=g)
    eom_state = SimpleNamespace(right_vectors=g)
    ground = SimpleNamespace(max_memory=4., reference=Reference(h, g),
        _integrals={"h": h, "g": g}, result=ground_state, lambda_result=lambda_state,
        solve_lambda=lambda: None,
        mf=SimpleNamespace(_scf_inputs=SimpleNamespace(overlap=s, hcore=h),
                           scf_result=SimpleNamespace(density_matrix=density, fock_matrix=h),
                           reference=SimpleNamespace(rep_tensor=ao_cache), params={"p": params}))
    obj = cc.EOMIP(ground)
    obj.result = eom_state
    monkeypatch.setattr(obj, "check_source", lambda: None)
    module = importlib.import_module("gradscf.cc.eom.properties")
    captured = {}
    def capture(*args, **kwargs):
        captured.update(kwargs)
    monkeypatch.setattr(module, "dyson_orbitals", capture)
    obj.dyson()
    # h/g are already charged with the three low-level state owners. Only
    # overlap, SCF density, separately cached AO data and parameters remain in
    # the facade reservation; none are visible in the selected CC source.
    assert captured["reserved_bytes"] == (30+40+50+60)*8
    assert captured["max_memory"] == 4.
