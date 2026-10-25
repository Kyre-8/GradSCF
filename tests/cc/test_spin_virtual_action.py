"""Spin CC virtual dressing: raw-index algebra and first/second AD parity."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.cc import _spin_equations, _spin_intermediates as imd
from gradscf.cc.uccsd import prepare_ucc_integrals


def _inputs(nocc, nvir, dtype=jnp.float64):
    rng = np.random.default_rng(6521)
    shapes = ((nocc, nvir), (nocc, nocc, nvir, nvir),
              (nocc, nvir, nvir, nvir), (nocc, nocc, nvir, nvir),
              (nvir,)*4)
    # No antisymmetry assumptions, including in independent raw ERI directions.
    return tuple(jnp.asarray(rng.normal(size=s)*.03, dtype) for s in shapes)


def _action(values, x, explicit=False):
    t1, t2, ovvv, oovv, vvvv = values
    eris = SimpleNamespace(ovvv=ovvv, oovv=oovv, vvvv=vvvv)
    if explicit:
        return jnp.einsum("abef,...ef->...ab", imd.cc_Wvvvv(t1, t2, eris), x)
    return imd.contract_Wvvvv(t1, t2, eris, x)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("nocc,nvir,leading", [(2, 5, (2, 3)), (6, 2, (6, 6)),
    (2, 5, ()), (0, 3, (2,)), (3, 0, (2,)), (2, 3, (0,))])
def test_action_matches_raw_dressed_tensor(dtype, nocc, nvir, leading):
    values = _inputs(nocc, nvir, dtype)
    rng = np.random.default_rng(6522)
    x = jnp.asarray(rng.normal(size=leading+(nvir, nvir)), dtype)
    actual = jax.jit(_action)(values, x)
    expected = _action(values, x, explicit=True)
    assert actual.dtype == dtype
    tolerance = 3e-7 if dtype == jnp.float32 else 2e-12
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("nocc,nvir", [(2, 5), (6, 2)])
def test_action_raw_jvp_vjp_hvp(nocc, nvir):
    values = _inputs(nocc, nvir)
    rng = np.random.default_rng(6523)
    x = jnp.asarray(rng.normal(size=(nocc, nocc, nvir, nvir))*.04)
    variables = values+(x,)
    direction = tuple(jnp.asarray(rng.normal(size=a.shape))*.02 for a in variables)
    probe = jnp.asarray(rng.normal(size=x.shape))

    def bundle(v, explicit):
        function = lambda w: _action(w[:-1], w[-1], explicit)
        value, jvp = jax.jvp(function, (v,), (direction,))
        _, pullback = jax.vjp(function, v)
        vjp, = pullback(probe)
        hvp = jax.jvp(jax.grad(lambda w: jnp.sum(function(w)**2)),
                      (v,), (direction,))[1]
        return value, jvp, vjp, hvp

    actual = jax.jit(lambda v: bundle(v, False))(variables)
    expected = jax.jit(lambda v: bundle(v, True))(variables)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-11)


def test_large_virtual_residual_never_builds_dressed_vvvv(monkeypatch):
    rng = np.random.default_rng(6524)
    n = 5
    h = (jnp.diag(jnp.arange(n, dtype=float)),)*2
    g = tuple(jnp.asarray(rng.normal(size=(n,)*4)*.01) for _ in range(3))
    ints, _, _ = prepare_ucc_integrals(h, g, nocc=(2, 1))
    no = 3
    nv = ints.fock.shape[0]-no
    t1, t2, *_ = _inputs(no, nv)
    expected = _spin_equations.residual(t1, t2, ints)

    def forbidden(*args, **kwargs):
        raise AssertionError("Spin residual materialized dressed Wvvvv")

    monkeypatch.setattr(imd, "cc_Wvvvv", forbidden)
    actual = jax.jit(lambda a, b: _spin_equations.residual(a, b, ints))(t1, t2)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)


def test_small_virtual_uses_cheaper_transient(monkeypatch):
    values = _inputs(8, 2)
    x = jnp.ones((8, 8, 2, 2))
    called = []
    original = imd.cc_Wvvvv

    def record(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(imd, "cc_Wvvvv", record)
    np.testing.assert_allclose(_action(values, x), _action(values, x, True),
                               atol=2e-12, rtol=2e-12)
    assert len(called) == 2


@pytest.mark.parametrize("nocc,nvir", [(2, 5), (6, 2)])
def test_explicit_tau_retains_its_independent_response(nocc, nvir):
    t1, t2, ovvv, oovv, vvvv = _inputs(nocc, nvir)
    eris = SimpleNamespace(ovvv=ovvv, oovv=oovv, vvvv=vvvv)
    tau = 1.3*t2
    singles_pair = imd.make_tau(jnp.zeros_like(t2), t1, t1)

    def value(a, explicit):
        if explicit:
            w = imd.cc_Wvvvv(t1, a-singles_pair, eris)
            return jnp.einsum("abef,ijef->ijab", w, t2)
        return imd.contract_Wvvvv(t1, t2, eris, t2, tau=a)

    actual = jax.jvp(lambda a: value(a, False), (tau,), (jnp.ones_like(tau),))
    expected = jax.jvp(lambda a: value(a, True), (tau,), (jnp.ones_like(tau),))
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("wide_field", [0, 1, 2, 3, 4, 5])
def test_action_mixed_dtype(wide_field):
    values = list(_inputs(2, 5, jnp.float32))
    values.append(jnp.ones((2, 2, 5, 5), dtype=jnp.float32))
    values[wide_field] = values[wide_field].astype(jnp.float64)
    actual = _action(values[:-1], values[-1])
    expected = _action(values[:-1], values[-1], True)
    assert actual.dtype == expected.dtype == jnp.float64
    np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=3e-7)
