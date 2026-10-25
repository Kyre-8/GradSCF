"""Compact bare UCC virtual action retains the original non-symmetric chart."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.cc.unrestricted_integrals import from_orbitals
from gradscf.cc import _spin_intermediates as imd


def _frame(dtype=jnp.float64):
    rng = np.random.default_rng(6552)
    cs = tuple(jnp.asarray(np.eye(5)+rng.normal(scale=.06, size=(5, 5)), dtype)
               for _ in range(2))
    fs = tuple(jnp.diag(jnp.asarray([-1.3, -.8, .3, .6, .9], dtype)) for _ in range(2))
    # No AO pair or within-pair symmetry: BA must retain the old AB pair transpose.
    g = jnp.asarray(rng.normal(scale=.02, size=(5,)*4), dtype)
    return cs, fs, g


def _snapshot(cs, fs, g, frozen, compact):
    options = {'vvvv': 'blocks'} if compact else {}
    return from_orbitals(cs, fs, (2, 1), jnp.asarray(-2., fs[0].dtype),
                         frozen=frozen, eri=g, max_memory=32., **options)


@pytest.mark.parametrize('frozen', [None, ([0, 3], [2, 4]),
    ([0, 1, 2, 3, 4], []), ([2, 3, 4], [1, 2, 3, 4])])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_compact_spin_virtual_storage_and_action_preserve_raw_chart(frozen, dtype):
    cs, fs, g = _frame(dtype)
    expected, occupied, virtual = _snapshot(cs, fs, g, frozen, False)
    actual, no, nv = _snapshot(cs, fs, g, frozen, True)
    assert (no, nv) == (occupied, virtual)
    assert not hasattr(actual.vvvv, 'shape'), 'Compact vvvv masquerades as a resident spin tensor'
    for name in actual._fields:
        if name != 'vvvv':
            assert getattr(actual, name).dtype == getattr(expected, name).dtype
            np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=2e-7)
    va, vb = virtual
    leaves = jax.tree.leaves(actual.vvvv)
    assert all(x.dtype == dtype for x in leaves)
    assert sum(x.size*x.dtype.itemsize for x in leaves) == (va**4+va**2*vb**2+vb**4)*jnp.dtype(dtype).itemsize
    rng = np.random.default_rng(6553)
    n, v = sum(no), sum(nv)
    t1 = jnp.asarray(rng.normal(scale=.01, size=(n, v)), dtype)
    t2 = jnp.asarray(rng.normal(scale=.01, size=(n, n, v, v)), dtype)
    x = jnp.asarray(rng.normal(scale=.02, size=(2, 3, v, v)), dtype)
    old = imd.contract_Wvvvv(t1, t2, expected, x)
    new = jax.jit(lambda data, rhs: imd.contract_Wvvvv(t1, t2, data, rhs))(actual, x)
    assert new.dtype == old.dtype == dtype
    np.testing.assert_allclose(new, old, atol=3e-7 if dtype == jnp.float32 else 2e-12,
                               rtol=3e-6 if dtype == jnp.float32 else 2e-11)


def test_compact_spin_virtual_joint_coefficient_ao_rhs_jvp_vjp_hvp():
    cs, fs, g = _frame()
    rng = np.random.default_rng(6554)
    x = jnp.asarray(rng.normal(scale=.03, size=(2, 7, 7)))
    probe = jnp.asarray(rng.normal(scale=.1, size=x.shape))
    t1 = jnp.asarray(rng.normal(scale=.01, size=(3, 7)))
    t2 = jnp.asarray(rng.normal(scale=.01, size=(3, 3, 7, 7)))

    def value(ca, cb, eri, rhs, compact):
        data, _, _ = _snapshot((ca, cb), fs, eri, None, compact)
        return jnp.vdot(imd.contract_Wvvvv(t1, t2, data, rhs), probe).real

    point = (*cs, g, x)
    direction = tuple(jnp.asarray(rng.normal(scale=.01, size=a.shape)) for a in point)
    actual = lambda *a: value(*a, True)
    expected = lambda *a: value(*a, False)
    np.testing.assert_allclose(actual(*point), expected(*point), atol=2e-12)
    operation = lambda fn: jax.jit(lambda *a: jax.jvp(fn, a, direction))(*point)
    np.testing.assert_allclose(operation(actual), operation(expected), atol=3e-12, rtol=3e-10)
    gradient = lambda fn: jax.grad(fn, argnums=(0, 1, 2, 3))
    for a, b in zip(jax.jit(gradient(actual))(*point), jax.jit(gradient(expected))(*point)):
        np.testing.assert_allclose(a, b, atol=3e-12, rtol=3e-10)
    operation = lambda fn: jax.jit(lambda *a: jax.jvp(gradient(fn), a, direction)[1])(*point)
    for a, b in zip(operation(actual), operation(expected)):
        np.testing.assert_allclose(a, b, atol=3e-11, rtol=3e-9)


def test_compact_spin_virtual_never_materializes_the_dense_w_fallback(monkeypatch):
    cs, fs, g = _frame()
    old, no, nv = _snapshot(cs, fs, g, None, False)
    new, _, _ = _snapshot(cs, fs, g, None, True)
    n, v = sum(no), sum(nv)
    t1, t2 = jnp.ones((n, v))*.01, jnp.ones((n, n, v, v))*.001
    rhs = jnp.ones((1000, v, v))*.01
    expected = imd.contract_Wvvvv(t1, t2, old, rhs)
    with pytest.raises(NotImplementedError, match='dense'):
        imd.cc_Wvvvv(t1, t2, new)

    def forbidden(*args, **kwargs):
        pytest.fail('Compact virtual action materialized a dense spin Wvvvv')

    monkeypatch.setattr(importlib.import_module('gradscf.cc._spin_intermediates'),
                        'cc_Wvvvv', forbidden)
    actual = imd.contract_Wvvvv(t1, t2, new, rhs)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)


def test_compact_budget_counts_per_request_mixed_coefficient_promotion(monkeypatch):
    module = importlib.import_module('gradscf.cc.unrestricted_integrals')
    nmo = 6
    ca = jnp.eye(nmo, dtype=jnp.float32)
    cb = jnp.eye(nmo, dtype=jnp.float64)*1.01
    fock = jnp.eye(nmo, dtype=jnp.float64)
    eri = jnp.zeros((nmo,)*4, dtype=jnp.float32)
    reservations = []

    class Preflight(Exception):
        pass

    def record(max_memory, *, resident_bytes, device):
        reservations.append(resident_bytes)
        raise Preflight

    monkeypatch.setattr(module, 'workspace_bytes', record)
    for alpha in (ca.astype(jnp.float64), ca):
        with pytest.raises(Preflight):
            module.from_orbitals((alpha, cb), (fock, fock), (2, 1), jnp.asarray(-4.),
                eri=eri, max_memory=32., vvvv='blocks')
    # Changing Ca64->Ca32 saves both its input and its selected-frame copy.
    # Its AA request alone now needs a float64 copy; AB/BB already return f64.
    expected = (nmo-2)**4*8-2*nmo*nmo*4
    assert reservations[1]-reservations[0] == expected
