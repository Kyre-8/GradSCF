"""EA's two virtual consumers share the original dressed-intermediate chart."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.cc.eom import _charged, _intermediates as imd
from gradscf.cc.integrals import prepare_integrals


def _inputs(dtype=jnp.float64, *, zero_singles=False):
    no, nv = 2, 3
    rng = np.random.default_rng(6561)
    shapes = ((no+nv,)*4, (no, nv), (no, no, nv, nv), (nv,), (no, nv, nv))
    values = tuple(jnp.asarray(rng.normal(scale=.03, size=s), dtype) for s in shapes)
    if zero_singles:
        values = values[:1]+(jnp.zeros_like(values[1]),)+values[2:]
    return values


def _evaluate(values, *, explicit):
    g, t1, t2, r1, r2 = values
    ints = prepare_integrals(jnp.zeros(g.shape[:2], g.dtype), g, nocc=t1.shape[0])
    w = _charged.intermediates(t1, t2, ints, "ea")
    if explicit:
        # Preserve the original Wvvvv and complete Wvvvo as the dense oracle.
        state = {key: value for key, value in vars(w).items() if key != "virtual"}
        state["Wvvvv"] = imd.Wvvvv(t1, t2, ints)
        state["Wvvvo"] = imd.Wvvvo(t1, t2, ints, state["Wvvvv"])
        w = SimpleNamespace(**state)
    return _charged.ea_action(r1, r2, w)


def test_ea_virtual_state_does_not_retain_unused_ground_blocks():
    g, t1, t2, _, _ = _inputs()
    ints = prepare_integrals(jnp.zeros(g.shape[:2], g.dtype), g, nocc=t1.shape[0])
    state = _charged.intermediates(t1, t2, ints, "ea")
    held = {id(x) for x in jax.tree.leaves(state.virtual)}
    for array in (ints.ovov, ints.ovvv, ints.vvvv):
        assert id(array) in held
    for array in (ints.fock, ints.mo_energy, ints.oooo, ints.ovoo, ints.oovv, ints.ovvo):
        assert id(array) not in held, "Virtual action retained unused ground integral blocks"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_ea_virtual_consumers_do_not_cache_four_virtual_intermediate(dtype, monkeypatch):
    values = _inputs(dtype)
    expected = jax.jit(lambda x: _evaluate(x, explicit=True))(values)

    def forbidden(*args, **kwargs):
        pytest.fail("EA constructed or cached the full dressed Wvvvv")

    monkeypatch.setattr(imd, "Wvvvv", forbidden)
    actual = jax.jit(lambda x: _evaluate(x, explicit=False))(values)
    tolerance = 4e-6 if dtype == jnp.float32 else 3e-13
    for a, e in zip(actual, expected):
        assert a.dtype == dtype
        np.testing.assert_allclose(a, e, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("zero_singles", [False, True])
def test_ea_virtual_joint_raw_chart_retains_jvp_vjp_hvp(zero_singles):
    values = _inputs(zero_singles=zero_singles)
    direction = jax.tree.map(lambda x: .003*jnp.cos(jnp.arange(x.size)).reshape(x.shape), values)

    def bundle(point, explicit):
        def loss(x):
            one, two = _evaluate(x, explicit=explicit)
            probe = jnp.sin(jnp.arange(two.size)).reshape(two.shape)
            return jnp.sum(one**2)+.3*jnp.sum(two**2)+.1*jnp.sum(probe*two)
        return jax.jvp(jax.value_and_grad(loss), (point,), (direction,))

    actual = jax.jit(lambda x: bundle(x, False))(values)
    expected = jax.jit(lambda x: bundle(x, True))(values)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, atol=3e-11, rtol=3e-10)
