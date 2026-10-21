"""Real AO TDHF/TDDFT shares linear response work for its coupled X/Y pair."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.tddft.response import (
    _RestrictedResponseOperatorData,
    _restricted_a_action,
    _restricted_b_action,
    _restricted_tdhf_action,
)


@pytest.fixture
def model():
    rng = np.random.default_rng(740)
    coeff, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    factors = rng.normal(size=(4, 5, 5)) * 0.04
    factors += factors.transpose(0, 2, 1)
    return dict(
        orbo=jnp.asarray(coeff[:, :2]),
        orbv=jnp.asarray(coeff[:, 2:]),
        eri=jnp.asarray(np.einsum("Ppq,Prs->pqrs", factors, factors)),
        occupied=jnp.array([[-1.2, 0.07], [0.07, -0.6]]),
        virtual=jnp.array([[0.3, -0.06, 0.02], [-0.06, 0.7, 0.09], [0.02, 0.09, 1.1]]),
        x=jnp.asarray(rng.normal(size=(3, 2, 3))),
        y=jnp.asarray(rng.normal(size=(3, 2, 3))),
    )


def _data(model, scale, calls, *, noncanonical):
    eri = (1.0 + 0.1 * scale) * model["eri"]
    occupied = (1.0 + 0.05 * scale) * model["occupied"]
    virtual = (1.0 + 0.05 * scale) * model["virtual"]

    def ao_response(density):
        calls.append(density.shape)
        return (
            jnp.einsum("pqrs,nrs->npq", eri, density)
            - 0.5 * jnp.einsum("prqs,nrs->npq", eri, density)
        )

    return _RestrictedResponseOperatorData(
        delta_eps=jnp.diag(virtual)[None, :] - jnp.diag(occupied)[:, None],
        orbo=model["orbo"],
        orbv=model["orbv"],
        ao_response_action_fn=ao_response,
        hybrid_fraction=1.0,
        fock_occupied=occupied if noncanonical else None,
        fock_virtual=virtual if noncanonical else None,
    )


def _four_actions(data, x, y):
    return (
        _restricted_a_action(data, x) + _restricted_b_action(data, y),
        _restricted_b_action(data, x) + _restricted_a_action(data, y),
    )


def _with_xc(data, model, params, calls):
    vector = model["x"][0].reshape(-1)
    kernel = ((0.2 + jnp.sin(params[0])) * jnp.outer(vector, vector)
              + params[1] * jnp.eye(vector.size))

    def xc_response(values):
        calls.append(values.shape)
        return (values.reshape(-1, vector.size) @ kernel).reshape(values.shape)

    return replace(data, xc_response_action_fn=xc_response, linear_xc_response=True)


@pytest.mark.parametrize("noncanonical", [False, True])
def test_tdhf_reuses_one_ao_action_for_nonsymmetric_xy(model, noncanonical):
    calls = []
    data = _data(model, 0.2, calls, noncanonical=noncanonical)
    x, y = model["x"], model["y"]
    expected = _four_actions(data, x, y)
    assert len(calls) == 4
    calls.clear()
    actual = _restricted_tdhf_action(data, x, y)
    for observed, wanted in zip(actual, expected):
        np.testing.assert_allclose(observed, wanted, atol=2e-12, rtol=2e-12)
    assert calls == [(3, 5, 5)]


def test_tdhf_fusion_keeps_parameter_and_xy_jvp_vjp(model):
    def action(scale, x, y, *, oracle):
        data = _data(model, scale, [], noncanonical=True)
        return (_four_actions if oracle else _restricted_tdhf_action)(data, x, y)

    point = (jnp.asarray(0.2), model["x"], model["y"])
    tangent = (jnp.asarray(0.17), 0.1 * model["y"], -0.2 * model["x"])
    cotangent = (0.3 * model["x"], -0.4 * model["y"])
    results = []
    for oracle in (True, False):
        function = lambda scale, x, y: action(scale, x, y, oracle=oracle)
        directional = jax.jit(lambda p, t: jax.jvp(function, p, t)[1])(point, tangent)
        reverse = jax.jit(lambda p: jax.vjp(function, *p)[1](cotangent))(point)
        results.append((directional, reverse))
    for observed, wanted in zip(jax.tree.leaves(results[1]), jax.tree.leaves(results[0])):
        np.testing.assert_allclose(observed, wanted, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("noncanonical", [False, True])
def test_tddft_reuses_one_ao_and_xc_action(model, noncanonical):
    calls, xc_calls = [], []
    data = _with_xc(
        _data(model, 0.2, calls, noncanonical=noncanonical),
        model, jnp.array([0.3, -0.1]), xc_calls,
    )
    x, y = model["x"], model["y"]
    expected = _four_actions(data, x, y)
    assert len(calls) == len(xc_calls) == 4
    calls.clear()
    xc_calls.clear()
    actual = _restricted_tdhf_action(data, x, y)
    for observed, wanted in zip(actual, expected):
        np.testing.assert_allclose(observed, wanted, atol=2e-12, rtol=2e-12)
    assert calls == [(3, 5, 5)]
    assert xc_calls == [(3, 2, 3)]


def test_tddft_fusion_keeps_xc_parameter_and_xy_jvp_vjp(model):
    def action(scale, xc_params, x, y, *, oracle):
        data = _with_xc(
            _data(model, scale, [], noncanonical=True), model, xc_params, [],
        )
        return (_four_actions if oracle else _restricted_tdhf_action)(data, x, y)

    point = (jnp.asarray(0.2), jnp.array([0.3, -0.1]), model["x"], model["y"])
    tangent = (jnp.asarray(0.17), jnp.array([0.07, 0.12]),
               0.1 * model["y"], -0.2 * model["x"])
    cotangent = (0.3 * model["x"], -0.4 * model["y"])
    results = []
    for oracle in (True, False):
        function = lambda scale, params, x, y: action(scale, params, x, y, oracle=oracle)
        value, directional = jax.jit(lambda p, t: jax.jvp(function, p, t))(point, tangent)
        reverse = jax.jit(lambda p: jax.vjp(function, *p)[1](cotangent))(point)
        results.append((value, directional, reverse))
    assert np.linalg.norm(np.asarray(results[0][2][1])) > 1e-6
    for observed, wanted in zip(jax.tree.leaves(results[1]), jax.tree.leaves(results[0])):
        np.testing.assert_allclose(observed, wanted, atol=2e-11, rtol=2e-11)


def test_tddft_keeps_separate_opaque_clipped_xc_actions(model):
    calls, xc_calls = [], []
    data = _data(model, 0.2, calls, noncanonical=True)

    def clipped_response(values):
        xc_calls.append(values.shape)
        return jnp.clip(1.1 * values, -.5, .5)

    data = replace(data, xc_response_action_fn=clipped_response)
    x, y = jnp.full_like(model["x"], .4), jnp.full_like(model["y"], .35)
    expected = _four_actions(data, x, y)
    assert len(calls) == len(xc_calls) == 4
    calls.clear()
    xc_calls.clear()
    actual = _restricted_tdhf_action(data, x, y)
    for observed, wanted in zip(actual, expected):
        np.testing.assert_allclose(observed, wanted, atol=2e-12, rtol=2e-12)
    assert calls == [(3, 5, 5)]
    assert xc_calls == [(3, 2, 3), (3, 2, 3)]


def test_tddft_opaque_clipped_xc_keeps_parameter_and_xy_jvp_vjp(model):
    def action(scale, xc_scale, x, y, *, oracle):
        data = replace(
            _data(model, scale, [], noncanonical=True),
            xc_response_action_fn=lambda values: jnp.clip(xc_scale * values, -.5, .5),
        )
        return (_four_actions if oracle else _restricted_tdhf_action)(data, x, y)

    point = (jnp.asarray(.2), jnp.asarray(1.1),
             jnp.full_like(model["x"], .4), jnp.full_like(model["y"], .35))
    tangent = (jnp.asarray(.17), jnp.asarray(.07),
               .1 * model["y"], -.2 * model["x"])
    cotangent = (.3 * model["x"], -.4 * model["y"])
    results = []
    for oracle in (True, False):
        function = lambda scale, params, x, y: action(scale, params, x, y, oracle=oracle)
        value, directional = jax.jit(lambda p, t: jax.jvp(function, p, t))(point, tangent)
        reverse = jax.jit(lambda p: jax.vjp(function, *p)[1](cotangent))(point)
        results.append((value, directional, reverse))
    for observed, wanted in zip(jax.tree.leaves(results[1]), jax.tree.leaves(results[0])):
        np.testing.assert_allclose(observed, wanted, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("extra", ["nonlocal_a", "nonlocal_b", "mo_action"])
def test_tdhf_keeps_general_response_fallback(model, extra):
    calls = []
    data = _with_xc(
        _data(model, 0.2, calls, noncanonical=True),
        model, jnp.array([0.3, -0.1]), [],
    )
    if extra == "mo_action":
        def mo_action(values, *, bottom_density, bottom_projection):
            return (0.17 if bottom_density else 0.13) * values
        data = replace(data, ao_mo_response_action_fn=mo_action)
    else:
        name = {"nonlocal_a": "nonlocal_xc_a_action_fn",
                "nonlocal_b": "nonlocal_xc_b_action_fn"}[extra]
        data = replace(data, **{name: lambda values: 0.11 * values})
    expected = _four_actions(data, model["x"], model["y"])
    calls.clear()
    actual = _restricted_tdhf_action(data, model["x"], model["y"])
    for observed, wanted in zip(actual, expected):
        np.testing.assert_allclose(observed, wanted, atol=2e-12, rtol=2e-12)
    assert len(calls) == (0 if extra == "mo_action" else 4)
