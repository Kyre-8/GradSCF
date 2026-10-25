"""Prepared restricted Lambda and local one-body partials, CPU/float64."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc
from gradscf.cc.amplitudes import AmplitudeSpace
from gradscf.cc.integrals import prepare_integrals
from gradscf.cc.rccsd import residual, correlation_energy


MODELS = ('ccsd', 'ccd', 'cc2', 'qcisd')


def _problem(dtype=jnp.float64):
    rng = np.random.default_rng(6411)
    factors = rng.normal(scale=.09, size=(3, 4, 4))
    factors = .5*(factors+factors.swapaxes(1, 2))
    g = jnp.asarray(np.einsum('Lpq,Lrs->pqrs', factors, factors), dtype=dtype)
    f = jnp.diag(jnp.asarray([-1.2, -.8, .4, .7], dtype=dtype))
    off = jnp.asarray(rng.normal(scale=.015, size=(4, 4)), dtype=dtype)
    f = f+.5*(off+off.T)
    h = f-2*jnp.einsum('pqii->pq', g[:, :, :2, :2])+jnp.einsum('piiq->pq', g[:, :2, :2, :])
    return h, g


def _config(method='ccsd', dtype=jnp.float64):
    tol = 2e-6 if dtype == jnp.float32 else 1e-11
    return cc.CCConfig(method=method, residual_tol=tol, conv_tol=tol*.1,
                       adjoint_tol=tol*.1)


def _left(ints, result, config):
    function = getattr(cc, 'solve_lambda_from_integrals', None)
    assert callable(function), 'Missing prepared restricted Lambda entry'
    return function(ints, result, config=config)


def _density(ints, result, left, *, frozen=None, config=None):
    module = importlib.import_module('gradscf.cc.properties')
    function = getattr(module, '_density_from_integrals', None)
    assert callable(function), 'Missing prepared restricted one-body partial'
    return function(ints, result, left, nmo=4, nocc=2, frozen=frozen, config=config)


def _raw_partial(h, g, result, left, *, frozen, config):
    """Independent old raw-Hamiltonian Lagrangian partial, with fixed states."""
    space = AmplitudeSpace(*result.t1.shape, config.method)

    def lagrangian(raw_h):
        ints = prepare_integrals(.5*(raw_h+raw_h.T), g, nocc=2, frozen=frozen)
        r = space.pack(*residual(result.t1, result.t2, ints, model=config.method))
        return (ints.reference_energy+correlation_energy(result.t1, result.t2, ints,
                model=config.method)+jnp.vdot(left.adjoint, r).real)

    return jax.grad(lagrangian)(h)


@pytest.mark.parametrize('method', MODELS)
@pytest.mark.parametrize('frozen', [None, [0, 3]])
def test_prepared_lambda_matches_raw_and_packed_dual(method, frozen):
    h, g = _problem()
    cfg = _config(method)
    ints = prepare_integrals(h, g, nocc=2, frozen=frozen)
    state = cc.run_cc_from_integrals(ints, config=cfg)
    expected = cc.solve_lambda(h, g, state, nocc=2, frozen=frozen, config=cfg)
    actual = _left(ints, state, cfg)
    assert state.converged and actual.converged and expected.converged
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, atol=2e-12, rtol=2e-11)
    space = AmplitudeSpace(ints.nocc, ints.nvir, method)
    dual1, dual2 = space.unpack(actual.adjoint)
    np.testing.assert_allclose(2*actual.l1, dual1, atol=2e-13)
    np.testing.assert_allclose(2*actual.l2-actual.l2.swapaxes(2, 3), dual2, atol=2e-13)
    x = space.pack(state.t1, state.t2)
    r = lambda t: space.pack(*residual(*space.unpack(t), ints, model=method))
    e = lambda t: correlation_energy(*space.unpack(t), ints, model=method)
    _, transpose = jax.vjp(r, x)
    norm = jnp.linalg.norm(transpose(actual.adjoint)[0]+jax.grad(e)(x))
    assert norm < 2e-11
    np.testing.assert_allclose(actual.residual_norm, norm, atol=2e-13)


@pytest.mark.parametrize('method', MODELS)
@pytest.mark.parametrize('frozen', [None, [0, 3]])
def test_prepared_density_matches_raw_partial_and_one_body_energy(method, frozen):
    h, g = _problem()
    cfg = _config(method)
    ints = prepare_integrals(h, g, nocc=2, frozen=frozen)
    state = cc.run_cc_from_integrals(ints, config=cfg)
    left = _left(ints, state, cfg)
    dm = _density(ints, state, left, frozen=frozen, config=cfg)
    expected = _raw_partial(h, g, state, left, frozen=frozen, config=cfg)
    np.testing.assert_allclose(dm, expected, atol=3e-12, rtol=2e-11)
    np.testing.assert_allclose(dm, cc.make_rdm1(h, g, state, nocc=2, frozen=frozen,
                                             config=cfg), atol=3e-12)
    np.testing.assert_allclose(dm, dm.T, atol=1e-13)
    np.testing.assert_allclose(jnp.trace(dm), 4., atol=2e-12)
    probe = jnp.asarray(np.random.default_rng(6412).normal(scale=.03, size=h.shape))
    probe = .5*(probe+probe.T)
    energy = lambda x: cc.run_cc(h+x*probe, g, nocc=2, frozen=frozen, config=cfg).total_energy
    np.testing.assert_allclose(jnp.sum(dm*probe), jax.grad(energy)(0.), atol=2e-10, rtol=2e-9)
    if frozen:
        np.testing.assert_allclose(dm[0], [2., 0., 0., 0.], atol=1e-13)
        np.testing.assert_allclose(dm[3], 0., atol=1e-13)


def test_prepared_properties_never_prepare_raw_integrals(monkeypatch):
    h, g = _problem()
    cfg = _config()
    ints = prepare_integrals(h, g, nocc=2, frozen=[0, 3])
    state = cc.run_cc_from_integrals(ints, config=cfg)

    def forbidden(*args, **kwargs):
        raise AssertionError('Prepared properties re-prepared raw Hamiltonian')

    for name in ('gradscf.cc.integrals', 'gradscf.cc.lambda_equations', 'gradscf.cc.properties'):
        monkeypatch.setattr(importlib.import_module(name), 'prepare_integrals', forbidden)
    left = _left(ints, state, cfg)
    dm = _density(ints, state, left, frozen=[0, 3], config=cfg)
    assert left.converged and np.isfinite(dm).all()


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_prepared_lambda_dtype_and_invalid_state(dtype):
    h, g = _problem(dtype)
    cfg = _config(dtype=dtype)
    ints = prepare_integrals(h, g, nocc=2)
    state = cc.run_cc_from_integrals(ints, config=cfg)
    left = _left(ints, state, cfg)
    assert state.converged and left.converged
    assert all(x.dtype == dtype for x in (left.l1, left.l2, left.adjoint))
    for bad in (state._replace(converged=jnp.asarray(False)),
                state._replace(method_id=jnp.asarray(1)),
                state._replace(t1=state.t1+.01)):
        invalid = _left(ints, bad, cfg)
        assert not invalid.converged and np.isnan(invalid.adjoint).all()
        assert np.isnan(_density(ints, bad, invalid, config=cfg)).all()


@pytest.mark.parametrize('frozen', [2, [2, 3], [0, 1, 2, 3]])
def test_prepared_empty_correlation_space_density(frozen):
    h, g = _problem()
    cfg = _config()
    ints = prepare_integrals(h, g, nocc=2, frozen=frozen)
    state = cc.run_cc_from_integrals(ints, config=cfg)
    left = _left(ints, state, cfg)
    dm = _density(ints, state, left, frozen=frozen, config=cfg)
    assert state.converged and left.converged
    np.testing.assert_allclose(dm, jnp.diag(jnp.array([2., 2., 0., 0.])), atol=1e-13)


def test_raw_density_mixed_precision_keeps_hamiltonian_partial_dtype():
    h, g = _problem()
    h = h.astype(jnp.float32)
    cfg = _config()
    state = cc.run_cc(h, g, nocc=2, config=cfg)
    left = cc.solve_lambda(h, g, state, nocc=2, config=cfg)
    actual = cc.make_rdm1(h, g, state, nocc=2, config=cfg)
    expected = _raw_partial(h, g, state, left, frozen=None, config=cfg)
    assert actual.dtype == expected.dtype == jnp.float32
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)


def test_raw_density_off_hermitian_chart_matches_original_partial_response():
    h, g = _problem()
    cfg = _config()
    off = jnp.asarray(np.random.default_rng(6414).normal(scale=.03, size=h.shape))
    h = h+off-off.T
    direction = jnp.asarray(np.random.default_rng(6415).normal(scale=.04, size=h.shape))
    probe = direction+.5*direction.T

    def value(x, raw):
        hs, gs = h+x*direction, g*(1+.13*x)
        state = cc.run_cc(hs, gs, nocc=2, frozen=[0, 3], config=cfg)
        left = cc.solve_lambda(hs, gs, state, nocc=2, frozen=[0, 3], config=cfg)
        dm = (_raw_partial(hs, gs, state, left, frozen=[0, 3], config=cfg) if raw else
              cc.make_rdm1(hs, gs, state, nocc=2, frozen=[0, 3], config=cfg))
        return jnp.sum(dm*probe)

    actual, expected = lambda x: value(x, False), lambda x: value(x, True)
    for operation in (lambda f: f(0.), lambda f: jax.jit(jax.grad(f))(0.),
                      lambda f: jax.jit(jax.grad(jax.grad(f)))(0.)):
        np.testing.assert_allclose(operation(actual), operation(expected), atol=2e-11, rtol=2e-9)


@pytest.mark.parametrize('frozen', [None, [0, 3]])
def test_prepared_density_joint_hg_jvp_vjp_hvp_includes_both_state_responses(frozen):
    h, g = _problem()
    cfg = _config()
    rng = np.random.default_rng(6413)
    dh = jnp.asarray(rng.normal(scale=.04, size=h.shape))
    dh = .5*(dh+dh.T)
    dg = .17*g
    probe = jnp.asarray(rng.normal(scale=.1, size=h.shape))

    def observe(hs, gs, prepared, freeze_states=False):
        ints = prepare_integrals(hs, gs, nocc=2, frozen=frozen)
        state = cc.run_cc_from_integrals(ints, config=cfg)
        left = _left(ints, state, cfg)
        if freeze_states:
            state, left = jax.tree.map(jax.lax.stop_gradient, (state, left))
        dm = (_density(ints, state, left, frozen=frozen, config=cfg) if prepared else
              _raw_partial(hs, gs, state, left, frozen=frozen, config=cfg))
        return jnp.sum(dm*probe)

    point, direction = (h, g), (dh, dg)
    actual = lambda *x: observe(*x, True)
    expected = lambda *x: observe(*x, False)
    np.testing.assert_allclose(actual(*point), expected(*point), atol=3e-12)
    jvp = jax.jit(lambda *x: jax.jvp(actual, x, direction)[1])(*point)
    raw_jvp = jax.jit(lambda *x: jax.jvp(expected, x, direction)[1])(*point)
    np.testing.assert_allclose(jvp, raw_jvp, atol=3e-11, rtol=3e-9)
    gradient = jax.jit(jax.grad(actual, argnums=(0, 1)))(*point)
    for a, e in zip(gradient, jax.jit(jax.grad(expected, argnums=(0, 1)))(*point)):
        np.testing.assert_allclose(a, e, atol=3e-11, rtol=3e-9)
    np.testing.assert_allclose(sum(jnp.vdot(a, b) for a, b in zip(gradient, direction)),
                               jvp, atol=2e-12)
    operation = lambda fn: jax.jit(lambda *x: jax.jvp(jax.grad(fn, argnums=(0, 1)),
                                                    x, direction)[1])(*point)
    hvp = operation(actual)
    for a, e in zip(hvp, operation(expected)):
        np.testing.assert_allclose(a, e, atol=3e-10, rtol=3e-8)
    step = 2e-4
    plus = tuple(x+step*d for x, d in zip(point, direction))
    minus = tuple(x-step*d for x, d in zip(point, direction))
    np.testing.assert_allclose(jvp, (actual(*plus)-actual(*minus))/(2*step), atol=2e-9, rtol=2e-6)
    grad = jax.jit(jax.grad(actual, argnums=(0, 1)))
    for a, p, m in zip(hvp, grad(*plus), grad(*minus)):
        np.testing.assert_allclose(a, (p-m)/(2*step), atol=2e-8, rtol=2e-5)
    fixed = lambda *x: observe(*x, True, freeze_states=True)
    assert abs(float(jvp-jax.jvp(fixed, point, direction)[1])) > 1e-7
