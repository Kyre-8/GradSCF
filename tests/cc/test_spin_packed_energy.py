"""Packed UCC energy preserves the unrestricted raw integral/amplitude chart."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.cc import _spin_equations
from gradscf.cc.spin_amplitudes import SpinAmplitudeSpace
from gradscf.cc.uccsd import SpinCCIntegrals


def _packed_energy():
    function = getattr(_spin_equations, 'packed_correlation_energy', None)
    assert callable(function), 'Missing packed UCC energy contraction'
    return function


def _energy_integrals(fock, oovv):
    # The energy callback must not consume the residual-only integral blocks.
    return SpinCCIntegrals(fock, None, None, oovv, None, None, None, 0.)


@pytest.mark.parametrize('occupied,virtual', [((2, 1), (2, 3)), ((0, 2), (3, 2)),
    ((1, 1), (0, 0)), ((0, 0), (2, 3))])
@pytest.mark.parametrize('method', ['ccsd', 'ccd'])
@pytest.mark.parametrize('dtypes', [(jnp.float64,)*3, (jnp.float32,)*3,
                                   (jnp.float64, jnp.float32, jnp.float32)])
def test_raw_chart_energy_jvp_vjp_hvp(occupied, virtual, method, dtypes):
    packed = _packed_energy()
    space = SpinAmplitudeSpace(occupied, virtual, method)
    no, nv = sum(occupied), sum(virtual)
    size = sum(indices.shape[1] for indices in space.indices)
    shapes = ((size,), (no+nv, no+nv), (no, no, nv, nv))
    rng = np.random.default_rng(13159)
    point = tuple(jnp.asarray(rng.normal(scale=.1, size=s), dtype)
                  for s, dtype in zip(shapes, dtypes))
    direction = tuple(jnp.asarray(rng.normal(scale=.2, size=s), dtype)
                      for s, dtype in zip(shapes, dtypes))

    def energy(x, f, g, direct):
        ints = _energy_integrals(f, g)
        return (packed(x, space, ints) if direct else
                _spin_equations.correlation_energy(*space.unpack(x), ints))

    def response(direct):
        function = lambda *args: energy(*args, direct=direct)
        value, jvp = jax.jvp(function, point, direction)
        _, pullback = jax.vjp(function, *point)
        vjp = pullback(jnp.ones_like(value))
        hvp = jax.jvp(jax.grad(function, argnums=(0, 1, 2)),
                      point, direction)[1]
        return value, jvp, vjp, hvp

    actual = jax.jit(lambda: response(True))()
    expected = jax.jit(lambda: response(False))()
    tolerance = 3e-6 if jnp.float32 in dtypes else 2e-12
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert a.dtype == e.dtype
        assert np.isfinite(a).all() and np.isfinite(e).all()
        np.testing.assert_allclose(a, e, atol=tolerance, rtol=tolerance)


def test_mixed_raw_eri_promotes_before_signed_sum_and_never_unpacks(monkeypatch):
    packed = _packed_energy()
    space = SpinAmplitudeSpace((1, 1), (1, 1))
    ns = space.indices[0].shape[1]
    vector = jnp.zeros(ns+1, jnp.float64).at[ns].set(2.)
    g = jnp.zeros((2, 2, 2, 2), jnp.float32)
    g = g.at[0, 1, 0, 1].set(2**24).at[1, 0, 0, 1].set(-1.)
    g = g.at[0, 1, 1, 0].set(2**24)
    ints = _energy_integrals(jnp.zeros((4, 4), jnp.float32), g)
    expected = _spin_equations.correlation_energy(*space.unpack(vector), ints)
    np.testing.assert_allclose(expected, .25, atol=0., rtol=0.)

    def forbidden(*args, **kwargs):
        pytest.fail('Packed energy expanded dense spin amplitudes')

    monkeypatch.setattr(SpinAmplitudeSpace, 'unpack', forbidden)
    actual = jax.jit(lambda x, f, g: packed(x, space, _energy_integrals(f, g)))(
        vector, ints.fock, g)
    assert actual.dtype == jnp.float64
    np.testing.assert_allclose(actual, expected, atol=0., rtol=0.)


@pytest.mark.parametrize('method', ['ccsd', 'ccd'])
@pytest.mark.parametrize('frozen', [None, ([0], [])])
def test_ground_and_lambda_share_energy_and_preserve_packed_dual(method, frozen, monkeypatch):
    packed = _packed_energy()
    from gradscf import cc
    from test_prepared_properties import _problem

    uccsd = importlib.import_module('gradscf.cc.uccsd')
    assert uccsd.packed_correlation_energy is packed
    h, g = _problem()
    ints, occupied, virtual = uccsd.prepare_ucc_integrals((h, h), (g, g, g),
                                                        nocc=(2, 1), frozen=frozen)
    space = SpinAmplitudeSpace(occupied, virtual, method)
    config = cc.CCConfig(method=method, residual_tol=1e-11, conv_tol=1e-12,
                         adjoint_tol=1e-12)

    def run():
        state = cc.run_ucc_from_integrals(ints, occupied, virtual, config=config)
        left = cc.solve_lambda_from_integrals(ints, state, config=config)
        return state, left

    old = lambda x, space, data: _spin_equations.correlation_energy(*space.unpack(x), data)
    with monkeypatch.context() as dense:
        dense.setattr(uccsd, 'packed_correlation_energy', old)
        dense.setattr(_spin_equations, 'packed_correlation_energy', old)
        expected = jax.jit(run)()
    calls = {'right': 0, 'left': 0}

    def right(*args):
        calls['right'] += 1
        return packed(*args)

    def left_energy(*args):
        calls['left'] += 1
        return packed(*args)

    monkeypatch.setattr(uccsd, 'packed_correlation_energy', right)
    monkeypatch.setattr(_spin_equations, 'packed_correlation_energy', left_energy)
    actual = jax.jit(lambda: run())()
    assert all(calls.values())
    assert actual[0].converged and actual[1].converged
    assert expected[0].converged and expected[1].converged
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.isfinite(a).all() and np.isfinite(e).all()
        np.testing.assert_allclose(a, e, atol=2e-11, rtol=2e-10)
    l1, dual2 = space.unpack_blocks(actual[1].adjoint)
    for a, e in zip(actual[1].l1, l1):
        np.testing.assert_allclose(a, e, atol=2e-13, rtol=0.)
    for a, e in zip(actual[1].l2, dual2):
        np.testing.assert_allclose(a, 4*e, atol=2e-13, rtol=0.)


@pytest.mark.parametrize('method', ['ccsd', 'ccd'])
def test_closed_shell_energy_and_response_match_restricted(method):
    _packed_energy()
    from gradscf import cc
    from test_prepared_properties import _problem

    h, g = _problem()
    direction = jnp.diag(jnp.array([.13, -.07, .05, -.03]))
    config = cc.CCConfig(method=method, residual_tol=1e-11, conv_tol=1e-12)

    def energy(x, unrestricted):
        moved = h+x*direction
        if unrestricted:
            result = cc.run_ucc((moved, moved), (g, g, g), nocc=(2, 2), config=config)
        else:
            result = cc.run_cc(moved, g, nocc=2, config=config)
        return jnp.where(result.converged, result.total_energy, jnp.nan)

    def bundle(x, unrestricted):
        function = lambda y: energy(y, unrestricted)
        return jax.jvp(jax.value_and_grad(function), (x,), (1.,))

    actual = jax.jit(lambda x: bundle(x, True))(.01)
    expected = jax.jit(lambda x: bundle(x, False))(.01)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-9)
