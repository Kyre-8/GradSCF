"""CCS owns no virtual-four input; other exported models retain its action.

Prepared local chart: Fock/occupied blocks and T are independent of vvvv.
"""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc
from gradscf.cc.integrals import prepare_integrals, from_orbitals
from gradscf.cc.rccsd import residual, correlation_energy
from test_prepared_properties import _problem
from test_scf_prepared import source


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_ccs_never_evaluates_discarded_doubles_or_virtual_action(dtype, monkeypatch):
    h, g = _problem(dtype)
    ints = prepare_integrals(h, g, nocc=2)
    t1 = jnp.arange(4., dtype=dtype).reshape(2, 2)*.01
    t2 = jnp.ones((2, 2, 2, 2), dtype=dtype)*.02
    expected = residual(t1, t2, ints, model='ccs')

    def forbidden(*args, **kwargs):
        pytest.fail('CCS evaluated the virtual action of its discarded doubles equation')

    monkeypatch.setattr(importlib.import_module('gradscf.cc._intermediates'),
                        'contract_Wvvvv', forbidden)
    actual = residual(t1, t2, ints._replace(vvvv=None), model='ccs')
    for a, b in zip(actual, expected):
        assert a.dtype == b.dtype == dtype
        np.testing.assert_allclose(a, b, atol=2e-6 if dtype == jnp.float32 else 2e-12)


def test_ccs_omitted_virtual_chart_preserves_joint_first_and_second_response(monkeypatch):
    h, g = _problem()
    t1 = jnp.arange(4.).reshape(2, 2)*.01
    t2 = jnp.zeros((2, 2, 2, 2))
    dh = jnp.array([[0., .01, 0., .02], [0., 0., .01, 0.],
                    [0., -.02, 0., .01], [.03, 0., 0., 0.]])
    probe = jnp.array([[.2, .7], [.1, -.3]])

    def value(x, omitted):
        ints = prepare_integrals(h+x*dh, g*(1+.13*x), nocc=2)
        if omitted:
            ints = ints._replace(oooo=None, vvvv=None)
        r1, _ = residual(t1*(1+x), t2, ints, model='ccs')
        return jnp.sum(r1*probe)+correlation_energy(t1*(1+x), t2, ints, model='ccs')

    def bundle(x, omitted):
        function = lambda y: value(y, omitted)
        v, tangent = jax.jvp(function, (x,), (1.,))
        gradient = jax.grad(function)(x)
        hessian = jax.jvp(jax.grad(function), (x,), (1.,))[1]
        return v, tangent, gradient, hessian

    expected = jax.jit(lambda x: bundle(x, False))(.01)
    monkeypatch.setattr(importlib.import_module('gradscf.cc._intermediates'),
        'contract_Wvvvv', lambda *a, **kw: pytest.fail('CCS differentiated an unused virtual action'))
    actual = jax.jit(lambda x: bundle(x, True))(.01)
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)


def test_ccs_snapshot_never_requests_unused_occupied_or_virtual_four_blocks(monkeypatch):
    module = importlib.import_module('gradscf.cc.integrals')
    h, g = _problem()
    expected = prepare_integrals(h, g, nocc=2)
    original = module.transform_eri_blocks
    requests = []

    def checked(spaces, blocks, **kwargs):
        assert (1, 1, 1, 1) not in blocks, 'Unused vvvv was requested'
        assert (0, 0, 0, 0) not in blocks, 'Unused oooo was requested'
        requests.extend(blocks)
        return original(spaces, blocks, **kwargs)

    monkeypatch.setattr(module, 'transform_eri_blocks', checked)
    actual = from_orbitals(jnp.eye(4), expected.fock, 2, expected.reference_energy,
                           eri=g, model='ccs', max_memory=2.)
    assert actual.oooo is actual.vvvv is None and len(requests) == 5
    assert (actual.nocc, actual.nvir) == (expected.nocc, expected.nvir)
    for name in actual._fields:
        if name not in {'oooo', 'vvvv'}:
            np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=2e-12)


def test_scf_ccs_ground_left_and_density_skip_virtual_preparation(source, monkeypatch):
    from gradscf.scf.reference import reference_from_source
    raw = reference_from_source(source)
    expected = cc.CCS(raw, conv_tol=1e-12, residual_tol=1e-10).run()
    density = expected.make_rdm1()
    module = importlib.import_module('gradscf.cc.integrals')
    original = module.transform_eri_blocks

    def checked(spaces, blocks, **kwargs):
        if (1, 1, 1, 1) in blocks or (0, 0, 0, 0) in blocks:
            pytest.fail('Selected CCS eagerly prepared an unused occupied/virtual block')
        return original(spaces, blocks, **kwargs)

    monkeypatch.setattr(module, 'transform_eri_blocks', checked)
    actual = cc.CCS(source, conv_tol=1e-12, residual_tol=1e-10).run()
    assert actual.converged and actual._integrals.oooo is actual._integrals.vvvv is None
    np.testing.assert_allclose(actual.e_tot, expected.e_tot, atol=3e-11)
    np.testing.assert_allclose(actual.make_rdm1(), density, atol=3e-9)


@pytest.mark.parametrize('model', ['ccs', 'cc2', 'qcisd'])
def test_model_virtual_consumer_census(model):
    h, g = _problem()
    ints = prepare_integrals(h, g, nocc=2)
    t1 = jnp.array([[.01, -.02], [.03, .04]])
    t2 = jnp.arange(16.).reshape(2, 2, 2, 2)*.001
    t2 = .5*(t2+t2.transpose(1, 0, 3, 2))

    def value(vvvv):
        data = ints._replace(vvvv=vvvv)
        r1, r2 = residual(t1, t2, data, model=model)
        return jnp.sum(r1)+jnp.sum(r2)+correlation_energy(t1, t2, data, model=model)

    derivative = jax.grad(value)(ints.vvvv)
    if model == 'ccs':
        np.testing.assert_array_equal(derivative, jnp.zeros_like(derivative))
    else:
        assert float(jnp.linalg.norm(derivative)) > 1e-12, 'An active virtual consumer was lost'
