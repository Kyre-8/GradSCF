"""Necessary AO2MO reference forwarding retains its raw C/h/G AD chart."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _point():
    rng = np.random.default_rng(2191)
    return tuple(jnp.asarray(rng.normal(size=shape))
                 for shape in ((4, 3), (4, 4), (4,)*4))


def test_reference_forwards_caller_and_one_body_owners(monkeypatch):
    module = importlib.import_module('gradscf.ao2mo.reference')
    c, h, g = _point()
    original = module.transform_eri_block
    observed = []

    def block(coefficients, **kwargs):
        observed.append(kwargs['reserved_bytes'])
        return original(coefficients, **kwargs)

    monkeypatch.setattr(module, 'transform_eri_block', block)
    actual = module.transform_integrals(h, c, eri=g, max_memory=1., reserved_bytes=4096)
    assert observed[0] >= 4096+h.nbytes+(4*3+3*3)*8
    np.testing.assert_allclose(actual[0], c.T@h@c, atol=2e-12)
    expected = jnp.einsum('pqrs,pP,qQ,rR,sS->PQRS', g, c, c, c, c)
    np.testing.assert_allclose(actual[1], expected, atol=2e-12)


@pytest.mark.parametrize('reserved', [-1, True, .5])
def test_invalid_reference_reservation_precedes_the_transform(reserved, monkeypatch):
    module = importlib.import_module('gradscf.ao2mo.reference')

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid reservation reached the numerical transform')

    monkeypatch.setattr(module, 'transform_eri_block', forbidden)
    c, h, g = _point()
    with pytest.raises(ValueError, match='reserved_bytes'):
        module.transform_integrals(h, c, eri=g, reserved_bytes=reserved)


def test_reference_joint_minimum_precedes_the_transform(monkeypatch):
    module = importlib.import_module('gradscf.ao2mo.reference')

    def forbidden(*args, **kwargs):
        pytest.fail('The reference one-body minimum was not checked first')

    monkeypatch.setattr(module, 'transform_eri_block', forbidden)
    c, h, g = _point()
    with pytest.raises(ValueError, match='memory|resident|budget'):
        module.transform_integrals(h, c, eri=g, max_memory=.001, reserved_bytes=750)


def test_reserved_reference_preserves_raw_chart_first_and_second_ad():
    module = importlib.import_module('gradscf.ao2mo.reference')
    point = _point()
    direction = jax.tree.map(lambda x: jnp.full_like(x, .007), point)

    def loss(values, prepared):
        c, h, g = values
        if prepared:
            hm, gm = module.transform_integrals(h, c, eri=g, max_memory=1., reserved_bytes=4096)
        else:
            hm = c.T@h@c
            gm = jnp.einsum('pqrs,pP,qQ,rR,sS->PQRS', g, c, c, c, c)
        return .03*jnp.sum(hm**2)+.001*jnp.sum(gm**2)

    def bundle(values, prepared):
        function = lambda x: loss(x, prepared)
        return jax.value_and_grad(function)(values), jax.jvp(jax.grad(function), (values,), (direction,))[1]

    a = jax.jit(lambda x: bundle(x, True))(point)
    e = jax.jit(lambda x: bundle(x, False))(point)
    for actual, expected in zip(jax.tree.leaves(a), jax.tree.leaves(e)):
        np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=2e-10)
