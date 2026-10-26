"""Analytic native coordinate Hessian products; finite differences are oracles."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals


def problem(operator, cart=True, d_shell=False):
    basis = {'H': [[0, [1.1, 1.]], [2, [.7, 1.]]]} if d_shell else '3-21g'
    top, p = integrals.prepare_basis('H 0 0 0; H .2 -.1 .8', basis=basis, cart=cart)
    plan = integrals.make_plan(top, backend='native')
    nnuc = len(top.nuclear_charges)
    x = jnp.concatenate((p.nuclear_coords, p.centers, jnp.array([[.1, -.2, .3]])))
    def value(coords):
        params = replace(p, nuclear_coords=coords[:nnuc], centers=coords[nnuc:-1])
        return plan.evaluate(operator, params, **({'origin':coords[-1]} if operator=='dipole' else {}))
    return value, x


def fd(function, x, direction, step=2e-4):
    return (8*(function(x+step*direction)-function(x-step*direction))
            -function(x+2*step*direction)+function(x-2*step*direction))/(12*step)


@pytest.mark.parametrize('cart', [True, False])
@pytest.mark.parametrize('operator', ['overlap','kinetic','nuclear','dipole','eri'])
def test_second_jvp_and_hvp_match_derivatives_of_first_order(operator, cart):
    value, x = problem(operator, cart)
    rng = np.random.default_rng(991)
    u,v = [jnp.asarray(rng.normal(size=x.shape))*.2 for _ in range(2)]
    cot = jnp.asarray(rng.normal(size=value(x).shape))
    first = lambda r: jax.jvp(value,(r,),(v,))[1]
    second = jax.jit(lambda r,d:jax.jvp(first,(r,),(d,))[1])(x,u)
    np.testing.assert_allclose(second,fd(first,x,u),atol=3e-8,rtol=3e-7)
    reverse = lambda r:jax.grad(lambda q:jnp.sum(value(q)*cot))(r)
    hvp = jax.jit(lambda r,d:jax.jvp(reverse,(r,),(d,))[1])(x,v)
    np.testing.assert_allclose(hvp,fd(reverse,x,v),atol=5e-8,rtol=3e-7)
    np.testing.assert_allclose(jnp.sum(hvp*u),jnp.sum(second*cot),atol=2e-10)


def test_explicit_hessian_modes_nonlinear_cotangent_and_vmap():
    value,x = problem('nuclear')
    loss = lambda r:jnp.sum(jnp.sin(value(r)*.3))
    forward = jax.jit(jax.jacfwd(jax.grad(loss)))(x)
    reverse = jax.jit(jax.jacrev(jax.grad(loss)))(x)
    np.testing.assert_allclose(forward,reverse,atol=2e-10)
    matrix = np.asarray(forward).reshape(x.size,x.size)
    np.testing.assert_allclose(matrix,matrix.T,atol=2e-10)
    probes=jnp.stack((jnp.ones_like(x),jnp.arange(x.size).reshape(x.shape)*.03))
    mapped=jax.jit(jax.vmap(lambda d:jax.jvp(jax.grad(loss),(x,),(d,))[1]))(probes)
    np.testing.assert_allclose(mapped.reshape(2,-1),probes.reshape(2,-1)@matrix.T,atol=2e-10)


@pytest.mark.parametrize('cart', [True, False])
@pytest.mark.parametrize('operator', ['overlap','kinetic','nuclear','dipole','eri'])
def test_second_geometry_with_d_shells(operator,cart):
    value,x=problem(operator,cart,d_shell=True)
    rng=np.random.default_rng(43)
    u,v=[jnp.asarray(rng.normal(size=x.shape))*.1 for _ in range(2)]
    first=lambda r:jax.jvp(value,(r,),(v,))[1]
    observed=jax.jit(lambda r:jax.jvp(first,(r,),(u,))[1])(x)
    np.testing.assert_allclose(observed,fd(first,x,u),atol=4e-8,rtol=4e-7)


def test_origin_origin_zero_and_translation_sum_rule():
    value,x=problem('dipole')
    cot=jnp.arange(value(x).size,dtype=float).reshape(value(x).shape)/17
    hessian=jax.jacfwd(jax.grad(lambda r:jnp.sum(value(r)*cot)))(x).reshape(x.size,x.size)
    np.testing.assert_allclose(hessian[-3:,-3:],0.,atol=1e-13)
    # Translate all nuclei, basis centers AND the dipole origin together.
    for axis in range(3):
        direction=jnp.zeros_like(x).at[:,axis].set(1.).ravel()
        np.testing.assert_allclose(hessian@direction,0.,atol=2e-10)


@pytest.mark.parametrize('zero_direction', [False, True])
def test_geometry_fifth_order_remains_unsupported(zero_direction):
    value,x=problem('overlap')
    first=lambda r:jax.jvp(value,(r,),(jnp.ones_like(r),))[1]
    second=lambda r:jax.jvp(first,(r,),(jnp.ones_like(r),))[1]
    third=lambda r:jax.jvp(second,(r,),(jnp.ones_like(r),))[1]
    fourth=lambda r:jax.jvp(third,(r,),(jnp.ones_like(r),))[1]
    direction = jnp.zeros_like(x) if zero_direction else jnp.ones_like(x)
    with pytest.raises(NotImplementedError,match='(?i)fifth.*geometry'):
        jax.jvp(fourth,(x,),(direction,))


def test_second_products_differentiate_directions_and_cotangents():
    value,x=problem('dipole')
    rng=np.random.default_rng(918)
    u,v=[jnp.asarray(rng.normal(size=x.shape)) for _ in range(2)]
    cot=jnp.asarray(rng.normal(size=value(x).shape))
    product=lambda a,b:jax.jvp(lambda r:jax.jvp(value,(r,),(b,))[1],(x,),(a,))[1]
    expected=jax.jvp(jax.grad(lambda r:jnp.sum(value(r)*cot)),(x,),(v,))[1]
    observed=jax.jit(jax.grad(lambda a:jnp.sum(product(a,v)*cot)))(u)
    np.testing.assert_allclose(observed,expected,atol=2e-10)
    pull=lambda c:jax.jvp(jax.grad(lambda r:jnp.sum(value(r)*c)),(x,),(v,))[1]
    transposed=jax.jit(jax.grad(lambda c:jnp.sum(pull(c)*u)))(cot)
    np.testing.assert_allclose(transposed,product(u,v),atol=2e-10)
    np.testing.assert_allclose(jax.jvp(lambda a:product(a,v),(u,),(2*u,))[1],2*product(u,v),atol=2e-10)
