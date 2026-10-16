"""Fixed-omega erf-Coulomb ERIs: independent values and coordinate AD."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals


ATOM = "H 0 0 0; H .2 -.1 .8"
BASIS = {"H": [[0, [1.1, 1.]], [2, [.7, 1.]]]}


def problem(cart, aosym="s1", omega=0.2):
    top, p = integrals.prepare_basis(ATOM, basis=BASIS, cart=cart)
    plan = integrals.make_plan(top, backend="native")
    # Independent nuclear/AO coordinates also catch accidental nuclear
    # attraction changes and derivative leakage into the fixed omega slot.
    coords = jnp.concatenate((p.nuclear_coords, p.centers))
    nnuc = len(top.nuclear_charges)

    def value(r):
        params = replace(p, nuclear_coords=r[:nnuc], centers=r[nnuc:])
        return plan.evaluate("eri", params, aosym=aosym, omega=omega)

    return top, p, plan, value, coords


def fd(function, x, direction, step=2e-4):
    return (8 * (function(x + step * direction) - function(x - step * direction))
            - function(x + 2 * step * direction)
            + function(x - 2 * step * direction)) / (12 * step)


@pytest.mark.parametrize("cart", [True, False])
@pytest.mark.parametrize("aosym", ["s1", "s4", "s8"])
def test_long_range_values_and_zero_omega_regression(cart, aosym):
    pyscf = pytest.importorskip("pyscf")
    _, p, plan, value, coords = problem(cart, aosym)
    mol = pyscf.gto.M(atom=ATOM, basis=BASIS, cart=cart, verbose=0)
    with mol.with_range_coulomb(0.2):
        expected = mol.intor("int2e", aosym=aosym)
    actual = jax.jit(value)(coords)
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)
    ordinary = plan.evaluate("eri", p, aosym=aosym)
    zero = plan.evaluate("eri", p, aosym=aosym, omega=0.0)
    np.testing.assert_array_equal(zero, ordinary)
    np.testing.assert_allclose(zero, mol.intor("int2e", aosym=aosym),
                               atol=3e-12, rtol=3e-11)
    assert np.max(np.abs(np.asarray(actual - ordinary))) > 1e-3


@pytest.mark.parametrize("cart", [True, False])
@pytest.mark.parametrize("aosym", ["s1", "s4", "s8"])
def test_long_range_geometry_jvp_vjp_hvp(cart, aosym):
    _, _, _, value, coords = problem(cart, aosym)
    rng = np.random.default_rng(20261015)
    u, v = [jnp.asarray(rng.normal(size=coords.shape)) * .2 for _ in range(2)]
    primal, tangent = jax.jit(lambda r, d: jax.jvp(value, (r,), (d,)))(coords, v)
    np.testing.assert_allclose(tangent, fd(value, coords, v), atol=2e-9, rtol=3e-7)
    cot = jnp.asarray(rng.normal(size=primal.shape))
    loss = lambda r: jnp.sum(value(r) * cot)
    grad = jax.jit(jax.grad(loss))
    gradient = grad(coords)
    np.testing.assert_allclose(jnp.sum(gradient * v), jnp.sum(cot * tangent),
                               atol=2e-10, rtol=2e-11)
    np.testing.assert_allclose(jnp.sum(gradient * v), fd(loss, coords, v),
                               atol=2e-9, rtol=3e-7)
    hvp = jax.jit(lambda r, d: jax.jvp(grad, (r,), (d,))[1])(coords, v)
    np.testing.assert_allclose(hvp, fd(grad, coords, v), atol=5e-8, rtol=4e-7)
    first = lambda r: jax.jvp(value, (r,), (v,))[1]
    second = jax.jit(lambda r: jax.jvp(first, (r,), (u,))[1])(coords)
    np.testing.assert_allclose(second, fd(first, coords, u), atol=4e-8, rtol=4e-7)
    np.testing.assert_allclose(jnp.sum(hvp * u), jnp.sum(second * cot), atol=2e-10)
    np.testing.assert_allclose(gradient[:2], 0.0, atol=1e-13)
    translation = jnp.ones_like(coords)
    np.testing.assert_allclose(jax.jvp(value, (coords,), (translation,))[1],
                               0.0, atol=2e-12)


@pytest.mark.parametrize("omega", [-.2, np.inf, -np.inf, np.nan, 1e200])
def test_invalid_omega_is_rejected(omega):
    _, p, plan, _, _ = problem(True)
    with pytest.raises(ValueError, match="(?i)omega.*(nonnegative|finite)"):
        plan.evaluate("eri", p, omega=omega)


def test_raw_native_omega_overflow_is_rejected():
    from gradscf.integrals.backends.native import evaluate
    top, p, plan, _, _ = problem(True)
    atm, bas, env = plan.pack(p)
    with pytest.raises(Exception, match="(?i)omega"):
        evaluate("eri", atm, bas, env.at[8].set(1e200), top.nao,
                 cart=top.cart).block_until_ready()


@pytest.mark.parametrize("operator", ["overlap", "kinetic", "nuclear", "dipole", "ecp"])
def test_omega_is_only_an_eri_parameter(operator):
    _, p, plan, _, _ = problem(True)
    with pytest.raises(ValueError, match="(?i)omega.*eri"):
        plan.evaluate(operator, p, omega=.2)


def test_omega_derivatives_and_dynamic_jit_are_rejected():
    _, p, plan, _, _ = problem(True)
    f = lambda omega: plan.evaluate("eri", p, omega=omega).sum()
    with pytest.raises(NotImplementedError, match="(?i)omega.*static"):
        jax.grad(f)(.2)
    with pytest.raises(NotImplementedError, match="(?i)omega.*static"):
        jax.jit(f)(.2)


def test_long_range_reference_backend_is_explicitly_unsupported():
    top, p, _, _, _ = problem(True)
    plan = integrals.make_plan(top, backend="jax_reference")
    with pytest.raises(NotImplementedError, match="(?i)(omega|long.range).*native"):
        plan.evaluate("eri", p, omega=.2)


def test_long_range_basis_and_third_geometry_ad_remain_unsupported():
    _, p, plan, value, coords = problem(True)
    for field in ("exponents", "coefficients"):
        values = getattr(p, field)
        f = lambda x: plan.evaluate("eri", replace(p, **{field: (x, *values[1:])}),
                                   omega=.2).sum()
        with pytest.raises(NotImplementedError, match="(?i)(exponent|coefficient|basis)"):
            jax.grad(f)(values[0])
    first = lambda r: jax.jvp(value, (r,), (jnp.ones_like(r),))[1]
    second = lambda r: jax.jvp(first, (r,), (jnp.ones_like(r),))[1]
    with pytest.raises(NotImplementedError, match="(?i)third.*geometry"):
        jax.jvp(second, (coords,), (jnp.ones_like(coords),))
