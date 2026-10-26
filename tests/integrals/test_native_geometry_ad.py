"""Independent first-order coordinate JVP/VJP contracts for native integrals."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals


@pytest.mark.parametrize("cart", [True, False])
@pytest.mark.parametrize("operator", ["overlap", "kinetic", "nuclear", "dipole", "eri"])
def test_native_geometry_jvp_vjp_fd_and_adjoint(operator, cart):
    top, params = integrals.prepare_basis("O 0 0 0; H .1 .75 .58; H -.1 -.75 .58",
                                          basis="sto-3g", cart=cart)
    plan = integrals.make_plan(top, backend="native")
    # Nuclear and shell coordinates are independent, so this also covers
    # floating centers and separates attraction-operator and AO responses.
    coords = jnp.concatenate([params.nuclear_coords, params.centers], axis=0)
    nnuc = len(top.nuclear_charges)
    def value(r):
        return plan.evaluate(operator, replace(params, nuclear_coords=r[:nnuc], centers=r[nnuc:]))
    rng = np.random.default_rng(841)
    direction = jnp.asarray(rng.normal(size=coords.shape))
    primal, tangent = jax.jit(lambda r, dr: jax.jvp(value, (r,), (dr,)))(coords, direction)
    h = 1e-4
    fd = (8*(value(coords+h*direction)-value(coords-h*direction))
          - value(coords+2*h*direction)+value(coords-2*h*direction))/(12*h)
    np.testing.assert_allclose(tangent, fd, atol=2e-8, rtol=2e-7)
    cot = jnp.asarray(rng.normal(size=primal.shape))  # intentionally nonsymmetric
    vjp = jax.jit(jax.grad(lambda r: jnp.sum(value(r)*cot)))(coords)
    np.testing.assert_allclose(jnp.sum(vjp*direction), jnp.sum(cot*tangent), atol=2e-10, rtol=2e-11)


def test_native_dipole_origin_gradient():
    top, p = integrals.prepare_basis("H 0 0 0; H .2 .1 .74", basis="3-21g")
    plan = integrals.make_plan(top, backend="native")
    cot = jnp.arange(3*top.nao**2, dtype=float).reshape(3, top.nao, top.nao)/13
    f = lambda o: jnp.sum(cot*plan.evaluate("dipole", p, origin=o))
    actual = jax.jit(jax.grad(f))(jnp.array([.1, -.2, .3]))
    expected = -jnp.einsum("xij,ij->x", cot, plan.evaluate("overlap", p))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("operator,field", [("eri", "exponents"), ("eri", "coefficients")])
def test_native_unimplemented_basis_derivatives_raise(operator, field):
    top, p = integrals.prepare_basis("H 0 0 0; H 0 0 .74", basis="3-21g")
    plan = integrals.make_plan(top, backend="native")
    values = getattr(p, field)
    def f(v):
        return plan.evaluate(operator, replace(p, **{field:(v, *values[1:])})).sum()
    with pytest.raises(NotImplementedError, match="(?i)(exponent|coefficient|basis)"):
        jax.grad(f)(values[0])


@pytest.mark.parametrize("cart", [True, False])
@pytest.mark.parametrize("operator", ["dipole", "eri"])
def test_native_d_shell_coordinate_derivatives(operator, cart):
    top, p = integrals.prepare_basis("H 0 0 0; H .2 .1 .8", cart=cart,
        basis={"H": [[0, [1.1, 1.]], [2, [.7, 1.]]]})
    plan = integrals.make_plan(top, backend="native")
    owners = np.repeat(np.arange(2), 2)
    def value(r):
        return plan.evaluate(operator, replace(p, nuclear_coords=r, centers=r[owners]))
    direction = jnp.array([[.2, -.1, .3], [-.1, .4, -.2]])
    r = p.nuclear_coords
    _, tangent = jax.jvp(value, (r,), (direction,))
    h = 1e-4
    fd = (8*(value(r+h*direction)-value(r-h*direction))-value(r+2*h*direction)+value(r-2*h*direction))/(12*h)
    np.testing.assert_allclose(tangent, fd, atol=2e-9, rtol=2e-7)


def test_native_jacfwd_jacrev_and_vmap():
    top, p = integrals.prepare_basis("H 0 0 0; H .2 .1 .8", basis="3-21g")
    plan = integrals.make_plan(top, backend="native")
    def value(r):
        return plan.evaluate("overlap", replace(p, centers=r))
    forward = jax.jit(jax.jacfwd(value))(p.centers)
    reverse = jax.jit(jax.jacrev(value))(p.centers)
    np.testing.assert_allclose(forward, reverse, atol=2e-12)
    batch = jnp.stack([p.centers, p.centers+.1])
    actual = jax.jit(jax.vmap(jax.grad(lambda r: value(r).sum())))(batch)
    np.testing.assert_allclose(actual[0], actual[1], atol=1e-12)


def test_native_geometry_derivatives_do_not_import_pyscf():
    import os
    from pathlib import Path
    import subprocess
    import sys
    script = '''
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "pyscf" or name.startswith("pyscf."):
        raise AssertionError("Native geometry AD must not import PySCF")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from dataclasses import replace
import jax
import jax.numpy as jnp
from gradscf import integrals
t,p=integrals.prepare_basis("H 0 0 0; H .2 .1 .8",basis="3-21g")
plan=integrals.make_plan(t,backend="native")
for op in ["overlap","kinetic","nuclear","dipole","eri"]:
    f=lambda r:plan.evaluate(op,replace(p,centers=r)).sum()
    g=jax.jit(jax.grad(f))(p.centers)
    _,v=jax.jit(lambda r,d:jax.jvp(f,(r,),(d,)))(p.centers,jnp.ones_like(p.centers))
    assert bool(jnp.all(jnp.isfinite(g))) and bool(jnp.isfinite(v))
    h=jax.jit(jax.hessian(f))(p.centers)
    assert bool(jnp.all(jnp.isfinite(h)))
assert not any(m=="pyscf" or m.startswith("pyscf.") for m in sys.modules)
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
               PYTHONPATH=str(Path("src").resolve()))
    result = subprocess.run([sys.executable,"-c",script],env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
