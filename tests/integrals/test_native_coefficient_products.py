"""Native coefficient products at fixed exponents and geometry (CPU float64)."""
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.basis.normalization import normalized_shell_coefficients


def test_native_coefficient_backend_is_available():
    assert importlib.util.find_spec(
        "gradscf.integrals.backends.native.autodiff.evaluation"
    ) is not None, "Native coefficient JVP/VJP backend is missing"


def _backend():
    test_native_coefficient_backend_is_available()
    from gradscf.integrals.backends.native.autodiff.evaluation import evaluate_differentiable
    return evaluate_differentiable


def _case(cart=True, custom=False):
    basis = ({"H": [
        [0, [1.1, .8, .15], [.3, -.2, .9]],
        [1, [.9, .6, -.4], [.25, .2, .7]],
        [2, [.7, .7, .2], [.2, -.1, .8]],
    ]} if custom else "3-21g")
    atom = "H 0 0 0; H .2 .1 .8" if custom else "O 0 0 0; H .1 .75 .58; H -.1 -.75 .58"
    top, params = integrals.prepare_basis(atom, basis=basis, cart=cart)
    plan = integrals.make_plan(top, backend="native")
    atm, bas, env = plan.pack(params)
    indices = np.concatenate([
        np.arange(int(row[6]), int(row[6]) + int(row[2])*int(row[3]))
        for row in bas
    ])
    coords = jnp.concatenate([params.nuclear_coords, params.centers])
    coeff = env[indices]
    origin = jnp.array([.1, -.2, .3], dtype=jnp.float64)
    fixed = env.at[indices].set(0).at[1:4].set(0)
    for row in atm:
        start = int(row[1])
        fixed = fixed.at[start:start+3].set(0)
    return top, params, plan, fixed, coeff, coords, origin, indices


def _fourth_order_fd(value, x, direction, h=1e-4):
    return (8*(value(x+h*direction)-value(x-h*direction))
            - value(x+2*h*direction)+value(x-2*h*direction))/(12*h)


@pytest.mark.parametrize("cart", [True, False])
@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("operator", ["overlap", "kinetic", "nuclear", "dipole"])
def test_normalized_coefficient_values_jvp_vjp(operator, custom, cart):
    evaluate = _backend()
    from gradscf.integrals.backends.native import evaluate as raw_evaluate
    top, _, plan, fixed, coeff, coords, origin, indices = _case(cart, custom)
    value = lambda c: evaluate(operator, plan.atm, plan.bas, fixed, c, coords,
                              origin, top.nao, cart=cart)
    # Active ENV coefficients have no normalization assumption, so individual
    # zeros and negative entries must also have well-defined derivatives.
    coeff = coeff.at[0].set(0).at[1].set(-abs(coeff[1]))
    env = fixed.at[indices].set(coeff).at[1:4].set(origin)
    for row, xyz in zip(plan.atm, coords):
        start = int(row[1])
        env = env.at[start:start+3].set(xyz)
    expected = raw_evaluate(operator, plan.atm, plan.bas, env, top.nao, cart)
    rng = np.random.default_rng(293)
    direction = jnp.asarray(rng.normal(size=coeff.shape))
    actual, tangent = jax.jit(lambda c, d: jax.jvp(value, (c,), (d,)))(coeff, direction)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(tangent, _fourth_order_fd(value, coeff, direction),
                               atol=2e-8, rtol=2e-8)
    cot = jnp.asarray(rng.normal(size=actual.shape))  # nonsymmetric on purpose
    gradient = jax.jit(jax.grad(lambda c: jnp.sum(cot*value(c))))(coeff)
    np.testing.assert_allclose(jnp.vdot(gradient, direction), jnp.vdot(cot, tangent),
                               atol=2e-10, rtol=2e-11)


@pytest.mark.parametrize("operator", ["overlap", "kinetic", "nuclear", "dipole"])
def test_coefficient_hessian_products_and_cotangent_derivative(operator):
    evaluate = _backend()
    top, _, plan, fixed, coeff, coords, origin, _ = _case()
    value = lambda c: evaluate(operator, plan.atm, plan.bas, fixed, c, coords,
                              origin, top.nao)
    rng = np.random.default_rng(501)
    u, v = (jnp.asarray(rng.normal(size=coeff.shape)) for _ in range(2))
    cot, dcot = (jnp.asarray(rng.normal(size=value(coeff).shape)) for _ in range(2))
    grad = jax.grad(lambda c: jnp.sum(cot*value(c)))
    hvp = jax.jit(lambda c, d: jax.jvp(grad, (c,), (d,))[1])(coeff, u)
    np.testing.assert_allclose(hvp, _fourth_order_fd(grad, coeff, u),
                               atol=2e-8, rtol=2e-8)
    bilinear = jax.jit(lambda c: jax.jvp(
        lambda x: jax.jvp(value, (x,), (v,))[1], (c,), (u,))[1])(coeff)
    np.testing.assert_allclose(jnp.vdot(v, hvp), jnp.vdot(cot, bilinear),
                               atol=2e-10, rtol=2e-11)
    pull = lambda w: jax.grad(lambda c: jnp.sum(w*value(c)))(coeff)
    cotangent_derivative = jax.jit(lambda w, dw: jax.jvp(pull, (w,), (dw,))[1])(cot, dcot)
    np.testing.assert_allclose(cotangent_derivative, pull(dcot), atol=2e-12)
    # A fixed-normalized-coefficient integral is quadratic. Its exact third
    # coefficient derivative is zero; this is not an unavailable AD rule.
    third = jax.jit(jax.jacfwd(jax.jacfwd(grad)))(coeff)
    np.testing.assert_array_equal(third, jnp.zeros_like(third))


def test_coefficient_jacfwd_jacrev_vmap_and_raw_normalization_chain():
    evaluate = _backend()
    top, params = integrals.prepare_basis("H 0 0 0; H .2 .1 .8", "3-21g")
    plan = integrals.make_plan(top, backend="native")
    atm, bas, env = plan.pack(params)
    indices = np.concatenate([np.arange(int(b[6]), int(b[6])+int(b[2])*int(b[3])) for b in bas])
    fixed = env.at[indices].set(0)
    coords = jnp.concatenate([params.nuclear_coords, params.centers])
    origin = jnp.zeros(3, dtype=jnp.float64)
    value = lambda c: evaluate("overlap", atm, bas, fixed, c, coords, origin, top.nao)
    coeff = env[indices]
    forward = jax.jit(jax.jacfwd(value))(coeff)
    reverse = jax.jit(jax.jacrev(value))(coeff)
    np.testing.assert_allclose(forward, reverse, atol=2e-12)
    batch = jnp.stack([coeff, coeff+.03])
    actual = jax.jit(jax.vmap(jax.grad(lambda c: value(c).sum())))(batch)
    np.testing.assert_allclose(actual, jnp.stack([jax.grad(lambda c: value(c).sum())(c) for c in batch]),
                               atol=2e-12)
    sizes = [c.size for c in params.coefficients]
    offsets = np.cumsum([0, *sizes])
    raw = jnp.concatenate([c.ravel() for c in params.coefficients])
    cot = jnp.arange(top.nao**2, dtype=jnp.float64).reshape(top.nao, top.nao)/17
    def objective(r):
        normalized = jnp.concatenate([
            normalized_shell_coefficients(l, a, r[start:end].reshape(c.shape)).T.ravel()
            for l, a, c, start, end in zip(top.angular_momenta, params.exponents,
                                          params.coefficients, offsets[:-1], offsets[1:])
        ])
        return jnp.sum(cot*value(normalized))
    u = jnp.linspace(-.2, .3, raw.size)
    gradient = jax.grad(objective)
    second = lambda r: jax.jvp(gradient, (r,), (u,))[1]
    third = jax.jit(lambda r: jax.jvp(second, (r,), (u,))[1])(raw)
    assert float(jnp.linalg.norm(third)) > 1e-4
    np.testing.assert_allclose(third, _fourth_order_fd(second, raw, u, h=5e-4),
                               atol=2e-7, rtol=2e-6)


def test_fixed_environment_and_mixed_geometry_derivatives_fail_closed():
    evaluate = _backend()
    top, _, plan, fixed, coeff, coords, origin, _ = _case()
    f = lambda e, c, r: evaluate("kinetic", plan.atm, plan.bas, e, c, r,
                                origin, top.nao).sum()
    with pytest.raises(NotImplementedError, match="(?i)(exponent|fixed|environment)"):
        jax.grad(f, argnums=0)(fixed, coeff, coords)
    with pytest.raises(NotImplementedError, match="(?i)(geometry|mixed|environment)"):
        jax.jacfwd(jax.grad(f, argnums=1), argnums=2)(fixed, coeff, coords)
    with pytest.raises(NotImplementedError, match="(?i)(geometry|mixed|coefficient|basis)"):
        jax.jacfwd(jax.grad(f, argnums=2), argnums=1)(fixed, coeff, coords)
    # The preserved geometry path still supports coordinate Hessians at a
    # fixed coefficient vector, and agrees with the existing native wrapper.
    from gradscf.integrals.backends.native.autodiff.geometry import evaluate_geometry
    env = fixed
    for b, start in zip(plan.bas, np.cumsum([0, *[int(b[2])*int(b[3]) for b in plan.bas]])):
        n = int(b[2])*int(b[3]); ce = int(b[6])
        env = env.at[ce:ce+n].set(coeff[start:start+n])
    old = lambda r: evaluate_geometry("kinetic", plan.atm, plan.bas, env,
                                     r, origin, top.nao).sum()
    new_hessian = jax.jit(jax.hessian(lambda r: f(fixed, coeff, r)))(coords)
    np.testing.assert_allclose(new_hessian, jax.jit(jax.hessian(old))(coords), atol=2e-12)


def test_simultaneous_coefficient_coordinate_and_origin_first_derivative():
    evaluate = _backend()
    top, _, plan, fixed, coeff, coords, origin, _ = _case()
    rng = np.random.default_rng(310)
    cot = jnp.asarray(rng.normal(size=(3, top.nao, top.nao)))
    value = lambda c, r, o: jnp.sum(cot*evaluate("dipole", plan.atm, plan.bas,
        fixed, c, r, o, top.nao))
    directions = tuple(jnp.asarray(rng.normal(size=x.shape)) for x in (coeff, coords, origin))
    primals = (coeff, coords, origin)
    tangent = jax.jit(lambda c, r, o, dc, dr, do:
                      jax.jvp(value, (c, r, o), (dc, dr, do))[1])(*primals, *directions)
    along_line = lambda t: value(*(x+t*d for x, d in zip(primals, directions)))
    fd = _fourth_order_fd(along_line, 0., 1.)
    np.testing.assert_allclose(tangent, fd, atol=2e-8, rtol=2e-8)
    gradients = jax.jit(jax.grad(value, argnums=(0, 1, 2)))(*primals)
    np.testing.assert_allclose(tangent, sum(jnp.vdot(g, d) for g, d in zip(gradients, directions)),
                               atol=2e-10, rtol=2e-11)


@pytest.mark.parametrize("alias", ["active_coefficient", "exponent", "nuclear_coordinate",
                                  "shell_coordinate", "origin", "auxiliary_coefficient",
                                  "auxiliary_exponent"])
def test_active_coefficient_storage_aliases_are_rejected(alias):
    evaluate = _backend()
    from gradscf.integrals.molecular.density_fitting import make_auxiliary_plan
    top, p = integrals.prepare_basis("H 0 0 0; H .2 .1 .8", "3-21g")
    split = 0
    operator = "overlap"
    if alias.startswith("auxiliary"):
        at, ap = integrals.prepare_basis("H 0 0 0; H .2 .1 .8", "sto-3g")
        auxiliary = make_auxiliary_plan(top, at)
        plan = auxiliary.combined
        atm, bas, env = auxiliary._pack(p, ap)
        coords = jnp.concatenate((p.nuclear_coords, p.centers, ap.centers))
        split = len(top.angular_momenta)
        operator = "three_center"
    else:
        plan = integrals.make_plan(top)
        atm, bas, env = plan.pack(p)
        coords = jnp.concatenate((p.nuclear_coords, p.centers))
    active = bas[:split] if split else bas
    indices = np.concatenate([np.arange(int(b[6]), int(b[6])+int(b[2])*int(b[3])) for b in active])
    coeff = env[indices]
    fixed = env.at[indices].set(0)
    bas = bas.copy()
    if alias == "active_coefficient":
        bas[1, 6] = bas[0, 6]
    else:
        target = {"exponent": int(bas[1, 5]),
                  "nuclear_coordinate": int(atm[0, 1]),
                  "shell_coordinate": int(atm[-1, 1]), "origin": 1}
        if split:
            target["auxiliary_coefficient"] = int(bas[split, 6])
            target["auxiliary_exponent"] = int(bas[split, 5])
        bas[0, 6] = target[alias]
    with pytest.raises(ValueError, match="(?i)(coefficient.*overlap|overlap.*coefficient|alias)"):
        evaluate(operator, atm, bas, fixed, coeff, coords, jnp.array([.1, .2, .3]),
                 plan.topology.nao, split=split)


def _three_center_metadata_case(primitive_counts, contraction_counts):
    from gradscf.integrals.basis import BasisTopology, BasisParameters
    top = BasisTopology((11, 11, 11), primitive_counts, contraction_counts,
                        (1,), False)
    parameters = BasisParameters(
        tuple(jnp.linspace(.2, 2., n) for n in primitive_counts),
        tuple(jnp.ones((np_, nc), dtype=jnp.float64)
              for np_, nc in zip(primitive_counts, contraction_counts)),
        jnp.zeros((3, 3), dtype=jnp.float64),
        jnp.zeros((1, 3), dtype=jnp.float64))
    plan = integrals.make_plan(top)
    atm, bas, env, coeff = plan.coefficient_data(parameters, active_shells=2)
    coords = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
    evaluate = _backend()
    value = lambda c: evaluate("three_center", atm, bas, env, c, coords,
        jnp.zeros(3, dtype=jnp.float64), top.nao, cart=False, split=2)
    norb = 23*sum(contraction_counts[:2])
    shape = (23*contraction_counts[2], norb*(norb+1)//2)
    return value, jax.ShapeDtypeStruct(coeff.shape, jnp.float64), shape


@pytest.mark.parametrize("product", ["vjp", "hessian_vjp"])
def test_spherical_three_center_primal_safe_but_identity_vjp_cache_overflows(product):
    value, coefficients, shape = _three_center_metadata_case((64, 1, 64), (1, 1, 64))
    # Regular Cartesian cache lower bound 3*78*78*(78*64) fits int32.
    # Replacing one orbital nctr by nprim=64 exceeds int32, despite the
    # smaller spherical output. Only abstract shapes are traced here.
    assert jax.eval_shape(value, coefficients).shape == shape
    first = lambda c, d: jax.jvp(value, (c,), (d,))[1]
    assert jax.eval_shape(first, coefficients, coefficients).shape == shape
    second = lambda c, d: jax.jvp(lambda x: first(x, d), (c,), (d,))[1]
    assert jax.eval_shape(second, coefficients, coefficients).shape == shape
    cotangent = jax.ShapeDtypeStruct(shape, jnp.float64)
    with pytest.raises(ValueError, match="(?i)(Cartesian|identity|cache|int32)"):
        if product == "vjp":
            jax.eval_shape(lambda c, cot: jax.grad(lambda x: jnp.sum(cot*value(x)))(c),
                           coefficients, cotangent)
        else:
            jax.eval_shape(lambda c, d, cot: jax.grad(lambda x: jnp.sum(cot*first(x, d)))(c),
                           coefficients, coefficients, cotangent)


def test_three_center_regular_cartesian_cache_overflow_is_rejected_before_output():
    value, coefficients, _ = _three_center_metadata_case((1, 1, 1), (64, 64, 64))
    with pytest.raises(ValueError, match="(?i)(Cartesian|cache|int32)"):
        jax.eval_shape(value, coefficients)
