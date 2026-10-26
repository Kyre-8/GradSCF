"""Native Gaussian-exponent products include normalization (CPU float64)."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import integrals


def basis(cart=False):
    return integrals.prepare_basis('H 0 0 0; H .2 .1 .8',
        {'H':[[0,[1.1,.8,.2],[.3,-.2,.9]],
              [1,[.9,.6],[.25,.3]],[2,[.7,.7],[.2,-.1]]]},cart=cart)


def shift(x,d,t):
    return jax.tree.map(lambda a,b:a+t*b,x,d)


def fd4(f,x,d,h=1e-4):
    return (8*(f(shift(x,d,h))-f(shift(x,d,-h)))
            -f(shift(x,d,2*h))+f(shift(x,d,-2*h)))/(12*h)


@pytest.mark.parametrize('cart',[True,False])
@pytest.mark.parametrize('operator',['overlap','kinetic','nuclear','dipole'])
def test_native_exponent_jvp_vjp_normalization_fd(operator,cart):
    top,p=basis(cart);plan=integrals.make_plan(top)
    f=lambda a:plan.evaluate(operator,replace(p,exponents=a))
    d=tuple(.07*jnp.sin(jnp.arange(a.size)+i+1)*a for i,a in enumerate(p.exponents))
    value,tangent=jax.jit(lambda a,d:jax.jvp(f,(a,),(d,)))(p.exponents,d)
    np.testing.assert_allclose(tangent,fd4(f,p.exponents,d),atol=3e-8,rtol=3e-7)
    cot=jnp.asarray(np.random.default_rng(19).normal(size=value.shape))
    g=jax.jit(jax.grad(lambda a:jnp.sum(f(a)*cot)))(p.exponents)
    np.testing.assert_allclose(sum(jnp.vdot(a,b) for a,b in zip(g,d)),
                               jnp.vdot(cot,tangent),atol=3e-10,rtol=3e-10)


@pytest.mark.parametrize('cart',[True,False])
def test_native_df_exponent_and_coefficient_joint_gradient(cart):
    top,p=basis(cart)
    at,ap=integrals.prepare_basis('H 0 0 0; H .2 .1 .8',
        {'H':[[0,[1.4,1.]],[1,[.6,1.]],[2,[.4,1.]]]},cart=cart)
    ri=integrals.make_auxiliary_plan(top,at);metric=ri.metric_factor(p,ap)
    def f(a,c):
        return ri.factors(replace(p,exponents=a,coefficients=c),ap,metric_factor=metric)
    args=(p.exponents,p.coefficients)
    directions=(tuple(.04*x for x in p.exponents),
        tuple(.03*jnp.cos(jnp.arange(x.size).reshape(x.shape)+1) for x in p.coefficients))
    value,dv=jax.jit(lambda x,d:jax.jvp(lambda x:f(*x),(x,),(d,)))(args,directions)
    np.testing.assert_allclose(dv,fd4(lambda x:f(*x),args,directions),atol=3e-8,rtol=3e-7)
    cot=jnp.asarray(np.random.default_rng(23).normal(size=value.shape))
    gradients=jax.jit(jax.grad(lambda x:jnp.sum(f(*x)*cot)))(args)
    dot=sum(jnp.vdot(g,d) for g,d in zip(jax.tree.leaves(gradients),jax.tree.leaves(directions)))
    np.testing.assert_allclose(dot,jnp.vdot(cot,dv),atol=3e-10,rtol=3e-10)


def test_exponent_higher_derivatives_fail_explicitly_until_implemented():
    top,p=integrals.prepare_basis('H 0 0 0','3-21g')
    plan=integrals.make_plan(top)
    f=lambda a:jnp.sum(plan.evaluate('kinetic',replace(p,exponents=(a,*p.exponents[1:]))))
    with pytest.raises(NotImplementedError,match='(?i)(exponent|mixed|higher)'):
        jax.hessian(f)(p.exponents[0])


def test_normalized_primitive_overlap_has_zero_exponent_derivative():
    top,p=integrals.prepare_basis('He 0 0 0',{'He':[[0,[.7,1.]],[1,[.6,1.]],[2,[.5,1.]]]},cart=False)
    plan=integrals.make_plan(top)
    f=lambda a:jnp.diag(plan.evaluate('overlap',replace(p,exponents=a)))
    tangent=jax.jvp(f,(p.exponents,),(p.exponents,))[1]
    np.testing.assert_allclose(tangent,0.,atol=2e-12,rtol=0.)


def test_exponent_jacfwd_jacrev_batch_and_angular_boundary():
    top,p=integrals.prepare_basis('H 0 0 0; H 0 0 .74','3-21g',cart=False)
    plan=integrals.make_plan(top)
    f=lambda a:plan.evaluate('kinetic',replace(p,exponents=(a,*p.exponents[1:])))
    a=p.exponents[0]
    np.testing.assert_allclose(jax.jit(jax.jacfwd(f))(a),jax.jit(jax.jacrev(f))(a),atol=2e-11)
    batch=jax.jit(jax.vmap(jax.grad(lambda x:f(x).sum())))(jnp.stack((a,a*1.03)))
    assert np.isfinite(batch).all()
    high,params=integrals.prepare_basis('H 0 0 0',{'H':[[11,[.7,1.]]]},cart=False)
    hplan=integrals.make_plan(high)
    with pytest.raises(NotImplementedError,match='(?i)(angular|l <= 10)'):
        jax.eval_shape(jax.grad(lambda a:hplan.evaluate('overlap',
            replace(params,exponents=(a,))).sum()),params.exponents[0])


@pytest.mark.parametrize('cart',[True,False])
def test_f_and_g_exponent_angular_raising(cart):
    top,p=integrals.prepare_basis('H 0 0 0; He .3 -.2 .9',
        {'H':[[3,[1.1,.7],[.4,.3]]],'He':[[4,[.8,.6],[.25,-.2]]]},cart=cart)
    plan=integrals.make_plan(top)
    f=lambda a:plan.evaluate('nuclear',replace(p,exponents=a))
    d=jax.tree.map(lambda x:.1*x,p.exponents)
    tangent=jax.jvp(f,(p.exponents,),(d,))[1]
    np.testing.assert_allclose(tangent,fd4(f,p.exponents,d),atol=3e-8,rtol=3e-7)


@pytest.mark.parametrize("alias", ["active_exponent", "auxiliary_exponent",
    "auxiliary_coefficient", "nuclear_coordinate", "shell_coordinate", "origin"])
def test_active_exponent_storage_aliases_are_rejected(alias):
    from gradscf.integrals.backends.native.autodiff.evaluation import _integral_function

    atom = "H 0 0 0; H 0 0 .74"
    top, p = integrals.prepare_basis(atom, "3-21g")
    at, ap = integrals.prepare_basis(atom, "sto-3g")
    ri = integrals.make_auxiliary_plan(top, at)
    atm, bas, _ = ri._pack(p, ap)
    split = len(top.angular_momenta)
    targets = {"active_exponent": bas[1, 5], "auxiliary_exponent": bas[split, 5],
        "auxiliary_coefficient": bas[split, 6], "nuclear_coordinate": atm[0, 1],
        "shell_coordinate": atm[-1, 1], "origin": 1}
    bas = bas.copy()
    bas[0, 5] = targets[alias]
    with pytest.raises(ValueError, match="(?i)(exponent.*overlap|overlap.*exponent)"):
        _integral_function("three_center", tuple(map(tuple, atm.tolist())),
            tuple(map(tuple, bas.tolist())), ri.combined.topology.nao, True, split)


def test_raised_cartesian_workspace_is_checked_before_allocation():
    from gradscf.integrals.basis import BasisTopology, BasisParameters
    from gradscf.integrals.backends.native.autodiff.evaluation import evaluate_differentiable

    # Ordinary three-center Cartesian storage fits int32; raising l by two
    # exceeds it, even though the requested spherical output is smaller.
    top = BasisTopology((10, 10, 10), (1, 1, 1), (13, 13, 13), (1,), False)
    p = BasisParameters(tuple(jnp.array([.7]) for _ in range(3)),
        tuple(jnp.ones((1, 13)) for _ in range(3)), jnp.zeros((3, 3)), jnp.zeros((1, 3)))
    plan = integrals.make_plan(top)
    atm, bas, env, c, alpha = plan.coefficient_data(p, active_shells=2, separate_exponents=True)
    coords = jnp.concatenate((p.nuclear_coords, p.centers))

    def value(a):
        return evaluate_differentiable("three_center", atm, bas, env, c, coords,
            jnp.zeros(3), top.nao, cart=False, split=2, exponents=a).sum()

    with pytest.raises(ValueError, match="raised Cartesian cache"):
        jax.eval_shape(jax.grad(value), alpha)
