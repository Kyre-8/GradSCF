"""T/Lambda and observable response retain the dense real density chart."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_rdm2_blocks import _states


@pytest.mark.parametrize('unrestricted,frozen',[(False,[0,3]),(True,([0],[]))])
def test_joint_state_observable_jvp_vjp_hvp(unrestricted,frozen):
    module = importlib.import_module('gradscf.cc.properties')
    n,nocc,result,left = _states(unrestricted,frozen)
    operators = tuple(jnp.asarray(np.random.default_rng(39+s).normal(size=(n,)*4))*.03
                      for s in range(3 if unrestricted else 1))
    weights = (.5,1.,.5) if unrestricted else (.5,)

    def objective(theta,*,dense):
        state = result._replace(t1=jax.tree.map(lambda a:a*(1+theta[0]),result.t1),
                               t2=jax.tree.map(lambda a:a*(1-.4*theta[1]),result.t2))
        adjoint = left._replace(l1=jax.tree.map(lambda a:a*(1+.2*theta[1]),left.l1),
                               l2=jax.tree.map(lambda a:a*(1-.3*theta[0]),left.l2))
        ops = tuple(g*(1+.1*theta[0]-.2*theta[1]) for g in operators)
        if dense:
            density = module._rdm2_dense_from_states(state,adjoint,nmo=n,nocc=nocc,
                                                     frozen=frozen,dtype=jnp.float64)
            densities = density if unrestricted else (density,)
            return sum(w*jnp.sum(g*d) for w,g,d in zip(weights,ops,densities))
        return module._contract_rdm2_from_states(ops if unrestricted else ops[0],
            state,adjoint,nmo=n,nocc=nocc,frozen=frozen,dtype=jnp.float64,max_memory=32.)

    theta = jnp.asarray([.04,-.03])
    direction = jnp.asarray([.7,-.4])
    actual = lambda t:objective(t,dense=False)
    oracle = lambda t:objective(t,dense=True)
    bundle = lambda f:jax.jit(lambda t,v:jax.jvp(jax.value_and_grad(f),(t,),(v,)))(theta,direction)
    observed,expected = bundle(actual),bundle(oracle)
    for a,b in zip(jax.tree.leaves(observed),jax.tree.leaves(expected)):
        np.testing.assert_allclose(a,b,atol=2e-11,rtol=0)
    (_,gradient),(jvp,hvp) = observed
    np.testing.assert_allclose(jvp,jnp.vdot(gradient,direction),atol=2e-13,rtol=0)
    for step in (5e-5,2e-5):
        difference = (jax.grad(actual)(theta+step*direction)
                      -jax.grad(actual)(theta-step*direction))/(2*step)
        np.testing.assert_allclose(hvp,difference,atol=2e-8,rtol=0)


def test_implicit_cc_lambda_two_body_observable_hvp(radical):
    """A fixed-MO one-body perturbation; this is not a nuclear derivative."""
    from gradscf import cc
    module = importlib.import_module('gradscf.cc.properties')
    _,h,g = radical
    frozen = ([0],[])
    config = cc.CCConfig(conv_tol=1e-12,residual_tol=1e-11)
    direction = jnp.diag(jnp.asarray([.13,-.07,.05]))
    probe = jnp.arange(81,dtype=jnp.float64).reshape((3,)*4)/300
    operators = probe,probe*.7,probe*.4

    def observable(x):
        hs = h[0]+x*direction,h[1]
        result = cc.run_ucc(hs,g,nocc=(2,1),frozen=frozen,config=config)
        left = cc.solve_lambda(hs,g,result,nocc=(2,1),frozen=frozen,config=config)
        return module._contract_rdm2_from_states(operators,result,left,
            nmo=3,nocc=(2,1),frozen=frozen,dtype=jnp.float64,config=config,max_memory=32.)

    ((value,gradient),(jvp,hvp)) = jax.jit(lambda x:jax.jvp(jax.value_and_grad(observable),
        (x,),(jnp.asarray(1.),)))(jnp.asarray(0.))
    assert jnp.isfinite(value) and abs(gradient) > 1e-9 and abs(hvp) > 1e-9
    np.testing.assert_allclose(gradient,jvp,atol=2e-12,rtol=0)
    grad = jax.jit(jax.grad(observable))
    for step in (1e-4,5e-5):
        first = (observable(step)-observable(-step))/(2*step)
        second = (grad(step)-grad(-step))/(2*step)
        np.testing.assert_allclose(gradient,first,atol=2e-7,rtol=0)
        np.testing.assert_allclose(hvp,second,atol=2e-6,rtol=0)
