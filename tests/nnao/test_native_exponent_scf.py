"""Native alpha/c gradients drive variational optimization through implicit SCF."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from gradscf import integrals, scf
from gradscf.integrals.molecular.density_fitting import unpack_factors


def solver_module():
    from gradscf.model.nnao.training import objective
    return objective


def problem():
    module = solver_module()
    atom='H 0 0 0; H 0 0 .74'
    top,p=integrals.prepare_basis(atom,{'H':[[0,[1.2,.7],[.28,.35]]]},cart=False)
    at,ap=integrals.prepare_basis(atom,'def2-universal-jkfit',cart=False)
    plan=integrals.make_plan(top);ri=integrals.make_auxiliary_plan(top,at)
    metric=ri.metric_factor(p,ap)
    enuc=scf.nuclear_repulsion_energy(p.nuclear_coords,jnp.asarray(top.nuclear_charges))
    solve,_=module._df_rhf_solver(2,1e-8,0.)
    def energy(x):
        alpha=p.exponents[0]*jnp.exp(x[:2])
        c=x[2:,None]
        bound=replace(p,exponents=(alpha,alpha),coefficients=(c,c))
        s=plan.evaluate('overlap',bound)
        h=plan.evaluate('kinetic',bound)+plan.evaluate('nuclear',bound)
        b=unpack_factors(ri.factors(bound,ap,metric_factor=metric),top.nao)
        return solve((s,h,b),enuc)
    return energy,jnp.array([0.,0.,.7,.35]),module.BasisRHF._checked_info


def test_log_exponent_and_coefficient_energy_gradients_match_reconverged_fd():
    energy,x,check=problem()
    (value,info),g=jax.jit(jax.value_and_grad(energy,has_aux=True))(x)
    check(value,info)
    assert np.isfinite(g).all() and np.linalg.norm(g[:2])>1e-4
    h=1e-4
    f=jax.jit(lambda x:energy(x)[0])
    fd=jnp.stack([(8*(f(x+h*d)-f(x-h*d))-f(x+2*h*d)+f(x-2*h*d))/(12*h)
                  for d in jnp.eye(4)])
    np.testing.assert_allclose(g,fd,atol=3e-8,rtol=3e-7)


def test_joint_adam_decreases_converged_energy_with_positive_exponents():
    import optax
    energy,x,check=problem()
    evaluate=jax.jit(jax.value_and_grad(energy,has_aux=True))
    (initial,info),gradient=evaluate(x);check(initial,info)
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.02))
    state=optimizer.init(x)
    for _ in range(12):
        updates,trial_state=optimizer.update(gradient,state,x)
        trial=optax.apply_updates(x,updates)
        (value,info),g=evaluate(trial)
        check(value,info)
        assert np.isfinite(g).all()
        x,state,gradient=trial,trial_state,g
    assert float(value)<float(initial)-1e-3
    assert np.isfinite(np.exp(x[:2])).all() and np.all(np.exp(x[:2])>0.)
    assert not np.allclose(x[:2],0.) and not np.allclose(x[2:],[.7,.35])


def test_water_log_exponent_gradient_matches_reconverged_scf():
    """Multiple occupied orbitals exercise a less symmetric SCF response."""
    module = solver_module()
    atom = 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587'
    top, p = integrals.prepare_basis(atom, '3-21g', cart=False)
    at, ap = integrals.prepare_basis(atom, 'def2-universal-jkfit', cart=False)
    plan = integrals.make_plan(top)
    ri = integrals.make_auxiliary_plan(top, at)
    metric = ri.metric_factor(p, ap)
    enuc = scf.nuclear_repulsion_energy(p.nuclear_coords, jnp.asarray(top.nuclear_charges))
    solve, _ = module._df_rhf_solver(10, 1e-8, 0.)
    direction = tuple(.1*jnp.cos(jnp.arange(a.size) + i) for i, a in enumerate(p.exponents))

    def energy(t):
        bound = replace(p, exponents=tuple(a*jnp.exp(t*d) for a, d in zip(p.exponents, direction)))
        s = plan.evaluate('overlap', bound)
        h = plan.evaluate('kinetic', bound) + plan.evaluate('nuclear', bound)
        factors = unpack_factors(ri.factors(bound, ap, metric_factor=metric), top.nao)
        return solve((s, h, factors), enuc)

    (value, info), gradient = jax.jit(jax.value_and_grad(energy, has_aux=True))(0.)
    module.BasisRHF._checked_info(value, info)
    evaluate = jax.jit(energy)

    def checked(t):
        value, info = evaluate(t)
        module.BasisRHF._checked_info(value, info)
        return value

    step = 1e-4
    finite_difference = (8*(checked(step)-checked(-step))-checked(2*step)+checked(-2*step))/(12*step)
    np.testing.assert_allclose(gradient, finite_difference, atol=2e-7, rtol=2e-6)
