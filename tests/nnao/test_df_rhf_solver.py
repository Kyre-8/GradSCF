"""Fixed-shape DF-RHF response reuses its JIT and differentiates physical inputs."""
from importlib import import_module
import importlib.util
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def module():
    return import_module("gradscf.model.nnao.training.objective")


def inputs():
    s = jnp.array([[1., .08], [.08, 1.1]])
    h = jnp.array([[-1., -.12], [-.12, .3]])
    b = jnp.array([[[.15, .02], [.02, .11]], [[.04, -.01], [-.01, .03]]])
    return s, h, b


def test_changed_integral_values_reuse_scf_and_backward_trace(monkeypatch):
    m = module(); calls = []
    original = m._run_scf_with_rescue
    def traced(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(m, '_run_scf_with_rescue', traced)
    _, value_grad = m._df_rhf_solver(2, 1e-8, 0.)
    data = inputs()
    first, _ = value_grad(data, 0.)
    changed = (data[0], data[1] + .01 * jnp.eye(2), data[2] * 1.02)
    second, gradients = value_grad(changed, 0.)
    assert len(calls) == 1, 'Changing integral values retraces the complete SCF/implicit solver.'
    assert float(first[0]) != float(second[0]) and bool(second[1]['converged'])
    assert all(np.isfinite(g).all() for g in gradients)


def test_scaled_basis_experiments_share_the_complete_scf_backward(monkeypatch):
    m=module();calls=[];original=m._run_scf_with_rescue
    def traced(*args,**kwargs):
        calls.append(True)
        return original(*args,**kwargs)
    monkeypatch.setattr(m,'_run_scf_with_rescue',traced)
    kwargs=dict(basis_family='szp663_direct',core_primitives=6,jk_backend='df',
        implicit_tolerance=1e-8,geometry=dict(name='H2',symbols=['H','H'],
            coords_angstrom=[[0.,0.,0.],[0.,0.,.74]],charge=0,spin=0))
    first=m.BasisRHF(**kwargs)
    changed=m.BasisRHF(**kwargs,log_exponent_scales={('H',0):.12})
    e1,g1,_=first.evaluate(first.layout.reference_outputs())
    e2,g2,_=changed.evaluate(changed.layout.reference_outputs())
    assert len(calls)==1, 'A new exponent pool recreates the full SCF/backward compilation.'
    assert e1!=e2 and np.isfinite(g1).all() and np.isfinite(g2).all()


def test_physical_input_implicit_gradient_and_hvp_match_finite_differences():
    m = module(); value, value_grad = m._df_rhf_solver(2, 1e-8, 0.)
    data = inputs()
    direction = (jnp.array([[.03, -.01], [-.01, .02]]),
                 jnp.array([[.02, .015], [.015, -.01]]), data[2] * .1)
    e = lambda x: value(x, .2)[0]
    shift = lambda t: jax.tree.map(lambda x, d: x + t * d, data, direction)
    (_, _), gradient = value_grad(data, .2)
    step = 1e-4
    fd = (8 * (e(shift(step)) - e(shift(-step)))
          - e(shift(2 * step)) + e(shift(-2 * step))) / (12 * step)
    actual = sum(jnp.vdot(g, d) for g, d in zip(gradient, direction))
    np.testing.assert_allclose(actual, fd, atol=2e-8, rtol=2e-7)
    gradient_fn = jax.grad(e)
    hvp = jax.jvp(gradient_fn, (data,), (direction,))[1]
    fd_hvp = jax.tree.map(lambda a, b: (a-b)/(2*step),
                         gradient_fn(shift(step)), gradient_fn(shift(-step)))
    for actual, expected in zip(hvp, fd_hvp):
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)


def test_native_contracted_integrals_compose_with_implicit_scf_without_primitive_cache():
    from gradscf import integrals, scf
    from gradscf.integrals.basis.contraction import primitive_basis, contraction_matrix
    from gradscf.integrals.molecular.density_fitting import project_factors, unpack_factors

    atom='H 0 0 0; H 0 0 .74'
    top,p=integrals.prepare_basis(atom,{'H':[[0,[1.2,.7],[.28,.35]]]},cart=False)
    at,ap=integrals.prepare_basis(atom,'def2-universal-jkfit',cart=False)
    plan=integrals.make_plan(top)
    ri=integrals.make_auxiliary_plan(top,at)
    metric=ri.metric_factor(p,ap)
    enuc=scf.nuclear_repulsion_energy(p.nuclear_coords,jnp.asarray(top.nuclear_charges))
    value,_=module()._df_rhf_solver(2,1e-8,0.)

    def native_loss(c):
        bound=replace(p,coefficients=c)
        physical=(plan.evaluate('overlap',bound),
            plan.evaluate('kinetic',bound)+plan.evaluate('nuclear',bound),
            unpack_factors(ri.factors(bound,ap,metric_factor=metric),top.nao))
        return value(physical,enuc)[0]

    # The production branch above requests only contracted native integrals.
    # Primitive data are used solely as an independent test reference.
    pt,pp=primitive_basis(top,p);primitive=integrals.make_plan(pt)
    sp=primitive.evaluate('overlap',pp)
    hp=primitive.evaluate('kinetic',pp)+primitive.evaluate('nuclear',pp)
    bp=integrals.make_auxiliary_plan(pt,at).factors(pp,ap,metric_factor=metric)
    def reference_loss(c):
        t=contraction_matrix(top,replace(p,coefficients=c))
        return value((t.T@sp@t,t.T@hp@t,project_factors(bp,t)),enuc)[0]

    c=p.coefficients
    direction=(jnp.array([[.17],[-.24]]),jnp.array([[.05],[-.08]]))
    actual=jax.jit(jax.value_and_grad(native_loss))(c)
    expected=jax.jit(jax.value_and_grad(reference_loss))(c)
    np.testing.assert_allclose(actual[0],expected[0],atol=2e-11,rtol=0.)
    for a,b in zip(actual[1],expected[1]):
        np.testing.assert_allclose(a,b,atol=2e-8,rtol=2e-7)
    hvp=jax.jit(lambda c,d:jax.jvp(jax.grad(native_loss),(c,),(d,))[1])(c,direction)
    correct=jax.jit(lambda c,d:jax.jvp(jax.grad(reference_loss),(c,),(d,))[1])(c,direction)
    for a,b in zip(hvp,correct):
        np.testing.assert_allclose(a,b,atol=3e-7,rtol=3e-6)
    step=1e-4
    shift=lambda h:jax.tree.map(lambda a,b:a+h*b,c,direction)
    fd=(8*(native_loss(shift(step))-native_loss(shift(-step)))
        -native_loss(shift(2*step))+native_loss(shift(-2*step)))/(12*step)
    np.testing.assert_allclose(sum(jnp.vdot(a,b) for a,b in zip(actual[1],direction)),
                               fd,atol=2e-8,rtol=2e-7)
