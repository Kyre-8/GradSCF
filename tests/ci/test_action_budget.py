"""Direct CI byte-selected string/RHS widths retain the full sigma action."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import ci
from gradscf.ci.direct import build_direct_hamiltonian, build_fluctuation


@pytest.mark.parametrize('builder', [build_direct_hamiltonian, build_fluctuation])
def test_explicit_budget_and_reservation_precede_numeric_kernel(builder, monkeypatch):
    import gradscf.ci.direct as direct
    space = ci.make_ci_space(3, 1)
    h, g = jnp.eye(3), jnp.ones((3,)*4)*.01
    monkeypatch.setattr(direct, '_kernels', lambda *a, **kw: pytest.fail('Budget rejection came after kernel allocation'))
    with pytest.raises(ValueError, match='budget|memory|resident'):
        builder(h, g, space, max_memory=.001, reserved_bytes=4000)


@pytest.mark.parametrize('unrestricted', [False, True])
def test_budgeted_rank_action_joint_jvp_vjp_hvp_matches_default(unrestricted):
    rng = np.random.default_rng(761)
    h = jnp.diag(jnp.array([-1.2, -.7, .4]))
    raw = jnp.asarray(rng.normal(scale=.01, size=(3,)*4))
    g = raw+raw.transpose(1,0,2,3)
    g = g+g.transpose(0,1,3,2)
    g = g+g.transpose(2,3,0,1)
    space = ci.make_uci_space(3, (2,1)) if unrestricted else ci.make_ci_space(3,1)
    hs, gs = ((h,h), (g,g,g)) if unrestricted else (h,g)
    x = jnp.asarray(rng.normal(size=(space.size, 7)))
    direction = jax.tree.map(lambda a: .02*jnp.ones_like(a), (hs,gs,x))
    def loss(point, budget):
        one,two,c = point
        op = build_fluctuation(one,two,space,max_input_rank=1,max_output_rank=2,
            **(dict(max_memory=.5,reserved_bytes=4096) if budget else {}))
        return jnp.sum(op.apply(c)**2)+.1*jnp.sum(op.T.apply(c)**2)
    def bundle(p, budget):
        fun=lambda a:loss(a,budget)
        return jax.value_and_grad(fun)(p),jax.jvp(jax.grad(fun),(p,),(direction,))[1]
    actual=jax.jit(lambda p:bundle(p,True))((hs,gs,x))
    expected=jax.jit(lambda p:bundle(p,False))((hs,gs,x))
    for a,e in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
        np.testing.assert_allclose(a,e,atol=3e-10,rtol=3e-9)
