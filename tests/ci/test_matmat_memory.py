"""Direct CI columns share a compact bounded loop without changing response."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.ci.direct import build_direct_hamiltonian
from gradscf.ci.space import make_ci_space, make_uci_space


def _problem(unrestricted=False):
    rng = np.random.default_rng(2309)
    h = jnp.asarray(rng.normal(scale=.2, size=(3, 3)))
    h = .5*(h+h.T)
    factors = rng.normal(scale=.05, size=(4, 3, 3))
    factors = factors+factors.transpose(0, 2, 1)
    g = jnp.asarray(np.einsum('Lpq,Lrs->pqrs', factors, factors))
    if unrestricted:
        space = make_uci_space(3, (2, 1), max_excitation=2)
        return (h, .9*h), (g, .8*g, .7*g), space
    return h, g, make_ci_space(3, 1, max_excitation=2)


def test_many_rhs_use_one_compact_loop():
    h, g, space = _problem()
    operator = build_direct_hamiltonian(h, g, space, block_size=2)
    vectors = jnp.ones((space.size, 13))
    graph = jax.make_jaxpr(operator.apply)(vectors).jaxpr
    # The old Python list retained four separately expanded batch results
    # before concatenate. The shared loop must have one static body instead.
    assert sum(e.primitive.name == 'scan' for e in graph.eqns) == 1
    assert not any(e.primitive.name == 'concatenate' for e in graph.eqns)


@pytest.mark.parametrize('unrestricted', [False, True])
@pytest.mark.parametrize('columns', [0, 1, 3, 4, 5, 13])
def test_batched_tail_matches_individual_actions(unrestricted, columns):
    h, g, space = _problem(unrestricted)
    operator = build_direct_hamiltonian(h, g, space, block_size=2)
    rng = np.random.default_rng(2310)
    vectors = jnp.asarray(rng.normal(size=(space.size, columns)))
    actual = jax.jit(operator.apply)(vectors)
    expected = (jnp.stack([operator.matvec(vectors[:, i]) for i in range(columns)], axis=1)
                if columns else jnp.empty_like(vectors))
    assert actual.shape == vectors.shape and actual.dtype == vectors.dtype
    if os.environ.get('GRADSCF_TEST_GPU') == '1':
        assert actual.devices() == {jax.devices('gpu')[0]}
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize('unrestricted', [False, True])
def test_batched_integral_state_jvp_vjp_hvp(unrestricted):
    h, g, space = _problem(unrestricted)
    rng = np.random.default_rng(2311)
    vectors = jnp.asarray(rng.normal(scale=.2, size=(space.size, 5)))
    direction = jnp.asarray(rng.normal(scale=.1, size=vectors.shape))

    def loss(t, states, batch):
        hh = jax.tree.map(lambda x: x*jnp.exp(.1*t), h)
        gg = jax.tree.map(lambda x: x*(1+.2*t), g)
        operator = build_direct_hamiltonian(hh, gg, space, block_size=2)
        value = (operator.apply(states) if batch else
                 jnp.stack([operator.matvec(states[:, i]) for i in range(5)], axis=1))
        return jnp.sum(jnp.sin(value)+.1*value**2)

    def response(batch):
        function = lambda t, x: loss(t, x, batch)
        point, tangent = (.2, vectors), (.1, direction)
        return (jax.value_and_grad(function, argnums=(0, 1))(*point),
                jax.jvp(function, point, tangent),
                jax.jvp(jax.grad(function, argnums=(0, 1)), point, tangent)[1])

    actual, expected = (jax.jit(lambda: response(batch))() for batch in (True, False))
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.isfinite(np.asarray(a)).all()
        assert np.isfinite(np.asarray(e)).all()
        if os.environ.get('GRADSCF_TEST_GPU') == '1':
            assert a.devices() == {jax.devices('gpu')[0]}
        np.testing.assert_allclose(a, e, atol=3e-12, rtol=3e-12, equal_nan=False)
