"""Selective cached J/K retains density and independent s8 integral AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.backends.native.jk import packed_jk


@pytest.mark.parametrize("with_j,with_k", [(True, True), (True, False), (False, True), (False, False)])
def test_packed_selected_operators_match_independent_full_oracle(with_j, with_k):
    topology, parameters = integrals.prepare_basis("H 0 0 0; H 0 0 1.4", "6-31g", cart=True)
    eri = integrals.make_plan(topology).evaluate("eri", parameters, aosym="s8")
    n = topology.nao
    ids = np.arange(n)
    hi, lo = np.maximum(ids[:, None], ids), np.minimum(ids[:, None], ids)
    pairs = hi*(hi+1)//2 + lo
    hi = np.maximum(pairs[:, :, None, None], pairs[None, None, :, :])
    lo = np.minimum(pairs[:, :, None, None], pairs[None, None, :, :])
    indices = jnp.asarray(hi*(hi+1)//2 + lo)
    density = jnp.asarray(np.random.default_rng(920).normal(size=(2, n, n)))

    def oracle(e, d):
        full = e[indices]
        return (jnp.einsum("pqrs,brs->bpq", full, d) if with_j else jnp.zeros_like(d),
                jnp.einsum("prqs,brs->bpq", full, d) if with_k else jnp.zeros_like(d))

    action = lambda e, d: packed_jk(e, d, with_j=with_j, with_k=with_k)
    for actual, expected in zip(jax.jit(action)(eri, density), oracle(eri, density)):
        np.testing.assert_allclose(actual, expected, atol=3e-12)
    loss = lambda apply, e, d: sum(jnp.sum(x*x) for x in apply(e, d))
    gradients = [jax.jit(jax.grad(lambda e, d: loss(apply, e, d), (0, 1)))(eri, density)
                 for apply in (action, oracle)]
    for actual, expected in zip(*gradients):
        np.testing.assert_allclose(actual, expected, atol=3e-10)
