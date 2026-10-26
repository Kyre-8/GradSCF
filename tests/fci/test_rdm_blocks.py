"""True spatial FCI density blocks without a complete n**4 density export."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import fci
from gradscf.fci import rdm


def _entry(name):
    function = getattr(rdm, name, None)
    assert callable(function), 'Missing tiled FCI density-block export'
    return function


@pytest.mark.parametrize('nelec', [(2, 2), (2, 1), (0, 2)])
def test_rectangular_transition_blocks_match_full_oracle(nelec):
    space = fci.make_fci_space(4, nelec)
    rng = np.random.default_rng(751)
    bra, ket = (jnp.asarray(rng.normal(size=space.shape)) for _ in range(2))
    indices = ((3, 0), (1,), (2, 1, 0), (0, 3))
    one, expected = rdm.trans_rdm12s(bra, ket, space)
    actual = _entry('trans_rdm2_block')(bra, ket, space, indices, spin_resolved=True,
                                       max_workspace_elements=64, max_memory=8.)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e[jnp.ix_(*map(jnp.asarray, indices))], atol=2e-11)
    np.testing.assert_allclose(_entry('trans_rdm2_block')(bra, ket, space, indices),
                              sum(actual), atol=2e-11)


def test_normalized_block_retains_jvp_vjp_hvp_and_small_output_cap():
    space = fci.make_fci_space(4, (2, 1))
    c = jnp.asarray(np.random.default_rng(752).normal(size=space.shape))
    indices = ((0,), (0, 1), (2,), (1, 2, 3))
    block = _entry('make_rdm2_block')
    direction = .03*jnp.cos(c)
    def scalar(x, sliced):
        tensor = (block(x, space, indices, max_workspace_elements=32) if sliced else
                  rdm.make_rdm2(x, space)[jnp.ix_(*map(jnp.asarray, indices))])
        return jnp.sum(tensor*tensor)
    def bundle(x, sliced):
        function = lambda a: scalar(a, sliced)
        return jax.value_and_grad(function)(x), jax.jvp(jax.grad(function), (x,), (direction,))[1]
    actual, expected = jax.jit(lambda x: bundle(x, True))(c), jax.jit(lambda x: bundle(x, False))(c)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, atol=2e-10, rtol=2e-10)
    with pytest.raises(ValueError, match='indices|orbital'):
        block(c, space, ((0,), (0,), (4,), (0,)))
