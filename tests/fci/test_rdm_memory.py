"""FCI densities use determinant tiles; requested RDM output remains explicit."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import fci
from gradscf.fci import rdm


@pytest.mark.parametrize('norb,nelec', [(4, (2, 2)), (5, (2, 1)), (4, (0, 2)), (3, (3, 3))])
def test_tiled_transition_and_normalized_densities_match_pyscf(monkeypatch, norb, nelec):
    from pyscf.fci import direct_spin1, spin_op
    space = fci.make_fci_space(norb, nelec)
    rng = np.random.default_rng(2201)
    bra, ket = (rng.normal(size=space.shape) for _ in range(2))
    expected = direct_spin1.trans_rdm12s(bra, ket, norb, nelec)
    norm = ket / np.linalg.norm(ket)
    expected_normal = direct_spin1.make_rdm12s(norm, norb, nelec)
    expected_spin = spin_op.spin_square(norm, norb, nelec)

    def forbidden(*args, **kwargs):
        pytest.fail('An RDM constructed complete orbital-pair excitation stacks')

    monkeypatch.setattr(rdm, 'excitation_actions', forbidden, raising=False)
    # Large enough for the requested n^4 output; smaller than n^2*Ndet in
    # the nontrivial sectors. This is a tile cap, not a determinant-space cap.
    limit = max(norb**4, norb**2)
    actual = rdm.trans_rdm12s(bra, ket, space, max_workspace_elements=limit)
    normalized = rdm.make_rdm12s(-2.3*ket, space, max_workspace_elements=limit)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, atol=3e-11, rtol=2e-12)
    for a, e in zip(jax.tree.leaves(normalized), jax.tree.leaves(expected_normal)):
        np.testing.assert_allclose(a, e, atol=3e-12, rtol=2e-12)
    actual_one = rdm.trans_rdm1s(bra, ket, space, max_workspace_elements=norb**2)
    np.testing.assert_allclose(actual_one, expected[0], atol=3e-11, rtol=2e-12)
    np.testing.assert_allclose(rdm.spin_square(ket, space, max_workspace_elements=norb**2),
                               expected_spin, atol=3e-12, rtol=2e-12)


def test_tiled_rdm_raw_bra_ket_gradient_and_hvp():
    space = fci.make_fci_space(4, (2, 1))
    rng = np.random.default_rng(2202)
    bra, ket = (jnp.asarray(rng.normal(size=space.shape)) for _ in range(2))
    probe = jnp.asarray(rng.normal(size=(4,)*4))
    direction = (jnp.ones_like(bra)*.03, jnp.ones_like(ket)*-.02)

    def objective(left, right, cap):
        one, two = rdm.trans_rdm12(left, right, space, max_workspace_elements=cap)
        return jnp.sum(one**2) + .02*jnp.sum(two*probe)

    full = lambda a, b: objective(a, b, 100000)
    blocked = lambda a, b: objective(a, b, 256)
    expected = jax.jit(jax.value_and_grad(full, argnums=(0, 1)))(bra, ket)
    actual = jax.jit(jax.value_and_grad(blocked, argnums=(0, 1)))(bra, ket)
    expected_hvp = jax.jvp(jax.grad(full, argnums=(0, 1)), (bra, ket), direction)[1]
    actual_hvp = jax.jit(lambda a, b: jax.jvp(jax.grad(blocked, argnums=(0, 1)),
                         (a, b), direction)[1])(bra, ket)
    for a, e in zip(jax.tree.leaves((actual, actual_hvp)),
                    jax.tree.leaves((expected, expected_hvp))):
        np.testing.assert_allclose(a, e, atol=3e-10, rtol=2e-12)
    for step in (3e-4, 1e-4):
        gradient = jax.jit(jax.grad(blocked, argnums=(0, 1)))
        plus = gradient(bra+step*direction[0], ket+step*direction[1])
        minus = gradient(bra-step*direction[0], ket-step*direction[1])
        for a, p, m in zip(actual_hvp, plus, minus):
            np.testing.assert_allclose(a, (p-m)/(2*step), atol=3e-6, rtol=2e-6)


def test_rdm_memory_budget_rejected_before_gather(monkeypatch):
    space = fci.make_fci_space(4, (2, 2))
    c = jnp.ones(space.shape)

    def forbidden(*args, **kwargs):
        pytest.fail('RDM budget was checked after link/gather work')

    monkeypatch.setattr(rdm, '_link_maps', forbidden)
    with pytest.raises(ValueError, match='(?i)(memory|budget|resident)'):
        rdm.trans_rdm12s(c, c, space, max_memory=.0001)
