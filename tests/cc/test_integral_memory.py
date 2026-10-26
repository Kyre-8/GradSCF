"""Restricted CC blocks avoid copying the complete active MO tensor."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.cc.integrals import prepare_integrals


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("frozen", [None, 1, [0, 4], [0, 1, 2], [3, 4, 5]])
def test_selected_blocks_without_active_four_index_copy(monkeypatch, dtype, frozen):
    rng = np.random.default_rng(9)
    h = rng.normal(size=(6, 6)).astype(dtype)
    h = (h + h.T) * dtype(0.5)
    factors = rng.normal(size=(4, 6, 6)).astype(dtype)
    factors = (factors + factors.swapaxes(1, 2)) * dtype(0.5)
    g = np.einsum("Lpq,Lrs->pqrs", factors, factors)
    removed = set(range(frozen)) if isinstance(frozen, int) else set(frozen or ())
    occ = [i for i in range(3) if i not in removed]
    vir = [i for i in range(3, 6) if i not in removed]
    active = np.asarray(occ + vir, dtype=np.int32)
    ix = jnp.ix_

    def selected_ix(*indices):
        if occ and vir and len(indices) == 4 and all(
            np.array_equal(index, active) for index in indices
        ):
            raise AssertionError("Complete active-four-index ERI copy requested")
        return ix(*indices)

    monkeypatch.setattr(jnp, "ix_", selected_ix)
    result = prepare_integrals(h, g, nocc=3, frozen=frozen, nuclear_repulsion=1.2)
    fock = h + 2 * np.einsum("pqii->pq", g[:, :, :3, :3])
    fock -= np.einsum("piiq->pq", g[:, :3, :3, :])
    expected_fock = fock[np.ix_(active, active)]
    tolerance = 2e-6 if dtype == np.float32 else 2e-12
    np.testing.assert_allclose(result.fock, expected_fock, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(result.mo_energy, np.diag(expected_fock), atol=tolerance)
    np.testing.assert_allclose(
        result.reference_energy, np.trace((h + fock)[:3, :3]) + 1.2,
        atol=tolerance, rtol=tolerance,
    )
    spaces = {
        "oooo": (occ, occ, occ, occ),
        "ovoo": (occ, vir, occ, occ),
        "ovov": (occ, vir, occ, vir),
        "oovv": (occ, occ, vir, vir),
        "ovvo": (occ, vir, vir, occ),
        "ovvv": (occ, vir, vir, vir),
        "vvvv": (vir, vir, vir, vir),
    }
    for name, indices in spaces.items():
        block = getattr(result, name)
        assert block.dtype == dtype
        np.testing.assert_array_equal(block, g[np.ix_(*indices)])
    assert result.nocc == len(occ)
    assert result.nvir == len(vir)


def test_nvir_uses_fock_space_without_accessing_vvvv():
    result = prepare_integrals(jnp.eye(5), jnp.zeros((5,) * 4), nocc=2)
    assert result._replace(vvvv=None).nvir == 3


def test_block_gradients_and_hessian_vector_products():
    rng = np.random.default_rng(11)
    h = jnp.asarray(rng.normal(size=(5, 5)))
    g = jnp.asarray(rng.normal(size=(5,) * 4))
    tangent = jnp.asarray(rng.normal(size=g.shape))
    active = jnp.asarray([1, 2, 3])

    def selected(tensor):
        ints = prepare_integrals(h, tensor, nocc=2, frozen=[0, 4])
        return sum(jnp.sum(block ** 2) for block in ints[2:9])

    def dense_reference(tensor):
        tensor = tensor[jnp.ix_(active, active, active, active)]
        o, v = slice(None, 1), slice(1, None)
        blocks = (tensor[o, o, o, o], tensor[o, v, o, o],
                  tensor[o, v, o, v], tensor[o, o, v, v],
                  tensor[o, v, v, o], tensor[o, v, v, v], tensor[v, v, v, v])
        return sum(jnp.sum(block ** 2) for block in blocks)

    for objective in (selected, dense_reference):
        value, gradient = jax.jit(jax.value_and_grad(objective))(g)
        hvp = jax.jit(lambda tensor, direction: jax.jvp(
            jax.grad(objective), (tensor,), (direction,)
        )[1])(g, tangent)
        if objective is selected:
            actual = value, gradient, hvp
        else:
            for result, expected in zip(actual, (value, gradient, hvp)):
                np.testing.assert_allclose(result, expected, atol=2e-12, rtol=2e-12)
