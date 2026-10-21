"""UCC selected spin blocks match the original dense spin expansion, including AD."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.cc.uccsd import SpinCCIntegrals, prepare_ucc_integrals


def _dense_reference(h1, eri, nocc, frozen=None, nuclear_repulsion=0.0):
    """Independent tiny-system oracle retaining the original full-spin formulas."""
    n = h1[0].shape[0]
    dtype = jnp.result_type(*h1, *eri)
    h = jnp.zeros((2*n, 2*n), dtype=dtype)
    h = h.at[:n, :n].set(h1[0]).at[n:, n:].set(h1[1])
    g = jnp.zeros((2*n,) * 4, dtype=dtype)
    g = g.at[:n, :n, :n, :n].set(eri[0])
    g = g.at[:n, :n, n:, n:].set(eri[1])
    g = g.at[n:, n:, :n, :n].set(eri[1].transpose(2, 3, 0, 1))
    g = g.at[n:, n:, n:, n:].set(eri[2])
    if frozen is None:
        frozen = ((), ())
    elif isinstance(frozen, int):
        frozen = (tuple(range(frozen)),) * 2
    full_occ = jnp.asarray(list(range(nocc[0])) + list(range(n, n+nocc[1])),
                           dtype=jnp.int32)
    fock = h + jnp.sum(g[:, :, full_occ, full_occ], axis=-1)
    fock -= jnp.sum(g[:, full_occ, full_occ, :], axis=1)
    energy = .5*jnp.sum((h + fock)[full_occ, full_occ]) + nuclear_repulsion
    occ = [[s*n+i for i in range(no) if i not in fr]
           for s, (no, fr) in enumerate(zip(nocc, frozen))]
    vir = [[s*n+i for i in range(no, n) if i not in fr]
           for s, (no, fr) in enumerate(zip(nocc, frozen))]
    active = jnp.asarray(occ[0] + occ[1] + vir[0] + vir[1], dtype=jnp.int32)
    fock = fock[jnp.ix_(active, active)]
    g = g[jnp.ix_(active, active, active, active)]
    g = g.transpose(0, 2, 1, 3) - g.transpose(0, 2, 3, 1)
    o, v = slice(None, sum(map(len, occ))), slice(sum(map(len, occ)), None)
    ints = SpinCCIntegrals(fock, g[o, o, o, o], g[o, o, o, v],
                          g[o, o, v, v], g[o, v, o, v],
                          g[o, v, v, v], g[v, v, v, v], energy)
    return ints, tuple(map(len, occ)), tuple(map(len, vir))


def _raw_integrals(n, dtype):
    # Deliberately impose no physical permutation symmetry on h or any ERI block.
    rng = np.random.default_rng(17)
    h = tuple(jnp.asarray(rng.normal(size=(n, n)), dtype=dtype) for _ in range(2))
    g = tuple(jnp.asarray(rng.normal(size=(n,) * 4), dtype=dtype) for _ in range(3))
    return h, g


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("nocc,frozen", [
    ((2, 1), None), ((2, 2), 1), ((2, 1), ([0, 3], [2])),
    ((0, 2), None), ((2, 0), None), ((4, 1), None), ((1, 4), None),
    ((0, 0), None), ((4, 4), None), ((2, 1), ([0, 1], [0])),
])
def test_all_spin_blocks_match_dense_reference(dtype, nocc, frozen):
    h, g = _raw_integrals(4, dtype)
    actual = prepare_ucc_integrals(h, g, nocc=nocc, frozen=frozen,
                                   nuclear_repulsion=1.3)
    expected = _dense_reference(h, g, nocc, frozen, nuclear_repulsion=1.3)
    assert actual[1:] == expected[1:]
    tolerance = 3e-6 if dtype == jnp.float32 else 2e-12
    for block, reference in zip(actual[0], expected[0]):
        assert block.dtype == dtype
        np.testing.assert_allclose(block, reference, atol=tolerance, rtol=tolerance)


def test_full_spin_and_active_tensor_expansion_is_forbidden(monkeypatch):
    import gradscf.ao2mo as ao2mo
    import gradscf.ao2mo.spin as spin
    import gradscf.cc.uccsd as uccsd

    h, g = _raw_integrals(4, jnp.float64)
    frozen = ([0], [3])
    expected = _dense_reference(h, g, (2, 1), frozen)
    active = np.asarray([1, 4, 2, 3, 5, 6], dtype=np.int32)
    zeros, ix = jnp.zeros, jnp.ix_

    def reject_expansion(*args, **kwargs):
        raise AssertionError("Complete spin-orbital ERI expansion requested")

    def selected_zeros(shape, *args, **kwargs):
        if tuple(np.atleast_1d(shape)) in ((8,) * 4, (6,) * 4):
            raise AssertionError("Complete spin-orbital or active-spin ERI allocated")
        return zeros(shape, *args, **kwargs)

    def selected_ix(*indices):
        if len(indices) == 4 and all(np.array_equal(i, active) for i in indices):
            raise AssertionError("Complete active-spin ERI copy requested")
        return ix(*indices)

    for module in (ao2mo, spin, uccsd):
        monkeypatch.setattr(module, "spin_orbital_integrals", reject_expansion,
                            raising=False)
    monkeypatch.setattr(jnp, "zeros", selected_zeros)
    monkeypatch.setattr(jnp, "ix_", selected_ix)
    actual = prepare_ucc_integrals(h, g, nocc=(2, 1), frozen=frozen)
    for block, reference in zip(actual[0], expected[0]):
        np.testing.assert_allclose(block, reference, atol=2e-12, rtol=2e-12)


def test_mixed_precision_uses_common_spin_dtype():
    h, g = _raw_integrals(4, jnp.float32)
    h = h[0].astype(jnp.float64), h[1]
    actual = prepare_ucc_integrals(h, g, nocc=(2, 1))[0]
    expected = _dense_reference(h, g, (2, 1))[0]
    for block, reference in zip(actual, expected):
        assert block.dtype == jnp.float64
        np.testing.assert_allclose(block, reference, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("nocc,frozen", [
    ((2, 1), None), ((2, 1), ([0], [2])), ((0, 2), None), ((1, 1), 1),
])
def test_raw_integral_gradients_and_hvp_match_dense_reference(nocc, frozen):
    h, g = _raw_integrals(3, jnp.float64)
    leaves = h + g
    rng = np.random.default_rng(19)
    directions = tuple(jnp.asarray(rng.normal(size=a.shape)) for a in leaves)

    def objective(variables, prepare):
        ints = prepare(variables[:2], variables[2:])[0]
        return sum((i+1)*jnp.sum(x**2) for i, x in enumerate(ints))

    selected = lambda variables: objective(variables, lambda h, g:
        prepare_ucc_integrals(h, g, nocc=nocc, frozen=frozen))
    dense = lambda variables: objective(variables, lambda h, g:
        _dense_reference(h, g, nocc, frozen))
    actual = jax.jit(jax.value_and_grad(selected))(leaves)
    expected = jax.jit(jax.value_and_grad(dense))(leaves)
    for result, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(result, reference, atol=2e-11, rtol=2e-12)
    actual = jax.jit(lambda x, v: jax.jvp(jax.grad(selected), (x,), (v,))[1])(
        leaves, directions)
    expected = jax.jit(lambda x, v: jax.jvp(jax.grad(dense), (x,), (v,))[1])(
        leaves, directions)
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, atol=2e-11, rtol=2e-12)
