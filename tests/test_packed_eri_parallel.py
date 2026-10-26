from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.integrals.molecular import jk as packed_eri


def _pair_index(nao: int) -> np.ndarray:
    rows, cols = np.tril_indices(int(nao))
    out = np.empty((int(nao), int(nao)), dtype=np.int32)
    ids = np.arange(rows.size, dtype=np.int32)
    out[rows, cols] = ids
    out[cols, rows] = ids
    return out


def test_packed_exchange_density_gradient_matches_dense_reference():
    nao = 3
    npair = nao*(nao+1)//2
    pair_index = _pair_index(nao)
    raw = np.arange(npair*npair, dtype=float).reshape(npair, npair)/17
    pair = (raw+raw.T)/2
    density = jnp.arange(nao*nao, dtype=float).reshape(nao, nao)/13
    cotangent = jnp.sin(density+1.)
    gradient = jax.grad(lambda d: jnp.sum(
        packed_eri.build_jk_from_eri_pair_matrix(jnp.asarray(pair), d)[1]*cotangent))(density)
    eri = pair[pair_index[:, :, None, None], pair_index[None, None, :, :]]
    expected = np.einsum('pq,prqs->rs', np.asarray(cotangent), eri)
    np.testing.assert_allclose(gradient, expected, atol=1e-12, rtol=1e-12)


def test_packed_exchange_contraction_matches_explicit_reference():
    nao = 4
    npair = nao * (nao + 1) // 2
    pair_index = _pair_index(nao)
    raw_pair = np.arange(1, npair * npair + 1, dtype=np.float64).reshape(npair, npair)
    pair = 0.5 * (raw_pair + raw_pair.T)
    density_raw = np.asarray(
        [
            [0.9, 0.1, -0.2, 0.3],
            [0.0, 0.8, 0.4, -0.1],
            [0.0, 0.0, 0.7, 0.2],
            [0.0, 0.0, 0.0, 0.6],
        ],
        dtype=np.float64,
    )
    density = density_raw + density_raw.T - np.diag(np.diag(density_raw))

    _, k_mat = packed_eri.build_jk_from_eri_pair_matrix(
        jnp.asarray(pair),
        jnp.asarray(density),
    )

    expected = np.zeros((nao, nao), dtype=np.float64)
    for p in range(nao):
        for q in range(nao):
            total = 0.0
            for r in range(nao):
                for s in range(nao):
                    total += pair[pair_index[p, r], pair_index[q, s]] * density[r, s]
            expected[p, q] = total
    expected = 0.5 * (expected + expected.T)

    assert np.allclose(np.asarray(k_mat), expected, atol=1e-10, rtol=1e-10)
