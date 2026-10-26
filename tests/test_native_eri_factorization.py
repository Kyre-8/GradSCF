"""Spectral factors represent native ERIs without an auxiliary fitting basis."""

import numpy as np
import jax.numpy as jnp

from gradscf.integrals.molecular.jk import build_jk_from_df
from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors_traceable, eri_to_df_factors
from gradscf.integrals.basis import prepare_basis
from gradscf.integrals.plan import make_plan


def test_native_water_spectral_factors_reconstruct_eri_and_jk():
    topology, parameters = prepare_basis(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        unit="Angstrom",
    )
    eri = np.asarray(make_plan(topology, backend="native").evaluate("eri", parameters))
    factors = eri_to_df_factors(eri, tol=1e-12)
    reconstructed = np.einsum("Qpq,Qrs->pqrs", factors, factors)
    np.testing.assert_allclose(reconstructed, eri, atol=2e-12, rtol=0)
    random = np.random.default_rng(42).normal(size=(topology.nao, topology.nao))
    density = 0.5 * (random + random.T)
    j_matrix, k_matrix = build_jk_from_df(factors, jnp.asarray(density))
    np.testing.assert_allclose(j_matrix, np.einsum("pqrs,rs->pq", eri, density), atol=2e-11, rtol=0)
    np.testing.assert_allclose(k_matrix, np.einsum("prqs,rs->pq", eri, density), atol=2e-11, rtol=0)
    rows, cols = np.tril_indices(topology.nao)
    pair = eri[rows[:, None], cols[:, None], rows[None, :], cols[None, :]]
    traceable_factors = eri_pair_matrix_to_df_factors_traceable(
        pair, nao=topology.nao, tol=1e-12
    )
    np.testing.assert_allclose(
        np.einsum("Qpq,Qrs->pqrs", traceable_factors, traceable_factors),
        eri,
        atol=2e-12,
        rtol=0,
    )
