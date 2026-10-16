"""AO-pair square-root response must not differentiate an auxiliary gauge."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors_traceable
from gradscf.integrals.molecular.jk import build_jk_from_df


def packed_factors(pair):
    factors = eri_pair_matrix_to_df_factors_traceable(pair, nao=2, tol=1e-8)
    rows, columns = np.tril_indices(2)
    return factors[:, rows, columns]


def test_repeated_auxiliary_values_have_finite_first_response():
    pair = jnp.diag(jnp.array([2., 2., .4]))
    tangent = jnp.array([[.3, .2, -.1], [.2, -.4, .15], [-.1, .15, .2]])
    root, response = jax.jit(lambda a, da: jax.jvp(packed_factors, (a,), (da,)))(pair, tangent)
    np.testing.assert_allclose(root.T@root, pair, atol=1e-13, rtol=0)
    # The full-shape factors choose the principal matrix square root.
    np.testing.assert_allclose(root, root.T, atol=1e-13, rtol=0)
    np.testing.assert_allclose(root@response+response@root, tangent, atol=2e-13, rtol=0)
    _, reverse = jax.vjp(packed_factors, pair)
    probe = jnp.array([[.2, -.1, .3], [-.1, .4, .2], [.3, .2, -.2]])
    np.testing.assert_allclose(jnp.sum(reverse(probe)[0]*tangent), jnp.sum(probe*response),
                               atol=2e-13, rtol=0)


@pytest.mark.parametrize("eigenvalues", [(0., 2., 2.), (2e-9, 2., 2.)])
def test_truncated_null_space_rotation_has_covariant_response(eigenvalues):
    pair = jnp.diag(jnp.array(eigenvalues))
    generator = jnp.array([[0., .3, -.2], [-.3, 0., .1], [.2, -.1, 0.]])
    tangent = generator@pair-pair@generator
    root, response = jax.jvp(packed_factors, (pair,), (tangent,))
    expected = generator@root-root@generator
    np.testing.assert_allclose(response, expected, atol=2e-13, rtol=0)
    # An independent displaced matrix path preserves the cutoff separation.
    from scipy.linalg import expm
    for step in (1e-3, 3e-4, 1e-4):
        plus, minus = expm(step*np.asarray(generator)), expm(-step*np.asarray(generator))
        finite = (packed_factors(plus@pair@plus.T)-packed_factors(minus@pair@minus.T))/(2*step)
        np.testing.assert_allclose(response, finite, atol=2e-7, rtol=0)


def test_coulomb_exchange_derivatives_reconstruct_full_eri_response():
    pair = jnp.diag(jnp.array([2., 2., .4]))
    tangent = jnp.array([[.3, .2, -.1], [.2, -.4, .15], [-.1, .15, .2]])
    density = jnp.array([[.7, -.1], [-.1, .4]])
    def jk(a):
        return build_jk_from_df(eri_pair_matrix_to_df_factors_traceable(a, nao=2, tol=1e-8), density)
    _, (dj, dk) = jax.jvp(jk, (pair,), (tangent,))
    # For two AOs, lower-triangle pair numbering is 00,10,11.
    index = np.array([[0, 1], [1, 2]])
    derivative = tangent[index[:, :, None, None], index[None, None, :, :]]
    np.testing.assert_allclose(dj, jnp.einsum('pqrs,rs->pq', derivative, density), atol=2e-13, rtol=0)
    np.testing.assert_allclose(dk, jnp.einsum('prqs,rs->pq', derivative, density), atol=2e-13, rtol=0)
