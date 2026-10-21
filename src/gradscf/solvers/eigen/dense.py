"""Small dense eigenproblem reference path, without unrolling eigenderivatives."""
import jax
import jax.numpy as jnp
from ..memory import apply_columns


def dense_vectors(operator, *, nroots, max_dense, batch_size=None):
    n = operator.shape[0]
    if n > max_dense:
        raise ValueError("Dense eigensolve exceeds max_dense; use Davidson")
    matrix = jax.lax.stop_gradient(apply_columns(
        operator, jnp.eye(n, dtype=operator.dtype), batch_size))
    return jnp.linalg.eigh(matrix)[1][:, :nroots]
