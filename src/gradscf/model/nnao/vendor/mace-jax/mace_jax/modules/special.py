import jax.numpy as jnp


def chebyshev_polynomial_t(x: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorized Chebyshev polynomial of the first kind.
    x: [batch, num_basis] or [batch, 1]
    n: [num_basis] or [batch, num_basis] (integers)
    Returns: [batch, num_basis]
    """
    x = jnp.asarray(x)
    n = jnp.asarray(n)
    if not jnp.issubdtype(n.dtype, jnp.integer):
        n = n.astype(jnp.int32)

    # Ensure n has same batch shape as x
    while n.ndim < x.ndim:
        n = jnp.expand_dims(n, 0)  # from (num_basis,) → (1, num_basis)
    n = jnp.broadcast_to(n, x.shape)  # [batch, num_basis]
    n_float = n.astype(x.dtype)

    abs_x = jnp.abs(x)
    inside_mask = abs_x <= 1

    inside = jnp.cos(n_float * jnp.arccos(jnp.clip(x, -1.0, 1.0)))

    outside_mag = jnp.cosh(n_float * jnp.arccosh(jnp.maximum(abs_x, 1.0)))
    outside_sign = jnp.where((x < -1) & (n % 2 == 1), -1.0, 1.0).astype(x.dtype)
    outside = outside_sign * outside_mag

    return jnp.where(inside_mask, inside, outside)
