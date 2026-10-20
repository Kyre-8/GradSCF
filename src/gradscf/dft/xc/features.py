"""Spin-density conventions and invariant-preserving local feature bridges."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .types import RestrictedFeatureBundle

_DENSITY_FLOOR = 1e-12


def restricted_feature_bundle_from_rho_grad_tau(
    rho: Array,
    grad: Array | None = None,
    tau: Array | None = None,
    *,
    density_floor: float = _DENSITY_FLOOR,
) -> RestrictedFeatureBundle:
    """Build a restricted-spin semilocal bundle from (rho, grad rho, tau)."""

    rho = jnp.maximum(jnp.asarray(rho), density_floor)
    if grad is None:
        grad = jnp.zeros(rho.shape + (3,), dtype=rho.dtype)
    else:
        grad = jnp.asarray(grad, dtype=rho.dtype)
    if tau is None:
        tau = jnp.zeros_like(rho)
    else:
        tau = jnp.maximum(jnp.asarray(tau, dtype=rho.dtype), 0.0)

    half_rho = 0.5 * rho
    half_grad = 0.5 * grad
    half_sigma = jnp.einsum("...x,...x->...", half_grad, half_grad)
    half_tau = 0.5 * tau
    return RestrictedFeatureBundle(
        rho_a=half_rho,
        rho_b=half_rho,
        sigma_aa=half_sigma,
        sigma_ab=half_sigma,
        sigma_bb=half_sigma,
        tau_a=half_tau,
        tau_b=half_tau,
    )


def spin_gradients_from_features(features):
    """Realize the complete 2x2 spin-gradient Gram matrix in Cartesian space.

    The tiny positive radicand supplies a finite one-sided zero-gradient
    convention. sigma_ab is retained, including antiparallel spin gradients.
    Invalid concrete Gram matrices are rejected instead of being reinterpreted.
    """
    aa = jnp.asarray(features.sigma_aa)
    ab = jnp.asarray(features.sigma_ab, dtype=aa.dtype)
    bb = jnp.asarray(features.sigma_bb, dtype=aa.dtype)
    if not any(isinstance(value, jax.core.Tracer) for value in (aa, ab, bb)):
        aa_np, ab_np, bb_np = np.asarray(aa), np.asarray(ab), np.asarray(bb)
        tolerance = 1e-12*np.maximum(np.maximum(aa_np, bb_np), 1e-15)**2
        if np.any(aa_np < 0) or np.any(bb_np < 0) or np.any(ab_np**2 > aa_np*bb_np+tolerance):
            raise ValueError("Spin sigma matrix must be positive semidefinite.")
    floor2 = jnp.asarray(1e-30, dtype=aa.dtype)
    a = jnp.sqrt(jnp.where(aa >= 0., aa, 0.) + floor2)
    b_parallel = ab/a
    residual = bb-b_parallel**2
    b_perpendicular = jnp.sqrt(jnp.where(residual >= 0., residual, 0.) + floor2)
    zero = jnp.zeros_like(a)
    return jnp.stack([a, zero, zero], axis=-1), jnp.stack([b_parallel, b_perpendicular, zero], axis=-1)
