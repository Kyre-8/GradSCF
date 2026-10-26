from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.lax import Precision

from ...dft.xc import RestrictedFeatureBundle

_DM21_BETA = 1.0 / 1024.0


def _bounded_ratio(numerator: Any, denominator: Any) -> jnp.ndarray:
    ratio = jnp.asarray(numerator) / jnp.maximum(jnp.asarray(denominator), 1e-30)
    ratio = jnp.nan_to_num(ratio, nan=0.0, posinf=1e30, neginf=0.0)
    beta = jnp.asarray(_DM21_BETA, dtype=ratio.dtype)
    return beta * ratio / (1.0 + beta * ratio)


@jax.custom_jvp
def _safe_sqrt_nonnegative(value: Any) -> jnp.ndarray:
    return jnp.sqrt(jnp.maximum(jnp.asarray(value), 0.0))


@_safe_sqrt_nonnegative.defjvp
def _safe_sqrt_nonnegative_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    root = _safe_sqrt_nonnegative(value)
    denominator = jnp.where(root > 0.0, 2.0 * root, 1.0)
    return root, jnp.where(jnp.asarray(value) > 0.0, tangent / denominator, 0.0)


def canonical_input_features(
    features: RestrictedFeatureBundle,
    hfx_a: Any,
    hfx_b: Any,
    *,
    density_floor: float = 1e-12,
) -> jnp.ndarray:
    rho_a = jnp.maximum(features.rho_a, density_floor)
    rho_b = jnp.maximum(features.rho_b, density_floor)
    rho = jnp.maximum(features.rho, density_floor)
    tau_a = jnp.maximum(features.tau_a, 0.0)
    tau_b = jnp.maximum(features.tau_b, 0.0)
    norm_grad_a = jnp.maximum(features.sigma_aa, 0.0)
    norm_grad_b = jnp.maximum(features.sigma_bb, 0.0)
    norm_grad = jnp.maximum(features.sigma, 0.0)
    tau_prefactor = (3.0 / 5.0) * (6.0 * jnp.pi**2) ** (2.0 / 3.0)
    reduced_grad = _bounded_ratio(
        _safe_sqrt_nonnegative(norm_grad), rho ** (4.0 / 3.0)
    )
    reduced_grad_a = _bounded_ratio(
        _safe_sqrt_nonnegative(norm_grad_a), rho_a ** (4.0 / 3.0)
    )
    reduced_grad_b = _bounded_ratio(
        _safe_sqrt_nonnegative(norm_grad_b), rho_b ** (4.0 / 3.0)
    )
    reduced_tau_a = _bounded_ratio(tau_a, tau_prefactor * rho_a ** (5.0 / 3.0))
    reduced_tau_b = _bounded_ratio(tau_b, tau_prefactor * rho_b ** (5.0 / 3.0))
    hfx_a = jnp.asarray(hfx_a)
    hfx_b = jnp.asarray(hfx_b)
    if hfx_a.ndim == rho_a.ndim:
        hfx_a = hfx_a[..., None]
    if hfx_b.ndim == rho_b.ndim:
        hfx_b = hfx_b[..., None]
    if hfx_a.shape[:-1] != rho_a.shape or hfx_b.shape[:-1] != rho_b.shape:
        raise ValueError(
            "Local HFX features must broadcast to the grid shape "
            f"(rho={rho_a.shape}, hfx_a={hfx_a.shape}, hfx_b={hfx_b.shape})."
        )
    leading = jnp.stack(
        [
            rho_a,
            rho_b,
            reduced_grad,
            reduced_grad_a,
            reduced_grad_b,
            reduced_tau_a,
            reduced_tau_b,
        ],
        axis=-1,
    )
    return jnp.concatenate([leading, hfx_a, hfx_b], axis=-1)


def enhanced_input_features(
    features: RestrictedFeatureBundle,
    semilocal_descriptor: Any,
    *,
    density_floor: float = 1e-12,
) -> jnp.ndarray:
    rho_a = jnp.maximum(features.rho_a, density_floor)
    rho_b = jnp.maximum(features.rho_b, density_floor)
    rho = jnp.maximum(features.rho, density_floor)
    sigma = jnp.maximum(features.sigma, 0.0)
    tau = jnp.maximum(features.tau_a + features.tau_b, 0.0)
    return jnp.stack(
        [
            rho_a,
            rho_b,
            rho,
            jnp.log1p(rho),
            jnp.sqrt(rho),
            features.sigma_aa,
            features.sigma_ab,
            features.sigma_bb,
            sigma,
            features.tau_a,
            features.tau_b,
            tau,
            semilocal_descriptor,
        ],
        axis=-1,
    )


def resolve_canonical_hfx_feature_channels(
    molecule: Any | None,
    features: RestrictedFeatureBundle,
    *,
    hf_energy_density: Any | None = None,
    hf_spin_energy_density: tuple[Any, Any] | None = None,
    hfx_channels: int = 2,
    strict_feature_alignment: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    target_channels = max(int(hfx_channels), 1)
    if hf_spin_energy_density is not None:
        hfx_a = jnp.asarray(hf_spin_energy_density[0])
        hfx_b = jnp.asarray(hf_spin_energy_density[1])
        if hfx_a.ndim == features.rho.ndim:
            hfx_a = hfx_a[..., None]
        if hfx_b.ndim == features.rho.ndim:
            hfx_b = hfx_b[..., None]
        if hfx_a.shape[-1] == 1 and target_channels > 1:
            hfx_a = jnp.repeat(hfx_a, target_channels, axis=-1)
        if hfx_b.shape[-1] == 1 and target_channels > 1:
            hfx_b = jnp.repeat(hfx_b, target_channels, axis=-1)
        return hfx_a, hfx_b

    cached = getattr(molecule, "hfx_local", None) if molecule is not None else None
    if cached is not None:
        cached = jnp.asarray(cached)
        if cached.ndim == 3 and cached.shape[0] == 2:
            if strict_feature_alignment and cached.shape[-1] != target_channels:
                raise ValueError(
                    "molecule.hfx_local omega-channel count must match hfx_channels "
                    f"(got {cached.shape[-1]} vs {target_channels})."
                )
            return cached[0], cached[1]
        raise ValueError(
            "molecule.hfx_local must have shape (2, ngrids, n_omega), "
            f"got {cached.shape}."
        )

    if strict_feature_alignment and molecule is not None:
        raise ValueError(
            "canonical input mode requires molecule.hfx_local with shape "
            "(2, ngrids, n_omega), or explicit hf_spin_energy_density channels. "
            "Build the reference with compute_local_hfx_features=True "
            "(typically omega values 0.0 and 0.4)."
        )

    hf_total = (
        jnp.zeros_like(features.rho)
        if hf_energy_density is None
        else jnp.asarray(hf_energy_density)
    )
    local_hfx = jnp.repeat(hf_total[..., None], target_channels, axis=-1)
    return local_hfx, local_hfx


def build_coefficient_inputs(
    features: RestrictedFeatureBundle,
    semilocal_energy_density: Any,
    hf_energy_density: Any,
    *,
    input_feature_mode: str,
    hf_input_mode: str,
    include_pt2_channel: bool,
    density_floor: float,
    hfx_channels: int,
    strict_feature_alignment: bool,
    include_hfx_channel: bool = True,
    pt2_energy_density: Any | None = None,
    molecule: Any | None = None,
    hf_spin_energy_density: tuple[Any, Any] | None = None,
    semilocal_descriptor: Any | None = None,
) -> jnp.ndarray:
    pt2_total = (
        jnp.zeros_like(features.rho)
        if pt2_energy_density is None
        else jnp.asarray(pt2_energy_density)
    )
    if input_feature_mode == "canonical":
        if include_hfx_channel:
            hfx_a, hfx_b = resolve_canonical_hfx_feature_channels(
                molecule,
                features,
                hf_energy_density=hf_energy_density,
                hf_spin_energy_density=hf_spin_energy_density,
                hfx_channels=hfx_channels,
                strict_feature_alignment=strict_feature_alignment,
            )
        else:
            target_channels = max(int(hfx_channels), 1)
            zeros = jnp.zeros_like(features.rho)[..., None]
            hfx_a = jnp.repeat(zeros, target_channels, axis=-1)
            hfx_b = hfx_a
        base = canonical_input_features(
            features,
            hfx_a,
            hfx_b,
            density_floor=density_floor,
        )
        if not include_pt2_channel:
            return base
        return jnp.concatenate([base, pt2_total[..., None]], axis=-1)

    if input_feature_mode != "enhanced":
        raise ValueError(
            f"Unsupported input_feature_mode={input_feature_mode!r}. "
            "Expected 'enhanced' or 'canonical'."
        )

    if semilocal_descriptor is None:
        density = jnp.maximum(jnp.asarray(features.rho), density_floor)
        semilocal_descriptor = jnp.nan_to_num(
            jnp.asarray(semilocal_energy_density) / density,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    base = enhanced_input_features(
        features,
        semilocal_descriptor,
        density_floor=density_floor,
    )
    extras = []
    if include_hfx_channel:
        hf_total = jnp.asarray(hf_energy_density)
        if hf_input_mode == "total_only":
            extras = [hf_total[..., None]]
        elif hf_input_mode == "spin_resolved":
            if hf_spin_energy_density is None:
                hf_a = hf_total
                hf_b = hf_total
            else:
                hf_a, hf_b = hf_spin_energy_density
            extras = [
                hf_total[..., None],
                jnp.asarray(hf_a)[..., None],
                jnp.asarray(hf_b)[..., None],
            ]
        else:
            raise ValueError(
                f"Unsupported hf_input_mode={hf_input_mode!r}. "
                "Expected 'total_only' or 'spin_resolved'."
            )
    if include_pt2_channel:
        extras.append(pt2_total[..., None])
    return jnp.concatenate([base, *extras], axis=-1)


def assemble_basis_channels(
    semilocal_local_channels: Any,
    *,
    hf_projected: Any,
    include_pt2_channel: bool,
    include_hfx_channel: bool = True,
    pt2_projected: Any | None = None,
) -> jnp.ndarray:
    channels = [jnp.asarray(semilocal_local_channels)]
    if include_pt2_channel:
        if pt2_projected is None:
            raise ValueError("pt2_projected must be provided when include_pt2_channel=True.")
        channels.append(jnp.asarray(pt2_projected)[..., None])
    if include_hfx_channel:
        channels.append(jnp.asarray(hf_projected)[..., None])
    return jnp.concatenate(channels, axis=-1)

__all__ = ['canonical_input_features', 'enhanced_input_features', 'resolve_canonical_hfx_feature_channels', 'build_coefficient_inputs', 'assemble_basis_channels']
