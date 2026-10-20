from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..tools.features import grid_features_with_spin_gradients_for_molecule
from ..dft.xc import parse_recipe
from ..dft.xc.evaluate import grid_hessian
from .response import (
    _project_grid_response_to_restricted_transition,
    _project_restricted_transition_to_grid,
    _restricted_response_factors,
)


SpinGridHVP = Callable[[Any, Array, Array], tuple[Array, Array]]


def _validate_spin_tangent(
    tangent: Array,
    *,
    nfeatures: int,
    ngrids: int,
    label: str,
) -> Array:
    array = jnp.asarray(tangent)
    expected_tail = (int(nfeatures), int(ngrids))
    if array.ndim != 3 or array.shape[1:] != expected_tail:
        raise ValueError(
            f"{label} must have shape (batch, {nfeatures}, {ngrids}), "
            f"got {array.shape}."
        )
    return array


def pack_spin_grid_tangents(
    tangent_a: Array,
    tangent_b: Array,
    *,
    ngrids: int,
    dtype: Any,
    trailing_zeros: int = 0,
) -> Array:
    nfeatures = jnp.asarray(tangent_a).shape[1]
    if nfeatures not in {4, 5}:
        raise ValueError("Spin-gradient tangents require four GGA or five MGGA features.")
    tangent_a = _validate_spin_tangent(
        jnp.asarray(tangent_a, dtype=dtype),
        nfeatures=nfeatures,
        ngrids=ngrids,
        label="tangent_a",
    )
    tangent_b = _validate_spin_tangent(
        jnp.asarray(tangent_b, dtype=dtype),
        nfeatures=nfeatures,
        ngrids=ngrids,
        label="tangent_b",
    )
    if tangent_a.shape[0] != tangent_b.shape[0]:
        raise ValueError("Alpha and beta spin-grid tangents must share a batch size.")
    channels = [
        tangent_a[:, 0:1],
        tangent_b[:, 0:1],
        tangent_a[:, 1:4],
        tangent_b[:, 1:4],
    ]
    if nfeatures == 5:
        channels.extend([tangent_a[:, 4:5], tangent_b[:, 4:5]])
    if trailing_zeros:
        channels.append(
            jnp.zeros(
                (tangent_a.shape[0], int(trailing_zeros), int(ngrids)),
                dtype=dtype,
            )
        )
    return jnp.concatenate(channels, axis=1).transpose(0, 2, 1)


def unpack_spin_grid_response(response: Array, *, include_tau: bool = False) -> tuple[Array, Array]:
    response = jnp.asarray(response).transpose(0, 2, 1)
    if response.shape[1] < 8:
        raise ValueError(
            "Spin-grid response must contain rho_a, rho_b, and both gradient blocks."
        )
    alpha = [response[:, 0:1], response[:, 2:5]]
    beta = [response[:, 1:2], response[:, 5:8]]
    if include_tau:
        if response.shape[1] != 10:
            raise ValueError("MGGA spin response requires both kinetic-density channels.")
        alpha.append(response[:, 8:9])
        beta.append(response[:, 9:10])
    return jnp.concatenate(alpha, axis=1), jnp.concatenate(beta, axis=1)


def build_spin_transition_factors(
    molecule: Any,
    orbo: Array,
    orbv: Array,
    *,
    feature_kind: str,
    dtype: Any,
    grid_start: Any = 0,
    grid_size: int | None = None,
):
    return _restricted_response_factors(
        molecule,
        orbo,
        orbv,
        feature_kind=str(feature_kind).upper(),
        dtype=dtype,
        grid_start=grid_start,
        grid_size=grid_size,
    )


def project_spin_transition_to_grid(factors: Any, values: Array) -> Array:
    return _project_restricted_transition_to_grid(factors, values)


def project_grid_response_to_spin_transition(factors: Any, values: Array) -> Array:
    return _project_grid_response_to_restricted_transition(factors, values)


def _build_blocked_spin_response_action(
    molecule: Any,
    orbo_a: Array,
    orbv_a: Array,
    orbo_b: Array,
    orbv_b: Array,
    block_hvp: Callable[..., tuple[Array, Array]],
    *,
    feature_kind: str,
    dtype: Any,
    block_size: int,
) -> Callable[[Array, Array], tuple[Array, Array]]:
    """Reduce local spin-field responses without retaining grid-by-OV arrays."""
    if isinstance(block_size, bool) or not isinstance(block_size, Integral) or block_size < 1:
        raise ValueError("block_size must be a positive integer.")
    weights = jnp.asarray(molecule.grid.weights, dtype=dtype)
    ngrids = int(weights.shape[0])
    width = min(int(block_size), max(1, ngrids))
    full_blocks, tail = divmod(ngrids, width)

    def contribution(start, size, alpha, beta):
        factors_a = build_spin_transition_factors(
            molecule, orbo_a, orbv_a, feature_kind=feature_kind, dtype=dtype,
            grid_start=start, grid_size=size,
        )
        factors_b = build_spin_transition_factors(
            molecule, orbo_b, orbv_b, feature_kind=feature_kind, dtype=dtype,
            grid_start=start, grid_size=size,
        )
        tangent_a = project_spin_transition_to_grid(factors_a, alpha)
        tangent_b = project_spin_transition_to_grid(factors_b, beta)
        response_a, response_b = block_hvp(start, size, tangent_a, tangent_b)
        block_weights = jax.lax.dynamic_slice_in_dim(weights, start, size)
        return (
            project_grid_response_to_spin_transition(
                factors_a, jnp.asarray(response_a, dtype=dtype)*block_weights[None, None, :]),
            project_grid_response_to_spin_transition(
                factors_b, jnp.asarray(response_b, dtype=dtype)*block_weights[None, None, :]),
        )

    rematerialized = jax.checkpoint(contribution, static_argnums=(1,))

    def action(alpha: Array, beta: Array) -> tuple[Array, Array]:
        alpha, beta = jnp.asarray(alpha, dtype=dtype), jnp.asarray(beta, dtype=dtype)

        def add_block(index, total):
            da, db = rematerialized(index*width, width, alpha, beta)
            return total[0] + da, total[1] + db

        out_a, out_b = jax.lax.fori_loop(
            0, full_blocks, add_block, (jnp.zeros_like(alpha), jnp.zeros_like(beta)))
        if tail:
            da, db = rematerialized(full_blocks*width, tail, alpha, beta)
            out_a, out_b = out_a + da, out_b + db
        return out_a, out_b

    return action


def build_unrestricted_semilocal_response_action(
    molecule: Any,
    orbo_a: Array,
    orbv_a: Array,
    orbo_b: Array,
    orbv_b: Array,
    response_hvp: SpinGridHVP,
    *,
    feature_kind: str,
    dtype: Any,
    block_size: int | None = None,
) -> Callable[[Array, Array], tuple[Array, Array]]:
    kind = str(feature_kind).upper()
    if kind not in {"LDA", "GGA", "MGGA"}:
        raise NotImplementedError(
            "Unrestricted semilocal response projection supports LDA/GGA/tau-dependent MGGA."
        )
    if (block_size is not None
            and type(getattr(response_hvp, "__self__", None)) is UnrestrictedSemilocalResponseFunctional):
        def block_hvp(start, size, tangent_a, tangent_b):
            def on_grid(value, axis=0):
                return (None if value is None else jax.lax.dynamic_slice_in_dim(
                    jnp.asarray(value), start, size, axis=axis))

            # Only the known local functional sees this view. Opaque callbacks
            # retain the original molecule and both complete spin-grid fields.
            view = SimpleNamespace(
                ao=on_grid(molecule.ao),
                ao_deriv1=on_grid(getattr(molecule, "ao_deriv1", None), axis=1),
                rdm1=molecule.rdm1,
                grid=SimpleNamespace(weights=on_grid(molecule.grid.weights),
                                     coords=on_grid(getattr(molecule.grid, "coords", None))),
                xc_density_floor=getattr(molecule, "xc_density_floor", 1e-12),
                dispersion_energy=getattr(molecule, "dispersion_energy", None),
            )
            return response_hvp(view, tangent_a, tangent_b)

        return _build_blocked_spin_response_action(
            molecule, orbo_a, orbv_a, orbo_b, orbv_b, block_hvp,
            feature_kind=kind, dtype=dtype, block_size=block_size,
        )
    factors_a = build_spin_transition_factors(
        molecule,
        orbo_a,
        orbv_a,
        feature_kind=kind,
        dtype=dtype,
    )
    factors_b = build_spin_transition_factors(
        molecule,
        orbo_b,
        orbv_b,
        feature_kind=kind,
        dtype=dtype,
    )
    weights = jnp.asarray(molecule.grid.weights, dtype=dtype)

    def action(alpha: Array, beta: Array) -> tuple[Array, Array]:
        tangent_a = project_spin_transition_to_grid(factors_a, alpha)
        tangent_b = project_spin_transition_to_grid(factors_b, beta)
        response_a, response_b = response_hvp(molecule, tangent_a, tangent_b)
        weighted_a = jnp.asarray(response_a, dtype=dtype) * weights[None, None, :]
        weighted_b = jnp.asarray(response_b, dtype=dtype) * weights[None, None, :]
        return (
            project_grid_response_to_spin_transition(factors_a, weighted_a),
            project_grid_response_to_spin_transition(factors_b, weighted_b),
        )

    return action


@dataclass(frozen=True)
class UnrestrictedSemilocalResponseFunctional:
    """Spin-polarized local-field Hessian with complete tau-dependent MGGA response."""

    xc_spec: str

    def __post_init__(self) -> None:
        spec = str(self.xc_spec).lower()
        recipe = parse_recipe(spec)
        kind = recipe.kind
        if kind not in {"HF", "LDA", "GGA", "MGGA"} or recipe.needs_laplacian:
            raise NotImplementedError(
                "Unrestricted response supports HF/LDA/GGA and tau-dependent MGGA."
            )
        object.__setattr__(self, "xc_spec", spec)
        coefficients = recipe.to_range_separated_hybrid_coefficients()
        object.__setattr__(self, "range_separated_coefficients", coefficients)
        object.__setattr__(self, "exact_exchange_fraction", float(
            recipe.exact_exchange.full_range if coefficients is None else coefficients[2]))
        object.__setattr__(self, "response_feature_kind", "LDA" if kind == "HF" else kind)

    def spin_grid_response_hvp(
        self,
        molecule: Any,
        tangent_a: Array,
        tangent_b: Array,
    ) -> tuple[Array, Array]:
        recipe = parse_recipe(self.xc_spec)
        recipe.require_complete_support(provided=() if getattr(molecule, "dispersion_energy", None) is None
                                        else ("dispersion",))
        if recipe.kind == "HF":
            # HF has no semilocal XC Hessian. Nonlocal exchange is assembled
            # separately using exact_exchange_fraction in the response action.
            return jnp.zeros_like(tangent_a), jnp.zeros_like(tangent_b)
        features, grad_a, grad_b = grid_features_with_spin_gradients_for_molecule(
            molecule
        )
        ngrids = int(features.rho.shape[0])
        nfeatures = {"LDA": 1, "GGA": 4, "MGGA": 5}[self.response_feature_kind]
        tangent_a = _validate_spin_tangent(
            tangent_a,
            nfeatures=nfeatures,
            ngrids=ngrids,
            label="tangent_a",
        )
        tangent_b = _validate_spin_tangent(
            tangent_b,
            nfeatures=nfeatures,
            ngrids=ngrids,
            label="tangent_b",
        )
        if tangent_a.shape[0] != tangent_b.shape[0]:
            raise ValueError("Alpha and beta spin-grid tangents must share a batch size.")

        if self.response_feature_kind == "LDA":
            base = jnp.stack([features.rho_a, features.rho_b], axis=-1)
            tangent = jnp.stack([tangent_a[:, 0], tangent_b[:, 0]], axis=-1)
        else:
            base = jnp.concatenate(
                [
                    features.rho_a[:, None],
                    features.rho_b[:, None],
                    grad_a,
                    grad_b,
                ],
                axis=-1,
            )
            if self.response_feature_kind == "MGGA":
                base = jnp.concatenate([base, features.tau_a[:, None], features.tau_b[:, None]], axis=-1)
            tangent = pack_spin_grid_tangents(
                tangent_a,
                tangent_b,
                ngrids=ngrids,
                dtype=base.dtype,
            )

        active = features.rho > getattr(molecule, "xc_density_floor", 1e-12)
        base = jnp.where(active[:, None], base, jnp.ones_like(base))
        hessian = grid_hessian(self.xc_spec, self.response_feature_kind)(base)
        hessian = jnp.where(active[:, None, None], hessian, 0.)
        response = jnp.einsum("gij,ngj->ngi", hessian, tangent)
        if self.response_feature_kind == "LDA":
            response = response.transpose(0, 2, 1)
            return response[:, 0:1], response[:, 1:2]
        return unpack_spin_grid_response(response, include_tau=self.response_feature_kind == "MGGA")


__all__ = [
    "SpinGridHVP",
    "UnrestrictedSemilocalResponseFunctional",
    "build_spin_transition_factors",
    "build_unrestricted_semilocal_response_action",
    "pack_spin_grid_tangents",
    "project_grid_response_to_spin_transition",
    "project_spin_transition_to_grid",
    "unpack_spin_grid_response",
]
