from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ..data.molecule import MoleculeSpec


from ..solvers.eigen.spectral import (
    regularized_eigh as _safe_symmetric_eigh,
    inverse_sqrt as _orthogonalizer,
    _spectral_orthogonalizer,
)


def _contains_jax_tracer(value: Any) -> bool:
    if isinstance(value, jax.core.Tracer):
        return True
    if isinstance(value, MoleculeSpec):
        return _contains_jax_tracer((value.coords_bohr, value.charges))
    if isinstance(value, dict):
        return any(_contains_jax_tracer(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_jax_tracer(item) for item in value)
    leaves = jax.tree_util.tree_leaves(value)
    return any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)


def _host_float_unless_traced(value: Any) -> Any:
    return value if _contains_jax_tracer(value) else float(value)


def _validate_density_matrix(
    density: Array | None,
    *,
    nao: int,
    dtype: Any,
    label: str,
    method: str,
) -> Array | None:
    if density is None:
        return None
    dm = jnp.asarray(density, dtype=dtype)
    if dm.ndim != 2 or tuple(dm.shape) != (nao, nao):
        raise ValueError(f"{label} must be a square ({nao}, {nao}) matrix for {method}.")
    return 0.5 * (dm + dm.T)


def _diagonalize_fock(
    fock: Array,
    x: Array,
    eigenvalue_jitter: float = 0.0,
) -> tuple[Array, Array]:
    f_ortho = x.T @ fock @ x
    f_ortho = 0.5 * (f_ortho + f_ortho.T)
    if eigenvalue_jitter != 0.0:
        shift = jnp.arange(f_ortho.shape[0], dtype=f_ortho.dtype) * eigenvalue_jitter
        f_ortho = f_ortho + jnp.diag(shift)
    mo_energy, coeff_ortho = _safe_symmetric_eigh(f_ortho)
    mo_coeff = x @ coeff_ortho
    return mo_energy, mo_coeff


def _build_density_from_occ(mo_coeff: Array, mo_occ: Array) -> Array:
    occ = jnp.asarray(mo_occ, dtype=mo_coeff.dtype)
    return jnp.einsum("pi,i,qi->pq", mo_coeff, occ, mo_coeff, precision=Precision.HIGHEST)

def _unrestricted_spin_electron_counts(
    total_electrons: int,
    spin: int,
) -> tuple[int, int]:
    total_electrons = int(total_electrons)
    spin = int(spin)
    if total_electrons <= 0:
        raise ValueError("Unrestricted references require a positive electron count.")
    if abs(spin) > total_electrons:
        raise ValueError(
            f"Spin={spin} is incompatible with total_electrons={total_electrons}."
        )
    if (total_electrons + spin) % 2 != 0:
        raise ValueError(
            f"Spin={spin} is incompatible with total_electrons={total_electrons}; "
            "N + spin must be even."
        )
    nalpha = (total_electrons + spin) // 2
    nbeta = total_electrons - nalpha
    if nalpha < 0 or nbeta < 0:
        raise ValueError(
            f"Failed to resolve spin electron counts for total_electrons={total_electrons}, spin={spin}."
        )
    return int(nalpha), int(nbeta)
