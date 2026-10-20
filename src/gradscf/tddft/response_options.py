from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from numbers import Integral, Real
from typing import Any, Literal, NamedTuple

import numpy as np


TwoElectronMode = Literal["auto", "direct", "df", "ris"]
RISFit = Literal["s", "sp", "spd"]


@dataclass(frozen=True)
class ResponseKernelOptions:
    """Options controlling the two-electron part of restricted TD response."""

    two_electron_mode: TwoElectronMode = "auto"
    ris_theta: float = 0.2
    ris_j_fit: RISFit = "sp"
    ris_k_fit: RISFit = "s"
    ris_aux_chunk_size: int = 256
    max_memory: float = 2000.0
    integral_cache: Literal["auto", "none"] = "auto"
    direct_tol: float = 0.0
    direct_workers: int = 8
    grid_block_size: int = 2048


_OPTION_FIELDS = {field.name for field in fields(ResponseKernelOptions)}
_TWO_ELECTRON_MODES = {"auto", "direct", "df", "ris"}
_RIS_FITS = {"s", "sp", "spd"}


def normalize_response_kernel_options(
    options: ResponseKernelOptions | dict[str, Any] | None,
) -> ResponseKernelOptions:
    if options is None:
        normalized = ResponseKernelOptions()
    elif isinstance(options, ResponseKernelOptions):
        normalized = options
    elif isinstance(options, dict):
        unknown = sorted(set(options) - _OPTION_FIELDS)
        if unknown:
            raise ValueError(f"Unknown response kernel option(s): {', '.join(unknown)}.")
        normalized = ResponseKernelOptions(**options)
    else:
        raise TypeError(
            "response_kernel_options must be None, a ResponseKernelOptions instance, "
            "or a mapping."
        )

    mode = str(normalized.two_electron_mode).lower()
    integral_cache = str(normalized.integral_cache).lower()
    if integral_cache not in {"auto", "none"}:
        raise ValueError("integral_cache must be 'auto' or 'none'.")
    if (isinstance(normalized.direct_tol, bool) or not isinstance(normalized.direct_tol, Real)
            or not isfinite(normalized.direct_tol) or normalized.direct_tol < 0):
        raise ValueError("direct_tol must be finite and nonnegative.")
    for name in ("direct_workers", "grid_block_size"):
        value = getattr(normalized, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if mode not in _TWO_ELECTRON_MODES:
        raise ValueError(
            "two_electron_mode must be one of {'auto', 'direct', 'df', 'ris'}, "
            f"got {normalized.two_electron_mode!r}."
        )
    j_fit = str(normalized.ris_j_fit).lower()
    k_fit = str(normalized.ris_k_fit).lower()
    if j_fit not in _RIS_FITS:
        raise ValueError(f"ris_j_fit must be one of {{'s', 'sp', 'spd'}}, got {j_fit!r}.")
    if k_fit not in _RIS_FITS:
        raise ValueError(f"ris_k_fit must be one of {{'s', 'sp', 'spd'}}, got {k_fit!r}.")
    if float(normalized.ris_theta) <= 0.0:
        raise ValueError("ris_theta must be positive.")
    if int(normalized.ris_aux_chunk_size) <= 0:
        raise ValueError("ris_aux_chunk_size must be positive.")
    if (isinstance(normalized.max_memory, bool) or not isinstance(normalized.max_memory, Real)
            or not isfinite(normalized.max_memory) or normalized.max_memory <= 0):
        raise ValueError("max_memory must be a finite positive number of MB.")

    return ResponseKernelOptions(
        two_electron_mode=mode,  # type: ignore[arg-type]
        ris_theta=float(normalized.ris_theta),
        ris_j_fit=j_fit,  # type: ignore[arg-type]
        ris_k_fit=k_fit,  # type: ignore[arg-type]
        ris_aux_chunk_size=int(normalized.ris_aux_chunk_size),
        max_memory=float(normalized.max_memory),
        integral_cache=integral_cache,
        direct_tol=float(normalized.direct_tol),
        direct_workers=int(normalized.direct_workers),
        grid_block_size=int(normalized.grid_block_size),
    )


class _ResponseMemoryPlan(NamedTuple):
    max_subspace: int
    batch_size: int
    eri_cache_bytes: int
    grid_block_size: int


def _native_direct_workspace_bytes(provider, workers, cutoff):
    """Bound one native call's batch-independent optimizer and shell arrays.

    The pinned native ABI uses float64, int32 and 64-bit pointers. Return
    None for dimension-only placeholders without a complete shell topology.
    """
    topology = provider.plan.topology
    names = ("angular_momenta", "primitive_counts", "contraction_counts",
             "nuclear_charges", "cart")
    if any(not hasattr(topology, name) for name in names):
        return None
    angular = topology.angular_momenta
    primitive = topology.primitive_counts
    contractions = topology.contraction_counts
    shells = len(angular)
    workers = min(workers, shells)
    total_primitive = sum(primitive)
    total_coefficients = sum(p*c for p, c in zip(primitive, contractions))
    max_l, max_p = max(angular), max(primitive)
    cart_dims = tuple((l+1)*(l+2)//2 for l in angular)
    max_cart = max(f*c for f, c in zip(cart_dims, contractions))
    max_output = max(c*(f if topology.cart else 2*l+1)
                     for l, f, c in zip(angular, cart_dims, contractions))
    nf, nc = max(cart_dims)**4, max_cart**4
    g_size = (2*max_l+1)**3 * (max_l+1)**2
    pair_scratch = (10*max_p**2 + 4*max(p*c for p, c in zip(primitive, contractions))
                    + 8*max_p + 3*nf)
    cache = max(6*g_size + nf + 3*nc + pair_scratch, nc + 4*nf)
    # Retained vector capacities can exceed the current shell's logical size.
    shapes = len(set(zip(angular, primitive, contractions)))
    scratch = 16*(max_output**4 + cache) + 128*shapes**4
    allowed_l = min(max_l, 6)
    cumulative_cart = (allowed_l+1)*(allowed_l+2)*(allowed_l+3)//6
    optimizer = (48 + 12*cumulative_cart**4 + 8*(max_l+1)*16**3
                 + 24*shells + 12*total_primitive + 4*total_coefficients)
    if total_primitive <= 2048:
        optimizer += 40*total_primitive**2 + 8*shells**2
    atoms = len(topology.nuclear_charges) + shells
    tables = (4*(6*atoms + 8*shells + shells+1)
              + 8*(20 + 3*atoms + total_primitive + total_coefficients))
    bounds = 8*shells**2 if cutoff > 0 else 0
    return optimizer + bounds + (workers+1)*tables + workers*scratch


def plan_response_memory(nao, dim, nroots, dtype, max_memory,
                         max_subspace=None, rpa=False, *, direct_basis=None,
                         integral_cache="auto", direct_workers=8, direct_tol=0.,
                         grid_block_size=0, spin_channels=1):
    """Bound primary solver/AO work arrays; max_memory is decimal MB.

    The estimate reserves common AO state, output vectors, QR/Ritz work and
    one microbatch of densities and J/K outputs. It is not an RSS bound on
    XLA compilation, external functional internals or allocator caches.
    """
    if (isinstance(max_memory, bool) or not isinstance(max_memory, Real)
            or not isfinite(max_memory) or max_memory <= 0):
        raise ValueError("max_memory must be a finite positive number of MB.")
    if (isinstance(direct_tol, bool) or not isinstance(direct_tol, Real)
            or not isfinite(direct_tol) or direct_tol < 0):
        raise ValueError("direct_tol must be finite and nonnegative.")
    if (isinstance(grid_block_size, bool) or not isinstance(grid_block_size, Integral)
            or grid_block_size < 0):
        raise ValueError("grid_block_size must be a nonnegative integer.")
    if (isinstance(spin_channels, bool) or not isinstance(spin_channels, Integral)
            or spin_channels not in (1, 2)):
        raise ValueError("spin_channels must be 1 or 2.")
    if min(nao, dim, nroots) < 1 or nroots > dim:
        raise ValueError("Invalid response dimensions or requested root count.")
    if max_subspace is not None and (isinstance(max_subspace, bool)
            or not isinstance(max_subspace, Integral)):
        raise ValueError("max_subspace must be an integer.")
    # Public eigensolvers promote float32 to float64 when x64 is enabled.
    # Eight bytes is conservative for the supported real response paths.
    itemsize = max(8, np.dtype(dtype).itemsize)
    budget = int(max_memory * 1_000_000)
    fixed = itemsize * (8 * spin_channels * nao**2 + 4 * dim * nroots)
    # The density transpose of native J/K concatenates the J/K cotangents.
    from ..integrals.backends.native.jk import NativeDirectBasis
    native = type(direct_basis) is NativeDirectBasis
    if isinstance(direct_workers, bool) or not isinstance(direct_workers, Integral) or direct_workers < 1:
        raise ValueError("direct_workers must be a positive integer.")
    native_fixed = 0
    legacy_native = native
    if native:
        workspace = _native_direct_workspace_bytes(direct_basis, direct_workers, direct_tol)
        if workspace is not None:
            # Primal/JVP/VJP calls can overlap. Each owns one shared optimizer,
            # shell scratch and tables; output rows need no worker J/K copies.
            native_fixed = 3*workspace
            legacy_native = False
    # Keep the old conservative worker estimate for dimension-only providers.
    ao_matrices = 12 + (4*direct_workers if legacy_native else 0)
    per_vector = itemsize * (ao_matrices * spin_channels * nao**2 + (8 if rpa else 4) * dim)
    minimum = min(dim, nroots + (2 if rpa else 3))
    default = max(8 * nroots, 128) if rpa else max(40, 4 * (nroots + 2))
    width = min(dim, default if max_subspace is None else int(max_subspace))
    if width < minimum:
        raise ValueError("max_subspace must retain requested and boundary roots plus expansion slots.")
    def basis_bytes(columns):
        return itemsize * ((10 if rpa else 6) * dim * columns + 8 * columns**2)
    # Reserve the same action share for builder and solver, independently of
    # the requested roots. The remaining budget belongs to the Ritz basis.
    # Block-local AO/MO factors and both sides of the grid projection, with
    # room for the maximum 32-vector batch and its transpose/recomputation.
    grid_point_bytes = itemsize * spin_channels * (96*nao + 384) if grid_block_size else 0
    action_budget = max(native_fixed + per_vector + grid_point_bytes, budget // 4)
    grid_budget = (min(action_budget-native_fixed-per_vector,
                       max(grid_point_bytes, action_budget // 4))
                   if grid_block_size else 0)
    integral_budget = action_budget - native_fixed - grid_budget
    cache_bytes = 0
    if integral_cache not in {"auto", "none"}:
        raise ValueError("integral_cache must be 'auto' or 'none'.")
    if native and integral_cache == "auto" and direct_tol == 0:
        import jax

        # Projected providers can have a much larger primitive AO dimension;
        # CUDA uses a different packed workspace. Keep both shell-direct.
        if jax.default_backend() == "cpu":
            candidate = direct_basis.eri_cache_nbytes
            # Eight workers retain two J/K accumulators. A mixed zero-basis /
            # density JVP can have three live calls: 3*2*8 extra AO matrices.
            cached_vector = itemsize * (60 * spin_channels * nao**2 + (8 if rpa else 4) * dim)
            # Reserve a second packed buffer for zero-basis JVP/recomputation.
            if 2*candidate + cached_vector <= integral_budget:
                cache_bytes, per_vector = candidate, cached_vector
    required = fixed + action_budget + basis_bytes(minimum)
    if required > budget:
        raise ValueError(f"Response memory budget is {budget} bytes; at least {required} bytes are required.")
    available = budget - fixed - action_budget
    if max_subspace is not None and basis_bytes(width) > available:
        raise ValueError("Requested max_subspace exceeds the response memory budget.")
    while basis_bytes(width) > available:
        width -= 1
    return _ResponseMemoryPlan(
        width, min(32, (integral_budget-2*cache_bytes) // per_vector), cache_bytes,
        min(grid_block_size, grid_budget // grid_point_bytes) if grid_point_bytes else 0)
