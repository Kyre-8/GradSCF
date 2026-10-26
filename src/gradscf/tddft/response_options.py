from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from numbers import Integral, Real
from typing import Any, Literal, NamedTuple

import numpy as np

from .._memory import DEFAULT_MAX_MEMORY_MB, array_storage_bytes, validate_memory_mb, workspace_bytes
from ..integrals.backends.native.jk import native_jk_workspace_bytes as _native_direct_workspace_bytes


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
    max_memory: float = DEFAULT_MAX_MEMORY_MB
    integral_cache: Literal["auto", "none"] = "auto"
    direct_tol: float = 0.0
    direct_workers: int = 8
    grid_block_size: int = 8192


_OPTION_FIELDS = {field.name for field in fields(ResponseKernelOptions)}
_TWO_ELECTRON_MODES = {"auto", "direct", "df", "ris"}
_RIS_FITS = {"s", "sp", "spd"}


def normalize_response_kernel_options(
    options: ResponseKernelOptions | dict[str, Any] | None,
    *,
    default_max_memory: float = DEFAULT_MAX_MEMORY_MB,
) -> ResponseKernelOptions:
    if options is None:
        normalized = ResponseKernelOptions(max_memory=default_max_memory)
    elif isinstance(options, ResponseKernelOptions):
        normalized = options
    elif isinstance(options, dict):
        unknown = sorted(set(options) - _OPTION_FIELDS)
        if unknown:
            raise ValueError(f"Unknown response kernel option(s): {', '.join(unknown)}.")
        normalized = ResponseKernelOptions(**{"max_memory": default_max_memory, **options})
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
    validate_memory_mb(normalized.max_memory)

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
    # Existing planner share, including fixed arrays; not a measured peak.
    action_bytes: int = 0


def _response_memory_inputs(molecule, resolved_xc, *, spin_channels=1):
    """Count resident arrays and unavoidable complete-grid callback fields.

    Only shapes and metadata are inspected; no array is copied to the host.
    External network activations are not sized here; the runtime reserve
    provides headroom rather than a guarantee for arbitrary model internals.
    """
    import jax

    names = ("ao", "ao_deriv1", "ao_laplacian", "mo_coeff", "mo_occ", "mo_energy",
             "rdm1", "h1e", "overlap_matrix", "fock_matrix", "dipole_integrals",
             "rep_tensor", "eri_pair_matrix", "eri_lr", "df_factors",
             "response_df_factors_j", "response_df_factors_k", "eri_ovov", "eri_ovvo",
             "eri_oovv", "hfx_local", "hfx_fxx", "hfx_nu", "pt2_local",
             "pt2_fock_response", "neural_xc_grid_payload", "direct_basis", "grid_ao")
    arrays = [getattr(molecule, name, None) for name in names]
    grid = getattr(molecule, "grid", None)
    arrays += [getattr(grid, "coords", None), getattr(grid, "weights", None),
               getattr(resolved_xc, "params", None)]
    resident = array_storage_bytes(*jax.tree_util.tree_leaves(arrays))
    weights = getattr(grid, "weights", None)
    ngrids = int(weights.shape[0]) if weights is not None else 0
    dtype = getattr(weights, "dtype", np.dtype(np.float64))
    itemsize = max(8, np.dtype(dtype).itemsize)
    kind = str(getattr(resolved_xc, "response_feature_kind", "LDA")).upper()
    width = {"LDA": 1, "GGA": 4, "MGGA": 5, "MGGA_LAPL": 6}.get(kind, 6)
    from ._semilocal_response import SemilocalResponseFunctional
    from ._unrestricted_semilocal_response import UnrestrictedSemilocalResponseFunctional

    standard_hf = (isinstance(resolved_xc, SemilocalResponseFunctional)
                   or type(resolved_xc) is UnrestrictedSemilocalResponseFunctional)
    standard_hf = standard_hf and resolved_xc.xc_spec == "hf"
    global_hvp = False
    if resolved_xc is not None and not standard_hf:
        known_spin_local = type(resolved_xc) is UnrestrictedSemilocalResponseFunctional
        global_hvp = not known_spin_local and any(callable(getattr(resolved_xc, name, None))
            for name in ("grid_response_hvp", "spin_grid_response_hvp", "ao_kernel_action"))
        # Equal-spin restricted kernels can retain both spin and projected
        # Hessians during construction. Unknown HVPs retain complete fields.
        spin_width = 2 * width
        resident += itemsize * ngrids * (
            4 * spin_width if global_hvp else
            2 * width**2 + 2 * spin_width**2 + 4 * spin_width)
    grid_vector = (6 * spin_channels * width * ngrids * itemsize if global_hvp else 0)
    provider = getattr(molecule, "grid_ao", None)
    ao_workspace = 0
    if provider is not None and resolved_xc is not None and not standard_hf:
        from ..integrals.grids.blocks import ao_point_bytes

        ao_workspace = ao_point_bytes(provider, deriv=0 if kind == "LDA" else
                                      2 if kind == "MGGA_LAPL" else 1)
    coefficients = getattr(molecule, "mo_coeff", None)
    device = getattr(coefficients, "device", None)
    if callable(device):
        device = device()
    return dict(resident_bytes=resident, grid_vector_bytes=grid_vector,
                ao_point_bytes=ao_workspace, device=device)


def plan_response_memory(nao, dim, nroots, dtype, max_memory,
                         max_subspace=None, rpa=False, *, direct_basis=None,
                         integral_cache="auto", direct_workers=8, direct_tol=0.,
                         grid_block_size=0, spin_channels=1, resident_bytes=0,
                         grid_vector_bytes=0, ao_point_bytes=0, device=None):
    """Bound primary solver/AO work arrays; max_memory is decimal MB.

    The estimate reserves common AO state, output vectors, QR/Ritz work and
    one microbatch of densities and J/K outputs. It is not an RSS bound on
    XLA compilation, external functional internals or allocator caches.
    """
    validate_memory_mb(max_memory)
    for name, value in (("resident_bytes", resident_bytes),
                        ("grid_vector_bytes", grid_vector_bytes), ("ao_point_bytes", ao_point_bytes)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
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
    # The density transpose of native J/K concatenates the J/K cotangents.
    from ..integrals.backends.native.jk import NativeDirectBasis
    native = type(direct_basis) is NativeDirectBasis
    if device is None and native:
        centers = getattr(getattr(direct_basis, "parameters", None), "centers", None)
        device = getattr(centers, "device", None)
        if callable(device):
            device = device()
    budget = workspace_bytes(max_memory, resident_bytes=resident_bytes, device=device)
    fixed = itemsize * (8 * spin_channels * nao**2 + 4 * dim * nroots)
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
    per_vector = (itemsize * (ao_matrices * spin_channels * nao**2 + (8 if rpa else 4) * dim)
                  + grid_vector_bytes)
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
    grid_point_bytes = (itemsize * spin_channels * (96*nao + 384) + ao_point_bytes
                        if grid_block_size else 0)
    action_budget = max(native_fixed + per_vector + grid_point_bytes, budget // 2)
    action_available = action_budget - native_fixed
    cache_bytes = 0
    if integral_cache not in {"auto", "none"}:
        raise ValueError("integral_cache must be 'auto' or 'none'.")
    if native and integral_cache == "auto" and direct_tol == 0:
        # Projected providers can have a much larger primitive AO dimension;
        # CUDA uses a different packed workspace. Unknown targets stay direct;
        # never initialize an unrelated default device to choose cache policy.
        if getattr(device, "platform", None) == "cpu":
            candidate = direct_basis.eri_cache_nbytes
            # Eight workers retain two J/K accumulators. A mixed zero-basis /
            # density JVP can have three live calls: 3*2*8 extra AO matrices.
            cached_vector = (itemsize * (60 * spin_channels * nao**2 + (8 if rpa else 4) * dim)
                             + grid_vector_bytes)
            # Reserve a second packed buffer for zero-basis JVP/recomputation.
            if 2*candidate + cached_vector + grid_point_bytes <= action_available:
                cache_bytes, per_vector = candidate, cached_vector
    action_available -= 2 * cache_bytes
    if not grid_point_bytes:
        batch_size = min(32, action_available // per_vector)
        grid_width = 0
    else:
        if 32 * per_vector + grid_point_bytes <= action_available:
            batch_size = 32
        else:
            grid_share = min(action_available - per_vector,
                             max(grid_point_bytes, action_available // 2))
            batch_size = min(32, (action_available - grid_share) // per_vector)
        grid_width = min(grid_block_size,
                         (action_available - batch_size * per_vector) // grid_point_bytes)
    required = fixed + action_budget + basis_bytes(minimum)
    if required > budget:
        raise ValueError(f"Response memory budget is {budget} bytes; at least {required} bytes are required.")
    available = budget - fixed - action_budget
    if max_subspace is not None and basis_bytes(width) > available:
        raise ValueError("Requested max_subspace exceeds the response memory budget.")
    while basis_bytes(width) > available:
        width -= 1
    return _ResponseMemoryPlan(
        width, batch_size, cache_bytes, grid_width, fixed+action_budget)
