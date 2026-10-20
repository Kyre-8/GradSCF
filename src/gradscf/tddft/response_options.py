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
    )


class _ResponseMemoryPlan(NamedTuple):
    max_subspace: int
    batch_size: int


def plan_response_memory(nao, dim, nroots, dtype, max_memory,
                         max_subspace=None, rpa=False):
    """Bound primary solver/AO work arrays; max_memory is decimal MB.

    The estimate reserves common AO state, output vectors, QR/Ritz work and
    one microbatch of densities and J/K outputs. It is not an RSS bound on
    XLA compilation, external functional internals or allocator caches.
    """
    if (isinstance(max_memory, bool) or not isinstance(max_memory, Real)
            or not isfinite(max_memory) or max_memory <= 0):
        raise ValueError("max_memory must be a finite positive number of MB.")
    if min(nao, dim, nroots) < 1 or nroots > dim:
        raise ValueError("Invalid response dimensions or requested root count.")
    if max_subspace is not None and (isinstance(max_subspace, bool)
            or not isinstance(max_subspace, Integral)):
        raise ValueError("max_subspace must be an integer.")
    # Public eigensolvers promote float32 to float64 when x64 is enabled.
    # Eight bytes is conservative for the supported real response paths.
    itemsize = max(8, np.dtype(dtype).itemsize)
    budget = int(max_memory * 1_000_000)
    fixed = itemsize * (8 * nao**2 + 4 * dim * nroots)
    # The density transpose of native J/K concatenates the J/K cotangents.
    per_vector = itemsize * (12 * nao**2 + (8 if rpa else 4) * dim)
    minimum = min(dim, nroots + (2 if rpa else 3))
    default = max(8 * nroots, 128) if rpa else max(40, 4 * (nroots + 2))
    width = min(dim, default if max_subspace is None else int(max_subspace))
    if width < minimum:
        raise ValueError("max_subspace must retain requested and boundary roots plus expansion slots.")
    def basis_bytes(columns):
        return itemsize * ((10 if rpa else 6) * dim * columns + 8 * columns**2)
    # Reserve the same action share for builder and solver, independently of
    # the requested roots. The remaining budget belongs to the Ritz basis.
    action_budget = max(per_vector, budget // 4)
    required = fixed + action_budget + basis_bytes(minimum)
    if required > budget:
        raise ValueError(f"Response memory budget is {budget} bytes; at least {required} bytes are required.")
    available = budget - fixed - action_budget
    if max_subspace is not None and basis_bytes(width) > available:
        raise ValueError("Requested max_subspace exceeds the response memory budget.")
    while basis_bytes(width) > available:
        width -= 1
    return _ResponseMemoryPlan(width, min(32, max(1, action_budget // per_vector)))
