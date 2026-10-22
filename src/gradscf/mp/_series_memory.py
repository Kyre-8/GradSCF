"""Known live-array estimates for the existing finite Taylor recurrence.

Histories and requested outputs are mandatory. Only an action's representation,
tile size and saved AD intermediates may change; no Taylor coefficient is cut.
Compilation, external-library scratch and allocator residency are not a cap.
"""
from math import comb
from typing import NamedTuple

import numpy as np

from .._memory import tree_storage_bytes, workspace_bytes
from ..ci.direct import _kernel_bytes, _topology
from ..ci.space import make_ci_space, make_uci_space


class SeriesMemory(NamedTuple):
    resident_bytes: int
    action_reserved_bytes: int
    operator_min_bytes: int
    packed: bool
    tile_elements: int
    checkpoint: bool


def history_bytes(space, config, dtype):
    """Mandatory state and requested output bytes, before integral preparation."""
    dimensions = (tuple(map(len, space.orbitals)) if config.algorithm == "tensor"
                  else tuple(map(len, _topology(space).orbitals)))
    va, vb, oa, ob = dimensions
    k = config.order // 2 if config.wavefunction_order is None else config.wavefunction_order
    output = (k + 1) * space.size if config.with_coefficients else 0
    if config.with_t2:
        output += oa * ob * va * vb
        unrestricted = space.unrestricted if config.algorithm == "tensor" else isinstance(space.nocc, tuple)
        if unrestricted:
            output += oa * oa * va * va + ob * ob * vb * vb
    return np.dtype(dtype).itemsize * ((3 * k + 2 * config.order + 14) * space.size + output)


def plan_series(hs, gs, focks, space, nocc, frozen, config, max_memory, *,
                prepared=None, reserved_bytes=0):
    """Reserve full histories once and share the same MB budget with actions.

    Raw rank views retain separate numerical kernels. Their existing CI kernel
    estimate is charged for every distinct view; selected prepared inputs can
    later replace this conservative raw-input bound with their actual payload.
    """
    k = config.order // 2 if config.wavefunction_order is None else config.wavefunction_order
    tensor = config.algorithm == "tensor"
    canonical_tensor = tensor and config.orbital_basis == "canonical"
    unrestricted = isinstance(nocc, tuple)
    dtype = np.result_type(*(f.dtype for f in focks))
    item = dtype.itemsize
    if tensor:
        build = make_uci_space if unrestricted else make_ci_space
        determinants = build(space.nmo, nocc, frozen=frozen,
            max_excitation=max(space.ranks), max_determinants=config.max_determinants)
    else:
        determinants = space
    topology = _topology(determinants)
    dimensions = tuple(map(len, topology.orbitals))
    # y_j, Psi_j and W Psi_j, the two Taylor input histories, current source,
    # channels, immutable updates and explicitly requested output copies.
    history = history_bytes(space, config, dtype)
    inputs = tree_storage_bytes(hs, gs)
    base = reserved_bytes + tree_storage_bytes(hs, gs, focks, space, prepared) + history
    workspace_bytes(max_memory, resident_bytes=base)
    tables = {(n, r, a) for block in topology.blocks
              for n, r in zip(dimensions, block.counts) for a in range(min(2, r) + 1)}
    maps = 10 * sum(comb(n, a) * comb(n, r - a) for n, r, a in tables) + 10 * space.size
    if prepared is not None:
        # Prepared rank views share these same live arrays; they do not retain
        # separate numerical kernels for each Taylor degree.
        kernel = kernels = 0
    elif canonical_tensor:
        # A spin/group partition cannot exceed the uncompressed (2N)^4
        # four-index payload. Include the one-body kernels and construction.
        kernel = item * (16 * space.nmo**4 + 8 * space.nmo**2)
        kernels = kernel
    else:
        kernel = _kernel_bytes(topology, True, dtype, None, None)
        maximum = max(space.ranks)
        ranks = {(min(maximum, 2 * j), min(maximum, 2 * j + 2)) for j in range(k)}
        ranks.add((min(maximum, 2 * k), min(maximum, max(0, 2 * (config.order - k - 1)))))
        ranks.add((min(maximum, 2), min(maximum, 4)))
        kernels = len(ranks) * kernel
        kernels += _kernel_bytes(topology, False, dtype, None, None)
    resident = base + kernels + maps
    available = workspace_bytes(max_memory, resident_bytes=resident)
    dense = 32 * getattr(space, "dense_elements", 0) * item
    extra_maps = 0
    if canonical_tensor:
        known = {(n, r) for block, tables in zip(space.blocks, space.maps)
                 for n, r, table in zip(dimensions, block.counts, tables) if table is not None}
        needed = {(n, r) for block in space.blocks for n, r in zip(dimensions, block.counts)}
        # Each missing map has one int32 index and one int8 sign per entry.
        extra_maps = 5 * sum(n**r for n, r in needed - known)
    packed = canonical_tensor and (space.dense_elements > config.max_tensor_elements
                                   or dense + extra_maps > available)
    if canonical_tensor and not packed:
        resident += extra_maps
        available -= extra_maps
    tile = min(config.max_tensor_elements, available // (4 * item + 48))
    if packed and tile < 1:
        raise ValueError("Minimum packed action tile exceeds the memory budget")
    # Noncanonical F_N is a one-body compact action. Removing one orbital
    # repeats a source coefficient at most rank times. A generic direct solve
    # probes N columns simultaneously; GMRES probes one. Canonical factors
    # retain the original exact-diagonal solve and need no action workspace.
    rank = max(space.ranks)
    columns = space.size if space.size <= 128 else 1
    scratch = (item * (6 + 5 * rank) * space.size * columns + 48 * rank * space.size
               + item * space.nmo**2) if config.orbital_basis == "noncanonical" else 0
    action_reserved = resident if prepared is not None else resident - inputs - kernel
    return SeriesMemory(resident, action_reserved, scratch, packed,
                        max(1, tile), canonical_tensor and not packed and k * dense > available)
