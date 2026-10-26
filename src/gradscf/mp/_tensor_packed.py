"""Packed MP actions using the shared compact-string contraction.

The original tensor normal-order kernels define the raw integral chart.
Only their contracted axes are antisymmetrized before reading increasing
strings; creator/spectator tiles and AD reuse the existing CI implementation.
"""
from itertools import permutations, product
from math import comb, isqrt, prod
from numbers import Integral

import jax.numpy as jnp
import numpy as np

from ..ci.direct import _action, _choices, _parity
from ..ci.space import make_ci_space, make_uci_space


def _compressed(kernel, created, contracted, dimensions):
    """Take increasing creator strings and antisymmetrize annihilator axes."""
    counts = created + contracted
    choices = tuple(_choices(n, rank)
                    for n, rank in zip(dimensions + dimensions, counts))
    value = 0.
    for orders in product(*(tuple(permutations(range(r))) for r in contracted)):
        indices = []
        for group, (rank, selected) in enumerate(zip(counts, choices)):
            order = tuple(range(rank)) if group < 4 else orders[group - 4]
            shape = [1] * 8
            shape[group] = len(selected)
            indices.extend(jnp.asarray(selected[:, i]).reshape(shape) for i in order)
        phase = prod(_parity(order) for order in orders)
        value = value + phase * (kernel[tuple(indices)] if indices else kernel)
    return value.reshape(prod(map(len, choices[:4])), prod(map(len, choices[4:])))


def _physical_space(space):
    """Use the existing packed-to-determinant phase map for the shared action."""
    frozen = tuple(tuple(p for p in range(space.nmo)
                         if p not in space.orbitals[spin] + space.orbitals[spin + 2])
                   for spin in (0, 1))
    build = make_uci_space if space.unrestricted else make_ci_space
    nocc = space.nocc if space.unrestricted else space.nocc[0]
    determinants = build(space.nmo, nocc,
        frozen=frozen if space.unrestricted else frozen[0],
        max_excitation=max(sum(b.excitation) for b in space.blocks),
        max_determinants=space.size)
    indices, signs = space.determinant_map(determinants.determinants)
    return determinants, jnp.asarray(indices), jnp.asarray(signs), jnp.asarray(np.argsort(indices))


def apply_packed(action, vector, *, max_input_rank=None, max_output_rank=None,
                 max_tile_elements=4096):
    """Apply original kernels through the shared compact CI action graph.

    The element limit bounds each tile's gather/contraction pair entries.
    Packed state/output vectors, compressed kernels, maps and AD state remain
    separate live storage; this is not a whole-process byte budget.
    """
    if (isinstance(max_tile_elements, bool) or not isinstance(max_tile_elements, Integral)
            or max_tile_elements < 1):
        raise ValueError("max_tile_elements must be a positive integer")
    if max_input_rank is not None:
        max_output_rank = (max_input_rank + 2 if max_output_rank is None
                           else min(max_output_rank, max_input_rank + 2))
    space = action.space
    dimensions = tuple(map(len, space.orbitals))
    specs = {}
    for source, target, kernel, axes, _, _, _ in action.plans:
        old, new = space.blocks[source], space.blocks[target]
        if ((max_input_rank is not None and sum(old.excitation) > max_input_rank)
                or (max_output_rank is not None and sum(new.excitation) > max_output_rank)
                or kernel in specs):
            continue
        contracted, offset = [], 0
        for count in old.counts:
            contracted.append(sum(offset <= a < offset + count for a in axes))
            offset += count
        contracted = tuple(contracted)
        created = tuple(t - s + l for t, s, l in zip(new.counts, old.counts, contracted))
        specs[kernel] = created, contracted
    if not specs:
        return jnp.zeros_like(vector)
    left = max(prod(comb(n, l) for n, l in zip(dimensions, contracted))
               for _, contracted in specs.values())
    if left > max_tile_elements:
        raise ValueError("Minimum packed contraction exceeds max_tile_elements")
    width = min(128, isqrt(max_tile_elements), max_tile_elements // left)
    # Cache only the shared CI graph's static topology. Kernels and vectors
    # remain explicit live arguments on every physical evaluation.
    determinants, indices, signs, inverse = _physical_space(space)
    kernels = tuple(_compressed(action.kernels[index], m, l, dimensions)
                    for index, (m, l) in specs.items())
    dtype = jnp.result_type(vector, *kernels)
    physical = (vector.astype(dtype) * signs)[inverse]
    columns = _action(determinants, width, tuple(specs.values()),
                      max_input_rank, max_output_rank, True)
    value = columns(kernels, jnp.zeros((), dtype), physical[:, None])[:, 0]
    return value[indices] * signs
