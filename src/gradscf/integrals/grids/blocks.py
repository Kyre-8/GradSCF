"""Differentiable AO quadrature blocks without a full-grid AO cache."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .ao import _evaluate_cartesian_ao_kernel, _as_device_array


def grid_slice(array, start, size: int, *, axis: int = 0):
    """Slice a fixed-size grid block, padding points beyond the grid with zero."""
    array = jnp.asarray(array)
    padding = [(0, 0)] * array.ndim
    padding[axis] = (0, size)
    return jax.lax.dynamic_slice_in_dim(
        jnp.pad(array, padding), start, size, axis=axis,
    )


def ao_block_size(basis, *, deriv: int, max_memory: float, requested: int = 0) -> int:
    """Plan an AO block in decimal MB, including primitive evaluation workspace.

    This is a working-array estimate, not a process/device memory limit. Other
    consumers (SCF matrices, response vectors, XC models) have separate storage.
    """
    if not np.isfinite(max_memory) or max_memory <= 0:
        raise ValueError("grid_max_memory must be a finite positive number of MB.")
    if requested < 0:
        raise ValueError("An AO block size must be nonnegative.")
    nprimitive = int(basis.ao_exponents_padded.shape[1])
    components = (1, 4, 5)[deriv]
    # Coordinates can promote a float32 model basis to float64 AO values.
    itemsize = max(8, np.dtype(basis.ao_exponents_padded.dtype).itemsize)
    per_point = max(1, basis.nao * itemsize * (12 * nprimitive + 2 * components + 8))
    available = int(max_memory * 1e6)
    if available < per_point:
        raise ValueError("grid_max_memory cannot hold one AO quadrature point's workspace.")
    limit = max(1, min(4096, available // per_point))
    return min(requested, limit) if requested else limit


@partial(jax.tree_util.register_dataclass,
         data_fields=("coords", "weights", "centers", "exponents", "coefficients"),
         meta_fields=("nprims", "angulars", "block_size"))
@dataclass(frozen=True)
class AOGrid:
    """AO supplier with dynamic geometry/basis leaves and bounded grid blocks.

    ``reduce`` checkpoints AO production together with its consumer. ``map``
    is for full-grid *small* density features; it must not be used to cache AO
    arrays in production SCF/response. All coordinates and quadrature weights
    remain differentiable, including a non-divisible final block.
    """

    coords: Any
    weights: Any
    centers: Any
    exponents: Any
    coefficients: Any
    nprims: tuple[int, ...]
    angulars: tuple[tuple[int, int, int], ...]
    block_size: int

    @classmethod
    def from_basis(cls, basis, coords, weights, *, block_size: int = 4096):
        if int(block_size) <= 0:
            raise ValueError("block_size must be positive.")
        coords, weights = _as_device_array(coords), _as_device_array(weights)
        if coords.ndim != 2 or coords.shape[1] != 3 or weights.shape != coords.shape[:1]:
            raise ValueError("AO grid requires coordinates (ngrids,3) and weights (ngrids,).")
        return cls(coords, weights, _as_device_array(basis.ao_centers),
                   _as_device_array(basis.ao_exponents_padded),
                   _as_device_array(basis.ao_coefficients_padded),
                   tuple(basis.ao_nprims_tuple), tuple(basis.ao_angulars), int(block_size))

    @property
    def nao(self) -> int:
        return int(self.centers.shape[0])

    @property
    def ngrids(self) -> int:
        return int(self.coords.shape[0])

    def block(self, start, size: int, *, deriv: int = 0):
        """Return AO, first derivatives, Laplacian, coordinates and weights.

        Absent derivative channels are ``None``. Response consumers may request
        a smaller block than the SCF block; this does not change the quadrature.
        """
        if size <= 0 or deriv not in (0, 1, 2):
            raise ValueError("AO blocks need size>0 and deriv in (0,1,2).")
        coords = grid_slice(self.coords, start, size)
        weights = grid_slice(self.weights, start, size)
        values = _evaluate_cartesian_ao_kernel(
            coords, self.centers, self.exponents, self.coefficients,
            jnp.asarray(self.nprims, dtype=jnp.int32),
            jnp.asarray(self.angulars, dtype=jnp.int32).reshape(self.nao, 3),
            deriv=deriv,
        )
        if deriv == 0:
            return values, None, None, coords, weights
        return values[0], values[:4], values[4] if deriv == 2 else None, coords, weights

    def reduce(self, function: Callable, initial, *, deriv: int = 0):
        """Reduce blocks; recompute block intermediates during reverse AD."""
        if self.ngrids == 0:
            return initial

        def body(carry, index):
            block = self.block(index * self.block_size, self.block_size, deriv=deriv)
            return function(carry, *block), None

        count = (self.ngrids + self.block_size - 1) // self.block_size
        return jax.lax.scan(jax.checkpoint(body), initial, jnp.arange(count))[0]

    def map(self, function: Callable, *, deriv: int = 0):
        """Collect grid-leading small feature arrays, trimming padded points."""
        def body(_, index):
            block = self.block(index * self.block_size, self.block_size, deriv=deriv)
            features = function(*block)
            for leaf in jax.tree_util.tree_leaves(features):
                if leaf.ndim == 0 or leaf.shape[0] != self.block_size:
                    raise ValueError("AOGrid.map outputs must have a leading grid-block axis.")
            return None, features

        count = (self.ngrids + self.block_size - 1) // self.block_size
        _, features = jax.lax.scan(jax.checkpoint(body), None, jnp.arange(count))
        return jax.tree_util.tree_map(
            lambda value: value.reshape((-1,) + value.shape[2:])[:self.ngrids], features,
        )
