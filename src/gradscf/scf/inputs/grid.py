"""Basis/grid construction and device-aware caching of AO values."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from gradscf._memory import (
    DEFAULT_MAX_MEMORY_MB, array_storage_bytes, validate_memory_mb, workspace_bytes,
)
from gradscf.integrals.basis import CartesianBasis
from gradscf.data.molecule import MoleculeSpec
from gradscf.integrals.grids.blocks import AOGrid, ao_block_size, ao_point_bytes

_GRID_AO_INPUT_CACHE_MAXSIZE = 32
_GRID_AO_INPUT_CACHE_MAXBYTES = 256_000_000
_GRID_AO_INPUT_CACHE: dict[tuple[Any, ...], Any] = {}

@dataclass(frozen=True)
class _GridAOInputBundle:
    basis: CartesianBasis
    coords: Array
    grid_weights: Array
    ao: Array
    ao_deriv1: Array
    ao_laplacian: Array | None
    grid_ao: AOGrid | None = None


@dataclass(frozen=True)
class _BasisGridContext:
    basis: CartesianBasis
    coords: Array
    grid_weights: Array
    geometry_is_traced: bool
    grid_ao_bundle: _GridAOInputBundle | None
    grid_ao: AOGrid | None = None


def _grid_device(array):
    """Inspect only an existing array or the caller's selected device context.

    A tracer need not expose a device. Leaving it unknown avoids initializing
    unrelated accelerators merely to estimate working-array storage.
    """
    device = getattr(array, "device", None)
    if callable(device):
        device = device()
    return device if device is not None else getattr(jax.config, "jax_default_device", None)


def _active_device_cache_key() -> tuple[str, int]:
    device = _grid_device(jnp.asarray(0.0))
    return (str(getattr(device, "platform", "unknown")), int(getattr(device, "id", -1)))


def _grid_ao_input_cache_key(
    *,
    spec: MoleculeSpec,
    basis: Any,
    max_l: int,
    grids_level: int,
    precompute_eri_groups: bool,
    needs_ao_laplacian: bool,
    grid_block_size: int = 0,
    grid_max_memory: float | None = None,
    max_memory: float = DEFAULT_MAX_MEMORY_MB,
    scf_max_cycle: int = 80,
    scf_history_matrices: int = 2,
    fixed_workspace_bytes: int = 0,
) -> tuple[Any, ...]:
    coords_bohr = np.asarray(jax.device_get(spec.coords_bohr), dtype=np.float64)
    return (
        tuple(spec.symbols),
        str(basis),
        int(max_l),
        int(grids_level),
        bool(precompute_eri_groups),
        bool(needs_ao_laplacian),
        int(grid_block_size), grid_max_memory, float(max_memory),
        int(scf_max_cycle), int(scf_history_matrices), int(fixed_workspace_bytes),
        coords_bohr.shape,
        coords_bohr.dtype.str,
        coords_bohr.tobytes(),
        _active_device_cache_key(),
    )


def _grid_resident_bytes(basis, coords, weights, *arrays, scf_max_cycle,
                         scf_history_matrices=2, fixed_workspace_bytes=0):
    # The eager scan result can retain both density and orbital histories.
    # Reserve those as well as SCF/DIIS matrices before allocating AO work.
    itemsize = max(8, np.dtype(coords.dtype).itemsize)
    matrices = (24 + int(scf_history_matrices) * int(scf_max_cycle)) * basis.nao**2 * itemsize
    if (isinstance(fixed_workspace_bytes, bool)
            or not isinstance(fixed_workspace_bytes, Integral) or fixed_workspace_bytes < 0):
        raise ValueError("fixed_workspace_bytes must be a nonnegative integer.")
    cached_arrays = (array for bundle in _GRID_AO_INPUT_CACHE.values() for array in (
        bundle.coords, bundle.grid_weights, bundle.ao,
        bundle.ao_deriv1, bundle.ao_laplacian,
        bundle.basis.ao_centers, bundle.basis.ao_exponents_padded,
        bundle.basis.ao_coefficients_padded,
    ))
    return matrices + fixed_workspace_bytes + array_storage_bytes(
        coords, weights, basis.ao_centers,
        basis.ao_exponents_padded, basis.ao_coefficients_padded,
        *arrays, *cached_arrays,
    )


def _streamed_grid(basis, coords, weights, *, needs_ao_laplacian,
                   grid_block_size, grid_max_memory,
                   max_memory=DEFAULT_MAX_MEMORY_MB, scf_max_cycle=80,
                   scf_history_matrices=2, fixed_workspace_bytes=0,
                   _keep_streamed=False):
    if grid_block_size < -1:
        raise ValueError("grid_block_size must be -1 (dense), 0 (auto), or positive.")
    validate_memory_mb(max_memory)
    if grid_max_memory is not None:
        validate_memory_mb(grid_max_memory, name="grid_max_memory")
    if not coords.shape[0]:
        return None
    deriv = 2 if needs_ao_laplacian else 1
    device = _grid_device(coords)
    available = workspace_bytes(max_memory, device=device, resident_bytes=
        _grid_resident_bytes(basis, coords, weights, scf_max_cycle=scf_max_cycle,
            scf_history_matrices=scf_history_matrices,
            fixed_workspace_bytes=fixed_workspace_bytes))
    full_workspace = coords.shape[0] * ao_point_bytes(basis, deriv=deriv)
    if grid_block_size == -1:
        if full_workspace > available:
            raise ValueError("Dense AO quadrature exceeds max_memory; use grid_block_size=0.")
        return None
    if grid_max_memory is not None:
        available = min(available, int(grid_max_memory * 1e6))
    if grid_block_size == 0 and full_workspace <= available and not _keep_streamed:
        return None
    block = ao_block_size(basis, deriv=deriv, max_memory=available / 1e6,
                          requested=max(0, grid_block_size), device=device)
    if grid_block_size == 0:
        block = min(block, int(coords.shape[0]))
    return AOGrid.from_basis(basis, coords, weights, block_size=block,
        adaptive=grid_block_size == 0 and grid_max_memory is None)


def _build_grid_ao_input_bundle(
    *,
    basis_from_molecule_spec: Callable[..., Any],
    build_molecular_grid_from_spec: Callable[..., Any],
    evaluate_cartesian_ao_with_derivatives: Callable[..., Any],
    spec: MoleculeSpec,
    basis: Any,
    max_l: int,
    grids_level: int,
    precompute_eri_groups: bool,
    needs_ao_laplacian: bool,
    grid_block_size: int = 0,
    grid_max_memory: float | None = None,
    max_memory: float = DEFAULT_MAX_MEMORY_MB,
    scf_max_cycle: int = 80,
    scf_history_matrices: int = 2,
    fixed_workspace_bytes: int = 0,
) -> _GridAOInputBundle:
    basis_cart = basis_from_molecule_spec(
        spec,
        basis=basis,
        max_l=max_l,
        precompute_eri_groups=precompute_eri_groups,
    )
    coords, weights = build_molecular_grid_from_spec(spec, level=grids_level)
    provider = _streamed_grid(basis_cart, coords, weights,
        needs_ao_laplacian=needs_ao_laplacian, grid_block_size=grid_block_size,
        grid_max_memory=grid_max_memory, max_memory=max_memory,
        scf_max_cycle=scf_max_cycle, scf_history_matrices=scf_history_matrices,
        fixed_workspace_bytes=fixed_workspace_bytes)
    if provider is not None:
        return _GridAOInputBundle(basis_cart, coords, weights,
            jnp.zeros((0, basis_cart.nao), dtype=coords.dtype),
            jnp.zeros((4, 0, basis_cart.nao), dtype=coords.dtype), None, provider)
    deriv_order = 2 if needs_ao_laplacian else 1
    ao, ao_derivs = evaluate_cartesian_ao_with_derivatives(
        basis_cart,
        coords,
        deriv=deriv_order,
    )
    if needs_ao_laplacian:
        ao_deriv1 = ao_derivs[:4]
        ao_laplacian = ao_derivs[4]
    else:
        ao_deriv1 = ao_derivs
        ao_laplacian = None
    return _GridAOInputBundle(
        basis=basis_cart,
        coords=coords,
        grid_weights=weights,
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
    )


def _cached_grid_ao_input_bundle(
    *,
    basis_from_molecule_spec: Callable[..., Any],
    build_molecular_grid_from_spec: Callable[..., Any],
    evaluate_cartesian_ao_with_derivatives: Callable[..., Any],
    spec: MoleculeSpec,
    basis: Any,
    max_l: int,
    grids_level: int,
    precompute_eri_groups: bool,
    needs_ao_laplacian: bool,
    grid_block_size: int = 0,
    grid_max_memory: float | None = None,
    max_memory: float = DEFAULT_MAX_MEMORY_MB,
    scf_max_cycle: int = 80,
    scf_history_matrices: int = 2,
    fixed_workspace_bytes: int = 0,
) -> _GridAOInputBundle:
    key = _grid_ao_input_cache_key(
        spec=spec,
        basis=basis,
        max_l=max_l,
        grids_level=grids_level,
        precompute_eri_groups=precompute_eri_groups,
        needs_ao_laplacian=needs_ao_laplacian,
        grid_block_size=grid_block_size, grid_max_memory=grid_max_memory,
        max_memory=max_memory, scf_max_cycle=scf_max_cycle,
        scf_history_matrices=scf_history_matrices,
        fixed_workspace_bytes=fixed_workspace_bytes,
    )
    cached = _GRID_AO_INPUT_CACHE.get(key)
    if cached is not None:
        # Capacity can change while the requested budget/cache key stays the
        # same. Replan blocks, retaining the expensive quadrature and arrays.
        provider = _streamed_grid(cached.basis, cached.coords, cached.grid_weights,
            needs_ao_laplacian=needs_ao_laplacian, grid_block_size=grid_block_size,
            grid_max_memory=grid_max_memory, max_memory=max_memory,
            scf_max_cycle=scf_max_cycle, scf_history_matrices=scf_history_matrices,
            fixed_workspace_bytes=fixed_workspace_bytes,
            _keep_streamed=cached.grid_ao is not None)
        if provider is None:
            return cached
        if cached.grid_ao is not None:
            if (provider.block_size == cached.grid_ao.block_size
                    and provider.adaptive == cached.grid_ao.adaptive):
                return cached
            provider = replace(cached.grid_ao, block_size=provider.block_size,
                               adaptive=provider.adaptive)
            bundle = replace(cached, grid_ao=provider)
        else:
            dtype, nao = cached.coords.dtype, cached.basis.nao
            bundle = replace(cached, ao=jnp.zeros((0, nao), dtype=dtype),
                ao_deriv1=jnp.zeros((4, 0, nao), dtype=dtype), ao_laplacian=None,
                grid_ao=provider)
        _GRID_AO_INPUT_CACHE.pop(key)
        _cache_bounded(_GRID_AO_INPUT_CACHE, key, bundle, _GRID_AO_INPUT_CACHE_MAXSIZE)
        return bundle
    bundle = _build_grid_ao_input_bundle(
        basis_from_molecule_spec=basis_from_molecule_spec,
        build_molecular_grid_from_spec=build_molecular_grid_from_spec,
        evaluate_cartesian_ao_with_derivatives=evaluate_cartesian_ao_with_derivatives,
        spec=spec,
        basis=basis,
        max_l=max_l,
        grids_level=grids_level,
        precompute_eri_groups=precompute_eri_groups,
        needs_ao_laplacian=needs_ao_laplacian,
        grid_block_size=grid_block_size, grid_max_memory=grid_max_memory,
        max_memory=max_memory, scf_max_cycle=scf_max_cycle,
        scf_history_matrices=scf_history_matrices,
        fixed_workspace_bytes=fixed_workspace_bytes,
    )
    _cache_bounded(_GRID_AO_INPUT_CACHE, key, bundle, _GRID_AO_INPUT_CACHE_MAXSIZE)
    return bundle


def _prepare_basis_grid_context(
    *,
    basis_from_molecule_spec: Callable[..., Any],
    build_molecular_grid_from_spec: Callable[..., Any],
    evaluate_cartesian_ao_with_derivatives: Callable[..., Any],
    spec: MoleculeSpec,
    basis: Any,
    max_l: int,
    grids_level: int,
    precompute_eri_groups: bool,
    needs_ao_laplacian: bool,
    needs_grid: bool = True,
    grid_block_size: int = 0,
    grid_max_memory: float | None = None,
    max_memory: float = DEFAULT_MAX_MEMORY_MB,
    scf_max_cycle: int = 80,
    scf_history_matrices: int = 2,
    fixed_workspace_bytes: int = 0,
) -> _BasisGridContext:
    geometry_is_traced = any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(spec.coords_bohr))
    if not needs_grid:
        # Pure HF ghost energies need the basis but no XC quadrature. Build
        # it from this spec instead of reusing charge-insensitive grid data.
        basis_cart = basis_from_molecule_spec(
            spec, basis=basis, max_l=max_l,
            precompute_eri_groups=precompute_eri_groups,
        )
        dtype = jnp.asarray(spec.coords_bohr).dtype
        bundle = _GridAOInputBundle(
            basis=basis_cart,
            coords=jnp.zeros((0, 3), dtype=dtype),
            grid_weights=jnp.zeros((0,), dtype=dtype),
            ao=jnp.zeros((0, basis_cart.nao), dtype=dtype),
            ao_deriv1=jnp.zeros((4, 0, basis_cart.nao), dtype=dtype),
            ao_laplacian=None,
        )
        return _BasisGridContext(
            basis=basis_cart, coords=bundle.coords,
            grid_weights=bundle.grid_weights,
            geometry_is_traced=geometry_is_traced,
            grid_ao_bundle=bundle,
        )
    if geometry_is_traced:
        coords, weights = build_molecular_grid_from_spec(spec, level=grids_level)
        basis_cart = basis_from_molecule_spec(
            spec,
            basis=basis,
            max_l=max_l,
            precompute_eri_groups=precompute_eri_groups,
        )
        return _BasisGridContext(
            basis=basis_cart,
            coords=coords,
            grid_weights=weights,
            geometry_is_traced=True,
            grid_ao_bundle=None,
            grid_ao=_streamed_grid(basis_cart, coords, weights,
                needs_ao_laplacian=needs_ao_laplacian, grid_block_size=grid_block_size,
                grid_max_memory=grid_max_memory, max_memory=max_memory,
                scf_max_cycle=scf_max_cycle, scf_history_matrices=scf_history_matrices,
                fixed_workspace_bytes=fixed_workspace_bytes),
        )
    # Fixed coordinates may still be closed over by a JIT of a field or model
    # parameter. Build/cache their constant grid eagerly: device inspection
    # cannot consume a tracer, and cached AO arrays must not escape a trace.
    with jax.ensure_compile_time_eval():
        grid_ao_bundle = _cached_grid_ao_input_bundle(
            basis_from_molecule_spec=basis_from_molecule_spec,
            build_molecular_grid_from_spec=build_molecular_grid_from_spec,
            evaluate_cartesian_ao_with_derivatives=evaluate_cartesian_ao_with_derivatives,
            spec=spec,
            basis=basis,
            max_l=max_l,
            grids_level=grids_level,
            precompute_eri_groups=precompute_eri_groups,
            needs_ao_laplacian=needs_ao_laplacian,
            grid_block_size=grid_block_size, grid_max_memory=grid_max_memory,
            max_memory=max_memory, scf_max_cycle=scf_max_cycle,
            scf_history_matrices=scf_history_matrices,
            fixed_workspace_bytes=fixed_workspace_bytes,
        )
    return _BasisGridContext(
        basis=grid_ao_bundle.basis,
        coords=grid_ao_bundle.coords,
        grid_weights=grid_ao_bundle.grid_weights,
        geometry_is_traced=False,
        grid_ao_bundle=grid_ao_bundle,
        grid_ao=grid_ao_bundle.grid_ao,
    )


def _grid_ao_payload(
    context: _BasisGridContext,
    *,
    evaluate_cartesian_ao_with_derivatives: Callable[..., Any],
    needs_ao_laplacian: bool,
) -> tuple[Array, Array, Array | None]:
    if context.grid_ao is not None:
        dtype = context.coords.dtype
        return (jnp.zeros((0, context.basis.nao), dtype=dtype),
                jnp.zeros((4, 0, context.basis.nao), dtype=dtype), None)
    if context.coords.shape[0] == 0 and context.grid_ao_bundle is not None:
        bundle = context.grid_ao_bundle
        return bundle.ao, bundle.ao_deriv1, bundle.ao_laplacian
    if context.geometry_is_traced:
        deriv_order = 2 if needs_ao_laplacian else 1
        ao, ao_derivs = evaluate_cartesian_ao_with_derivatives(
            context.basis,
            context.coords,
            deriv=deriv_order,
        )
        if needs_ao_laplacian:
            return ao, ao_derivs[:4], ao_derivs[4]
        return ao, ao_derivs, None
    bundle = context.grid_ao_bundle
    if bundle is None:
        raise ValueError("Non-traced basis/grid context is missing cached AO data.")
    return bundle.ao, bundle.ao_deriv1, bundle.ao_laplacian


def _cache_bounded(
    cache: dict[tuple[Any, ...], Any],
    key: tuple[Any, ...],
    value: Any,
    maxsize: int,
) -> None:
    def nbytes(bundle):
        return sum(array.nbytes for array in (
            bundle.coords, bundle.grid_weights, bundle.ao,
            bundle.ao_deriv1, bundle.ao_laplacian,
        ) if array is not None)

    size = nbytes(value)
    if size > _GRID_AO_INPUT_CACHE_MAXBYTES:
        return
    while cache and (len(cache) >= maxsize
                     or sum(nbytes(bundle) for bundle in cache.values()) + size
                     > _GRID_AO_INPUT_CACHE_MAXBYTES):
        cache.pop(next(iter(cache)))
    cache[key] = value
