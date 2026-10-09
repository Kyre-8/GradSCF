"""Shared local exact-exchange features and bounded-memory nu storage."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import Precision
from gradscf.integrals import rinv_matrices

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ChunkedHFXNu:
    """Lazy grid-chunk API for local HFX ``nu`` matrices.

    The dense equivalent has shape ``(n_omega, ngrids, nao, nao)``.  This API
    keeps the same public shape contract while exposing only concrete grid
    slices through :meth:`grid_chunk`.
    """

    shape: tuple[int, int, int, int]
    chunk_size: int
    _grid_chunk_fn: Any | None
    _grid_chunk_padded_fn: Any | None = None
    _source_kind: str = "static"
    _dense_array: Any | None = None
    _hdf5_filename: str | None = None
    _hdf5_dataset_path: str | None = None

    @classmethod
    def from_basis(cls, basis, coords, *, omega_values, chunk_size=512):
        """Generate grid chunks using GradSCF's standalone Coulomb kernels."""
        coords = jnp.asarray(coords)
        if chunk_size < 1 or coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("Expected positive chunk_size and coordinates with shape (ngrid, 3)")
        omegas = tuple(float(w) for w in omega_values)
        if not omegas:
            raise ValueError("At least one omega channel is required")
        shape = (len(omegas), coords.shape[0], basis.nao, basis.nao)

        def evaluate(points):
            return jnp.stack([rinv_matrices(basis, points,
                zeta=None if abs(w) < 1e-14 else w*w, engine="auto",
                grid_chunk_size=max(1, min(chunk_size, points.shape[0]))) for w in omegas])

        def chunk(start, stop):
            if stop <= start:
                return jnp.zeros((len(omegas), 0, basis.nao, basis.nao), dtype=coords.dtype)
            return evaluate(coords[int(start):int(stop)])

        def padded(start, size):
            if coords.shape[0] == 0:
                return jnp.zeros((len(omegas), int(size), basis.nao, basis.nao), dtype=coords.dtype)
            indices = jnp.asarray(start)+jnp.arange(int(size))
            valid = (indices >= 0) & (indices < coords.shape[0])
            points = coords[jnp.clip(indices, 0, coords.shape[0]-1)]
            return evaluate(points)*valid[None, :, None, None]

        return cls(shape=shape, chunk_size=int(chunk_size), _grid_chunk_fn=chunk,
                   _grid_chunk_padded_fn=padded)

    def tree_flatten(self):
        if self._source_kind == "dense":
            return (
                (self._dense_array,),
                (self.shape, int(self.chunk_size), "dense"),
            )
        if self._source_kind == "hdf5":
            return (
                (),
                (
                    self.shape,
                    int(self.chunk_size),
                    "hdf5",
                    self._hdf5_filename,
                    self._hdf5_dataset_path,
                ),
            )
        return (), (self.shape, int(self.chunk_size), "static", self)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        shape = tuple(int(dim) for dim in aux_data[0])
        chunk_size = int(aux_data[1])
        source_kind = str(aux_data[2])
        if source_kind == "dense":
            (dense_array,) = children
            return cls(
                shape=shape,
                chunk_size=chunk_size,
                _grid_chunk_fn=None,
                _source_kind="dense",
                _dense_array=dense_array,
            )
        if source_kind == "hdf5":
            return cls.from_hdf5_dataset(
                aux_data[3],
                aux_data[4],
                chunk_size=chunk_size,
            )
        return aux_data[3]

    @property
    def ndim(self) -> int:
        return 4

    @classmethod
    def from_dense(cls, dense: Any, *, chunk_size: int = 512) -> "ChunkedHFXNu":
        array = np.asarray(dense)
        if array.ndim != 4:
            raise ValueError(
                "dense HFX nu cache must have shape (n_omega, ngrids, nao, nao), "
                f"got {array.shape}."
            )

        shape = tuple(int(dim) for dim in array.shape)
        return cls(
            shape=shape,
            chunk_size=int(chunk_size),
            _grid_chunk_fn=None,
            _grid_chunk_padded_fn=None,
            _source_kind="dense",
            _dense_array=array,
        )

    @classmethod
    def from_hdf5_dataset(
        cls,
        filename: str | Path,
        dataset_path: str,
        *,
        chunk_size: int = 512,
    ) -> "ChunkedHFXNu":
        import h5py

        filename_s = str(Path(filename).expanduser().resolve())
        dataset_path_s = str(dataset_path)
        with h5py.File(filename_s, "r") as handle:
            dataset = handle[dataset_path_s]
            if dataset.ndim != 4:
                raise ValueError(
                    "HDF5 HFX nu cache must have shape (n_omega, ngrids, nao, nao), "
                    f"got {dataset.shape}."
                )
            shape = tuple(int(dim) for dim in dataset.shape)
            callback_dtype = np.dtype(
                jax.dtypes.canonicalize_dtype(np.dtype(dataset.dtype))
            )

        def grid_chunk(start: int, stop: int) -> np.ndarray:
            with h5py.File(filename_s, "r") as handle:
                return np.asarray(handle[dataset_path_s][:, int(start) : int(stop)])

        def grid_chunk_padded(start: Any, chunk_size: int) -> jnp.ndarray:
            chunk_size_i = int(chunk_size)
            chunk_shape = (shape[0], chunk_size_i, shape[2], shape[3])

            def read_chunk(start_arg: Any) -> np.ndarray:
                read_start = int(np.asarray(start_arg))
                read_stop = min(read_start + chunk_size_i, shape[1])
                output = np.zeros(chunk_shape, dtype=callback_dtype)
                if read_stop <= read_start:
                    return output
                with h5py.File(filename_s, "r") as handle:
                    data = np.asarray(
                        handle[dataset_path_s][:, read_start:read_stop],
                        dtype=callback_dtype,
                    )
                output[:, : data.shape[1]] = data
                return output

            return jax.pure_callback(
                read_chunk,
                jax.ShapeDtypeStruct(chunk_shape, callback_dtype),
                jnp.asarray(start, dtype=jnp.int32),
            )

        return cls(
            shape=shape,
            chunk_size=int(chunk_size),
            _grid_chunk_fn=grid_chunk,
            _grid_chunk_padded_fn=grid_chunk_padded,
            _source_kind="hdf5",
            _dense_array=None,
            _hdf5_filename=filename_s,
            _hdf5_dataset_path=dataset_path_s,
        )


    def grid_chunk(self, start: int, stop: int) -> Any:
        if self._source_kind == "dense" and self._dense_array is not None:
            return jnp.asarray(self._dense_array)[:, int(start) : int(stop)]
        if self._grid_chunk_fn is None:
            raise AttributeError("This HFX nu source does not expose a grid chunk callback.")
        return self._grid_chunk_fn(int(start), int(stop))

    def grid_chunk_padded(self, start: Any, chunk_size: int) -> Any:
        chunk_size_i = int(chunk_size)
        if self._source_kind == "dense" and self._dense_array is not None:
            dense = jnp.asarray(self._dense_array)
            indices = jnp.asarray(start, dtype=jnp.int32) + jnp.arange(
                chunk_size_i,
                dtype=jnp.int32,
            )
            chunk = jnp.take(dense, indices, axis=1, mode="clip")
            valid = indices < int(self.shape[1])
            return jnp.where(
                valid.reshape((1, chunk_size_i, 1, 1)),
                chunk,
                jnp.zeros_like(chunk),
            )
        if self._grid_chunk_padded_fn is not None:
            return self._grid_chunk_padded_fn(start, chunk_size_i)
        if self._grid_chunk_fn is None:
            raise AttributeError("This HFX nu source does not expose a grid chunk callback.")
        start_i = int(start)
        stop_i = min(start_i + chunk_size_i, self.shape[1])
        chunk = jnp.asarray(self.grid_chunk(start_i, stop_i))
        pad = chunk_size_i - int(chunk.shape[1])
        if pad <= 0:
            return chunk
        return jnp.pad(chunk, ((0, 0), (0, pad), (0, 0), (0, 0)))

    def materialize(self) -> np.ndarray:
        if self._source_kind == "dense" and self._dense_array is not None:
            return np.asarray(jax.device_get(self._dense_array))
        chunks = [
            np.asarray(
                self.grid_chunk(start, min(start + int(self.chunk_size), self.shape[1]))
            )
            for start in range(0, self.shape[1], int(self.chunk_size))
        ]
        if not chunks:
            return np.zeros(self.shape, dtype=np.float64)
        return np.concatenate(chunks, axis=1)


def hfx_nu_source(molecule: Any | None) -> Any | None:
    if molecule is None:
        return None
    nu = getattr(molecule, "hfx_nu", None)
    if nu is not None:
        return nu
    return getattr(molecule, "hfx_nu_api", None)


def has_hfx_nu_source(molecule: Any | None) -> bool:
    return hfx_nu_source(molecule) is not None


def is_chunked_hfx_nu(source: Any | None) -> bool:
    return source is not None and callable(getattr(source, "grid_chunk", None))


def hfx_nu_shape(source: Any) -> tuple[int, int, int, int]:
    shape = getattr(source, "shape", None)
    if shape is None:
        shape = jnp.asarray(source).shape
    if len(shape) != 4:
        raise ValueError(
            "HFX nu source must have shape (n_omega, ngrids, nao, nao), "
            f"got {shape}."
        )
    return tuple(int(dim) for dim in shape)


def hfx_nu_grid_chunk(
    source: Any,
    start: int,
    stop: int,
    *,
    n_omega: int | None = None,
    dtype: Any | None = None,
) -> jnp.ndarray:
    start_i = int(start)
    stop_i = int(stop)
    if is_chunked_hfx_nu(source):
        chunk = source.grid_chunk(start_i, stop_i)
    else:
        chunk = jnp.asarray(source)[:, start_i:stop_i]
    if n_omega is not None:
        chunk = chunk[: int(n_omega)]
    return jnp.asarray(chunk, dtype=dtype)


def hfx_nu_grid_chunk_padded(
    source: Any,
    start: int,
    chunk_size: int,
    *,
    n_omega: int | None = None,
    dtype: Any | None = None,
) -> jnp.ndarray:
    _, ngrids, _, _ = hfx_nu_shape(source)
    chunk_size_i = int(chunk_size)
    if is_chunked_hfx_nu(source):
        chunk = source.grid_chunk_padded(start, chunk_size_i)
        if n_omega is not None:
            chunk = chunk[: int(n_omega)]
        return jnp.asarray(chunk, dtype=dtype)
    start_i = int(start)
    stop_i = min(start_i + chunk_size_i, ngrids)
    chunk = hfx_nu_grid_chunk(
        source,
        start_i,
        stop_i,
        n_omega=n_omega,
        dtype=dtype,
    )
    pad = chunk_size_i - int(chunk.shape[1])
    if pad <= 0:
        return chunk
    return jnp.pad(chunk, ((0, 0), (0, pad), (0, 0), (0, 0)))




def _local_hfx_features_from_nu_cache(
    ao: Any,
    dm_spin: tuple[Any, Any],
    nu_cache: Any,
    *,
    return_fxx: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    ao_arr = jnp.asarray(ao)
    dm_a, dm_b = (jnp.asarray(dm_spin[0]), jnp.asarray(dm_spin[1]))
    nu = jnp.asarray(nu_cache)

    e_a = jnp.einsum("gp,pq->gq", ao_arr, dm_a, precision=Precision.HIGHEST)
    e_b = jnp.einsum("gp,pq->gq", ao_arr, dm_b, precision=Precision.HIGHEST)
    fxx_a = jnp.einsum("wgbc,gc->wgb", nu, e_a, precision=Precision.HIGHEST)
    fxx_b = jnp.einsum("wgbc,gc->wgb", nu, e_b, precision=Precision.HIGHEST)
    exx_a = -0.5 * jnp.einsum("gq,wgq->wg", e_a, fxx_a, precision=Precision.HIGHEST)
    exx_b = -0.5 * jnp.einsum("gq,wgq->wg", e_b, fxx_b, precision=Precision.HIGHEST)
    exx = jnp.stack([exx_a.T, exx_b.T], axis=0)
    exx = jnp.nan_to_num(exx, nan=0.0, posinf=0.0, neginf=0.0)
    if not return_fxx:
        return exx
    fxx = jnp.nan_to_num(0.5 * (fxx_a + fxx_b), nan=0.0, posinf=0.0, neginf=0.0)
    return exx, fxx


def _local_hfx_features_from_basis_dm(
    basis: Any,
    ao: Any,
    dm_spin: tuple[Any, Any],
    coords: Any,
    *,
    omega_values: tuple[float, ...],
    chunk_size: int = 512,
    return_nu: bool = False,
    return_fxx: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
    coords_arr = jnp.asarray(coords)
    ao_arr = jnp.asarray(ao)
    ngrid = int(coords_arr.shape[0])
    hfx_chunks: list[jnp.ndarray] = []
    nu_chunks_per_omega: list[jnp.ndarray] = []
    fxx_chunks_per_omega: list[jnp.ndarray] = []

    for omega in omega_values:
        zeta = None if abs(float(omega)) < 1e-14 else float(omega) * float(omega)
        omega_nu_chunks: list[jnp.ndarray] = []
        omega_hfx_chunks: list[jnp.ndarray] = []
        omega_fxx_chunks: list[jnp.ndarray] = []
        for start in range(0, ngrid, int(chunk_size)):
            end = min(start + int(chunk_size), ngrid)
            nu_chunk = rinv_matrices(
                basis,
                coords_arr[start:end],
                zeta=zeta,
                engine="auto",
                grid_chunk_size=min(int(chunk_size), max(1, end - start)),
            )
            hfx_result = _local_hfx_features_from_nu_cache(
                ao_arr[start:end],
                dm_spin,
                nu_chunk[None, ...],
                return_fxx=return_fxx,
            )
            if return_fxx:
                hfx_chunk, fxx_chunk = hfx_result
                omega_fxx_chunks.append(fxx_chunk[0])
            else:
                hfx_chunk = hfx_result
            omega_hfx_chunks.append(hfx_chunk[:, :, 0])
            if return_nu:
                omega_nu_chunks.append(nu_chunk)
        hfx_chunks.append(jnp.concatenate(omega_hfx_chunks, axis=1))
        if return_nu:
            nu_chunks_per_omega.append(jnp.concatenate(omega_nu_chunks, axis=0))
        if return_fxx:
            fxx_chunks_per_omega.append(jnp.concatenate(omega_fxx_chunks, axis=0))

    hfx_local = jnp.stack(hfx_chunks, axis=-1)
    outputs = [hfx_local]
    if return_nu:
        outputs.append(jnp.stack(nu_chunks_per_omega, axis=0))
    if return_fxx:
        outputs.append(jnp.stack(fxx_chunks_per_omega, axis=0))
    if len(outputs) == 1:
        return hfx_local
    return tuple(outputs)

__all__ = [
    'ChunkedHFXNu',
    '_local_hfx_features_from_basis_dm',
    '_local_hfx_features_from_nu_cache',
    'has_hfx_nu_source',
    'hfx_nu_grid_chunk',
    'hfx_nu_grid_chunk_padded',
    'hfx_nu_shape',
    'hfx_nu_source',
    'is_chunked_hfx_nu',
]
