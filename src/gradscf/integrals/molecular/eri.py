"""AO-pair layouts and explicit Cartesian-reference ERI entry points."""
from functools import lru_cache
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array

_REFERENCE_EXPORTS = {'eri_element', 'eri_tensor', 'eri_pair_matrix_packed',
                      'eri_tensor_screened', 'precompile_eri_kernels'}
__all__ = ['packed_eri_shape', *sorted(_REFERENCE_EXPORTS)]


def __getattr__(name):
    if name not in _REFERENCE_EXPORTS:
        raise AttributeError(name)
    from gradscf.integrals.backends.jax_reference import two_electron
    value = getattr(two_electron, name)
    globals()[name] = value
    return value


def packed_eri_shape(nao: int, ndim: int):
    npair=nao*(nao+1)//2
    return (npair*(npair+1)//2,) if ndim==1 else (npair,npair)


@lru_cache(maxsize=64)
def _pair_metadata(nao: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.tril_indices(int(nao))
    pair_index = np.empty((int(nao), int(nao)), dtype=np.int32)
    pair_ids = np.arange(rows.size, dtype=np.int32)
    pair_index[rows, cols] = pair_ids
    pair_index[cols, rows] = pair_ids
    multiplicity = np.where(rows == cols, 1.0, 2.0)
    return (
        rows.astype(np.int32),
        cols.astype(np.int32),
        pair_index,
        multiplicity.astype(np.float64),
    )


def _metadata_arrays(nao: int, dtype) -> tuple[Array, Array, Array, Array]:
    rows, cols, pair_index, multiplicity = _pair_metadata(int(nao))
    return (
        jnp.asarray(rows, dtype=jnp.int32),
        jnp.asarray(cols, dtype=jnp.int32),
        jnp.asarray(pair_index, dtype=jnp.int32),
        jnp.asarray(multiplicity, dtype=dtype),
    )
