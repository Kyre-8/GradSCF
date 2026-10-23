"""Real CI total-spin expectation, including distinct alpha/beta MO frames.

Uses S^2 = S_z^2 + (S_+S_- + S_-S_+)/2 and the existing compact CI action.
No total-spin projection or integer multiplicity assignment is performed.
"""
import jax.numpy as jnp

from .._memory import (DEFAULT_MAX_MEMORY_MB, array_device, array_storage_bytes,
                       resolve_memory_mb, validate_reserved_bytes, workspace_bytes)
from .space import CISpace, UCISpace
from .properties import _coefficients
from .direct import _topology, _kernels, _kernel_bytes, _budgeted_columns


class _ExchangeBlock:
    """Index the factorized ab exchange directly into compressed CI kernels."""
    def __init__(self, overlap, groups, types):
        self.overlap, self.groups, self.types = overlap, groups, types

    def __getitem__(self, indices):
        if self.types[0] % 2 == self.types[2] % 2:
            return jnp.zeros(jnp.broadcast_shapes(*(i.shape for i in indices)), self.overlap.dtype)
        p, q, r, s = (jnp.asarray(self.groups[g], dtype=jnp.int32)[i]
                       for g, i in zip(self.types, indices))
        if self.types[0] % 2:
            return self.overlap[r, q]*self.overlap[s, p]
        return self.overlap[p, s]*self.overlap[q, r]


def _exchange_value(coefficients, space, overlap, max_memory, reserved_bytes):
    c = _coefficients(coefficients, space)
    topology = _topology(space)
    dtype = jnp.result_type(c, overlap)
    # Normalization, input promotion and the scalar contraction's CI vectors
    # coexist with the shared compressed kernels. Their active group widths
    # can still give quartic storage; no full RDM or total-MO tensor is formed.
    held = reserved_bytes+array_storage_bytes(coefficients, c, overlap)
    held += (8*space.nmo**2+4*space.size)*dtype.itemsize
    workspace_bytes(max_memory, resident_bytes=held+_kernel_bytes(
        topology, True, dtype, None, None), device=array_device(c, overlap))
    overlap = overlap.astype(dtype)
    na, nb = topology.nocc
    focks = (jnp.matmul(overlap[:, :nb], overlap[:, :nb].T, precision='highest'),
             jnp.matmul(overlap[:na].T, overlap[:na], precision='highest'))
    constant = jnp.sum(overlap[:na, :nb]**2)
    kernels = _kernels(focks, ((None, None), (None, None)), topology,
        block_source=lambda types: _ExchangeBlock(overlap, topology.orbitals, types))
    action = _budgeted_columns(space, kernels, constant, c[:, None], 128,
        max_memory, held, None, None)[:, 0]
    return jnp.einsum('i,i->', c, action, precision='highest')


def spin_square(coefficients, space, *, overlap_ab=None,
                max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0):
    """Return (<S^2>, sqrt(1+4<S^2>)) for a normalized real CI state.

    Normalization follows the spatial densities. Restricted spaces share orbitals.
    UCISpace requires overlap_ab[p,q] = <p_alpha|q_beta> explicitly, even for
    ROHF/common frames (supply the identity in that case). Each spin frame must
    be orthonormal; consistency of overlap_ab with the orbitals is the caller's
    responsibility. Frozen electrons participate in the density contraction.
    The returned effective multiplicity need not be an integer for mixed spin.
    Integral response requires implicit-eigenvector CI coefficients; the default
    eigenvalue-only mode stops coefficient response.
    The original decimal-MB budget covers known caller arrays, compressed
    exchange kernels and the shared sigma workspace; no full RDM is exported.
    Compressed creator/annihilator kernels can still have quartic-sized pieces.
    """
    maximum = resolve_memory_mb(max_memory)
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    workspace_bytes(maximum, resident_bytes=reserved_bytes+array_storage_bytes(coefficients),
                    device=array_device(coefficients, overlap_ab))
    if isinstance(space, CISpace):
        if overlap_ab is not None:
            raise ValueError("overlap_ab applies only to unrestricted CI spaces")
        spin_space = UCISpace(space.nmo, (space.nocc, space.nocc), space.max_excitation,
                              (space.frozen, space.frozen), space.determinants, space.ranks)
        overlap = jnp.eye(space.nmo)
    elif isinstance(space, UCISpace):
        if overlap_ab is None:
            raise ValueError("Unrestricted spin_square requires overlap_ab = C_alpha.T @ S @ C_beta")
        spin_space = space
        overlap = jnp.asarray(overlap_ab)
        if overlap.shape != (space.nmo, space.nmo):
            raise ValueError("overlap_ab must have shape (nmo,nmo)")
        if jnp.iscomplexobj(overlap):
            raise NotImplementedError("Spin diagnostics currently require real orbital overlaps")
    else:
        raise TypeError("Spin diagnostics require a determinant CISpace or UCISpace")
    na, nb = spin_space.nocc
    exchange = _exchange_value(coefficients, spin_space, overlap, maximum, reserved_bytes)
    ss = .25*(na-nb)**2 + .5*(na+nb)-exchange
    ss = jnp.where(jnp.all(jnp.isfinite(overlap)), ss, jnp.nan)
    return ss, jnp.sqrt(1+4*ss)
