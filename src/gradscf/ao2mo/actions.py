"""Exact native virtual ERI actions without four-index AO or MO storage."""
from dataclasses import dataclass
from math import prod

import jax
import jax.numpy as jnp

from gradscf._memory import (
    DEFAULT_MAX_MEMORY_MB, array_storage_bytes, resolve_memory_mb,
    validate_reserved_bytes, workspace_bytes,
)
from gradscf.integrals.backends.native.jk import (
    NativeDirectBasis, ProjectedNativeDirectBasis, native_jk_workspace_bytes,
)


def _native_action_basis(basis):
    primitive = basis.primitive_basis if isinstance(basis, ProjectedNativeDirectBasis) else basis
    if not isinstance(primitive, NativeDirectBasis) or primitive.plan.backend != 'native':
        raise TypeError('VirtualERI requires NativeDirectBasis or a single projected native basis; nested projection is unsupported')
    return primitive


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VirtualERI:
    """Bare ``(ac|bd)`` action in a real supplied virtual orbital frame.

    The native exchange operator accepts general nonsymmetric densities. Its
    existing coordinate and density AD rules remain live, together with JAX
    coefficient/projection response. Only native and single-level projected
    native bases are supported. No ERI cache or four-virtual array is created.

    ``max_memory`` is a decimal-MB forward-array plan. ``reserved_bytes`` pays
    for caller-owned resident blocks. Complete solver/AD/process residency is
    not bounded here; the caller must reserve its other live arrays separately.
    """
    coefficients: object
    basis: object
    max_memory: float = DEFAULT_MAX_MEMORY_MB
    reserved_bytes: int = 0

    def tree_flatten(self):
        return (self.coefficients, self.basis), (self.max_memory, self.reserved_bytes)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*children, *metadata)

    def contract(self, x):
        """Return ``sum_cd (ac|bd) x[...,c,d]`` in bounded RHS batches."""
        max_memory = resolve_memory_mb(self.max_memory)
        reserved = validate_reserved_bytes(self.reserved_bytes)
        c, basis = self.coefficients, self.basis
        primitive = _native_action_basis(basis)
        if (getattr(c, 'ndim', None) != 2 or c.shape[0] != basis.nao
                or (isinstance(basis, ProjectedNativeDirectBasis)
                    and basis.transform.shape != (primitive.nao, basis.nao))):
            raise ValueError('VirtualERI coefficients/projection must match the native AO dimensions')
        x = jnp.asarray(x)
        nvir, nao = c.shape[1], c.shape[0]
        if x.ndim < 2 or x.shape[-2:] != (nvir, nvir):
            raise ValueError('VirtualERI RHS must have shape (..., nvir, nvir)')
        inputs = (c, x, basis.transform) if isinstance(basis, ProjectedNativeDirectBasis) else (c, x)
        if any(jnp.iscomplexobj(value) for value in inputs):
            raise NotImplementedError('VirtualERI requires real coefficients, projection and RHS')
        if any(not jnp.issubdtype(value.dtype, jnp.floating) for value in inputs):
            raise ValueError('VirtualERI coefficients, projection and RHS must be floating point')
        # Check all known residency and promotions before creating density or
        # output buffers. Three output buffers cover loop carry/update staging.
        output = jax.ShapeDtypeStruct(x.shape, jnp.float64)
        promotions = tuple(jax.ShapeDtypeStruct(value.shape, jnp.float64)
                           for value in (c, x) if value.dtype != jnp.float64)
        resident = array_storage_bytes(c, x, *jax.tree.leaves(basis), *promotions)
        device = getattr(x, 'device', None)
        device = device() if callable(device) else device
        if device is None:
            device = getattr(jax.config, 'jax_default_device', None)
        available = workspace_bytes(max_memory, resident_bytes=reserved+resident+3*prod(output.shape)*8,
                                    device=device)
        count = prod(x.shape[:-2])
        if not count or not nvir:
            return jnp.zeros(x.shape, dtype=jnp.float64)
        native_workspace = native_jk_workspace_bytes(primitive, 1, 0.)
        if native_workspace is None:
            raise TypeError('VirtualERI requires a complete native shell topology')
        # One density/K pair plus four CPU/CUDA transfer/staging buffers.
        # direct_jk.cc writes disjoint rows directly into K, without batch
        # partials; shell/optimizer storage is charged above. Counting host
        # NumPy and CPU-device buffers separately also covers non-aliasing.
        per_rhs = 8*(6*nao**2+2*nao*nvir+3*nvir**2)
        if isinstance(basis, ProjectedNativeDirectBasis):
            per_rhs += 8*(6*primitive.nao**2+4*primitive.nao*nao)
        # The measured four-worker exchange reduces native action latency.
        # Keep the one-worker minimum for tight budgets, and charge the actual
        # optimizer/shell workspace before selecting a simultaneous RHS batch.
        workers = 1
        parallel_workspace = native_jk_workspace_bytes(primitive, 4, 0.)
        if parallel_workspace is not None and parallel_workspace+per_rhs <= available:
            workers, native_workspace = 4, parallel_workspace
        block = min(count, (available-native_workspace)//per_rhs)
        if block < 1:
            raise ValueError('Minimum native virtual-action RHS exceeds the working-memory budget')
        c, flat = jnp.asarray(c, dtype=jnp.float64), x.astype(jnp.float64).reshape(count, nvir, nvir)

        def body(index, result):
            # The final fixed-width window may overlap its predecessor. This
            # avoids padding all RHSs; the repeated exact outputs are identical.
            start = jnp.minimum(index*block, count-block)
            rhs = jax.lax.dynamic_slice_in_dim(flat, start, block)
            density = jnp.einsum('pc,bcd,qd->bpq', c, rhs, c, precision='highest')
            _, exchange = basis.get_jk(density, with_j=False, with_k=True,
                                       screening_threshold=0., workers=workers)
            value = jnp.einsum('pa,bpq,qc->bac', c, exchange, c, precision='highest')
            return jax.lax.dynamic_update_slice_in_dim(result, value, start, axis=0)

        # A constrained multi-block reverse pass recomputes each density/K
        # instead of retaining an AO-matrix tape across all RHS blocks.
        loop_body = jax.checkpoint(body) if block < count else body
        result = jax.lax.fori_loop(0, (count+block-1)//block, loop_body,
                                  jnp.zeros((count, nvir, nvir), dtype=jnp.float64))
        return result.reshape(x.shape)


def contract_vvvv(integrals, x):
    """Apply a native provider or the unchanged raw dense chemists' chart."""
    if isinstance(integrals, VirtualERI):
        return integrals.contract(x)
    return jnp.einsum('acbd,...cd->...ab', jnp.asarray(integrals), x, precision='highest')
