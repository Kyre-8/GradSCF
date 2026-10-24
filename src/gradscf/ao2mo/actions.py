"""Exact native virtual ERI actions without four-index AO or MO storage."""
from dataclasses import dataclass
from math import prod

import jax
import jax.numpy as jnp

from gradscf._memory import (
    DEFAULT_MAX_MEMORY_MB, array_device, array_storage_bytes, resolve_memory_mb,
    validate_reserved_bytes, workspace_bytes, _requested_workspace_bytes,
)
from gradscf.integrals.backends.native.jk import (
    NativeDirectBasis, ProjectedNativeDirectBasis, native_jk_workspace_bytes,
)


def _native_action_basis(basis):
    primitive = basis.primitive_basis if isinstance(basis, ProjectedNativeDirectBasis) else basis
    if not isinstance(primitive, NativeDirectBasis) or primitive.plan.backend != 'native':
        raise TypeError('VirtualERI requires NativeDirectBasis or a single projected native basis; nested projection is unsupported')
    return primitive


def _native_action_workspace(basis, nvir, workers=1):
    """Known native optimizer bytes and one complete action RHS scratch."""
    primitive = _native_action_basis(basis)
    native = native_jk_workspace_bytes(primitive, workers, 0.)
    nao = basis.nao
    per_rhs = 8*(6*nao**2+2*nao*nvir+3*nvir**2)
    if isinstance(basis, ProjectedNativeDirectBasis):
        per_rhs += 8*(6*primitive.nao**2+4*primitive.nao*nao)
    return native, per_rhs


def _native_action_plan(c, basis, x, max_memory, reserved_bytes, *, working_bytes=None):
    """Shared worker/tile choice and opaque native/host-staging byte charge."""
    nvir = c.shape[1]
    output = jax.ShapeDtypeStruct(x.shape, jnp.float64)
    promotions = tuple(jax.ShapeDtypeStruct(value.shape, jnp.float64)
                       for value in (c, x) if value.dtype != jnp.float64)
    resident = array_storage_bytes(c, x, *jax.tree.leaves(basis), *promotions)
    resident += reserved_bytes+3*prod(output.shape)*8
    if working_bytes is None:
        available = workspace_bytes(max_memory, resident_bytes=resident,
                                    device=array_device(x, c))
    else:
        # This call-scoped snapshot already includes the shared 20% reserve.
        # A later smaller MB override can only tighten it, without re-querying
        # device capacity between forecasting and tracing the same action.
        available = min(validate_reserved_bytes(working_bytes, name='_working_bytes'),
                        _requested_workspace_bytes(max_memory))-resident
        if available < 0:
            raise ValueError('Known resident arrays exceed the working-memory budget')
    count = prod(x.shape[:-2])
    if not count or not nvir:
        return 1, 0, 0
    native_workspace, per_rhs = _native_action_workspace(basis, nvir)
    if native_workspace is None:
        raise TypeError('VirtualERI requires a complete native shell topology')
    # One density/K pair plus four CPU/CUDA transfer/staging buffers.
    # Keep the existing measured four-worker policy whenever its full native
    # workspace and at least one RHS fit; otherwise retain the one-worker path.
    workers = 1
    parallel_workspace = native_jk_workspace_bytes(_native_action_basis(basis), 4, 0.)
    if parallel_workspace is not None and parallel_workspace+per_rhs <= available:
        workers, native_workspace = 4, parallel_workspace
    block = min(count, (available-native_workspace)//per_rhs)
    if block < 1:
        raise ValueError('Minimum native virtual-action RHS exceeds the working-memory budget')
    # JAX graph outputs expose density/K/projection arrays, but not the four
    # conservative CPU/CUDA staging copies in the existing per-RHS envelope.
    staging = 4*block*_native_action_basis(basis).nao**2*8
    return workers, block, native_workspace+staging


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
    _working_bytes: int | None = None

    def tree_flatten(self):
        return (self.coefficients, self.basis), (self.max_memory, self.reserved_bytes, self._working_bytes)

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
        nvir = c.shape[1]
        if x.ndim < 2 or x.shape[-2:] != (nvir, nvir):
            raise ValueError('VirtualERI RHS must have shape (..., nvir, nvir)')
        inputs = (c, x, basis.transform) if isinstance(basis, ProjectedNativeDirectBasis) else (c, x)
        if any(jnp.iscomplexobj(value) for value in inputs):
            raise NotImplementedError('VirtualERI requires real coefficients, projection and RHS')
        if any(not jnp.issubdtype(value.dtype, jnp.floating) for value in inputs):
            raise ValueError('VirtualERI coefficients, projection and RHS must be floating point')
        # Check all known residency and promotions before creating density or
        # output buffers. Three output buffers cover loop carry/update staging.
        workers, block, _ = _native_action_plan(c, basis, x, max_memory, reserved,
                                               working_bytes=self._working_bytes)
        count = prod(x.shape[:-2])
        if not block:
            return jnp.zeros(x.shape, dtype=jnp.float64)
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
