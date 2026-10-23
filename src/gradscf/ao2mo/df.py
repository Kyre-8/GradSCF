"""Selected three-index DF factors with bounded auxiliary workspaces."""
import jax
import jax.numpy as jnp

from gradscf._memory import (DEFAULT_MAX_MEMORY_MB, array_storage_bytes,
                             validate_memory_mb, validate_reserved_bytes, workspace_bytes)


def _factor_block_size(factors, left, right, max_memory, reserved_bytes):
    dtype = jnp.result_type(factors, left, right)
    q, nao, _ = factors.shape
    nl, nr = left.shape[1], right.shape[1]
    resident = reserved_bytes+array_storage_bytes(factors, left, right)+2*q*nl*nr*dtype.itemsize
    device = getattr(factors, 'device', None)
    device = device() if callable(device) else device
    if device is None:
        device = getattr(jax.config, 'jax_default_device', None)
    available = workspace_bytes(max_memory, resident_bytes=resident, device=device)
    if not q or not nl or not nr:
        return 0
    # Promoted auxiliary slice, both possible half-transform layouts and
    # immutable output updates. The original AO factors remain caller-owned.
    per_aux = (nao*nao+2*nao*max(nl, nr)+3*nl*nr)*dtype.itemsize
    if available < per_aux:
        raise ValueError('Minimum DF factor workspace exceeds the memory budget')
    return min(q, available//per_aux)


def transform_df_factors(factors, left, right, *, max_memory=DEFAULT_MAX_MEMORY_MB,
                          reserved_bytes=0):
    """Return B[Q,i,a] without a four-index output or a padded AO-factor copy.

    Raw nonsymmetric AO factors are not symmetrized. The complete requested
    Qia output is mandatory; only the auxiliary contraction workspace is tiled.
    Checkpointing recomputes block intermediates in AD. The necessary input
    cotangent and arbitrary outer AD/allocator storage remain outside this plan.
    """
    maximum, reserved = validate_memory_mb(max_memory), validate_reserved_bytes(reserved_bytes)
    factors, left, right = map(jnp.asarray, (factors, left, right))
    if (factors.ndim != 3 or factors.shape[1] != factors.shape[2]
            or left.ndim != 2 or right.ndim != 2
            or left.shape[0] != factors.shape[1] or right.shape[0] != factors.shape[2]):
        raise ValueError('Supply AO factors (naux,nao,nao) and two matching coefficient matrices')
    if any(jnp.iscomplexobj(a) for a in (factors, left, right)):
        raise NotImplementedError('DF transformations require real arrays')
    block = _factor_block_size(factors, left, right, maximum, reserved)
    output = jnp.zeros((factors.shape[0], left.shape[1], right.shape[1]),
                       jnp.result_type(factors, left, right))
    if not block:
        return output
    def contract(value):
        return jnp.einsum('Qpq,pi,qa->Qia', value, left, right, precision='highest')
    def step(index, value):
        start = index*block
        source = jax.lax.dynamic_slice_in_dim(factors, start, block, axis=0)
        return jax.lax.dynamic_update_slice_in_dim(value, contract(source), start, axis=0)
    full, remainder = divmod(factors.shape[0], block)
    output = jax.lax.fori_loop(0, full, jax.checkpoint(step, prevent_cse=False), output)
    if remainder:
        output = output.at[full*block:].set(jax.checkpoint(contract)(factors[full*block:]))
    return output
