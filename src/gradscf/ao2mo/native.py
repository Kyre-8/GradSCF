"""Exact shell-slab AO->MO transformation with compact, recomputed JAX loops."""
from dataclasses import replace
from math import prod

import jax
import jax.numpy as jnp
import numpy as np

from gradscf._memory import array_storage_bytes, workspace_bytes
from gradscf.integrals.backends.native.jk import (
    NativeDirectBasis, ProjectedNativeDirectBasis, native_jk_workspace_bytes,
)
from gradscf.integrals.backends.native.autodiff.geometry import evaluate_eri_slab


def _shell_groups(counts, width, *, split=False):
    """Contiguous whole-shell groups; optionally prevent one full-AO window."""
    offsets = np.concatenate(([0], np.cumsum(counts)))
    groups, start = [], 0
    while start < len(counts):
        stop = start + 1
        while stop < len(counts) and offsets[stop + 1] - offsets[start] <= width:
            stop += 1
        if split and start == 0 and stop == len(counts) and stop > 1:
            stop = max(1, stop // 2)
        groups.append((start, stop, int(offsets[start])))
        start = stop
    return np.asarray(groups, dtype=np.int32), max(
        int(offsets[stop] - offsets[start]) for start, stop, _ in groups)


def _slab_plan(basis, coefficients, arrays, env, coords, max_memory):
    """Budget forward resident arrays, three outputs and bounded slab/half buffers.

    The native optimizer/shell estimate is shared with direct J/K. This is a
    forward-array plan; complete solver/AD residency is charged by consumers.
    Three slab buffers conservatively include CPU staging for the CUDA bridge.
    """
    topology = basis.plan.topology
    counts = tuple(nc * ((l + 1)*(l + 2)//2 if topology.cart else 2*l + 1)
                   for l, nc in zip(topology.angular_momenta, topology.contraction_counts))
    device = None
    for array in (*coefficients, *arrays):
        if not isinstance(array, jax.core.Tracer):
            device = getattr(array, 'device', None)
            if callable(device):
                device = device()
            if device is not None:
                break
    if device is None:
        device = getattr(jax.config, 'jax_default_device', None)
    resident = array_storage_bytes(*arrays, *coefficients, env, coords,
                                   *jax.tree.leaves(basis.parameters))
    available = workspace_bytes(max_memory, resident_bytes=resident, device=device)
    shape = tuple(c.shape[1] for c in coefficients)
    available -= 3 * prod(shape) * 8
    if available < 0:
        raise ValueError('Native AO->MO output buffers exceed the working-memory budget')
    if not prod(shape):
        return None
    available -= native_jk_workspace_bytes(basis, 1, 0.)
    # Logarithmic width candidates keep planning cheap even for many shells.
    widths = [max(counts)]
    while widths[-1] < basis.nao:
        widths.append(min(basis.nao, 2*widths[-1]))
    p, q, _, s = shape
    best = None
    for q_limit in widths:
        q_groups, nq = _shell_groups(counts, q_limit)
        for r_limit in widths:
            r_groups, nr = _shell_groups(counts, r_limit, split=True)
            slab = basis.nao**2 * nq * nr
            first = p * nq * nr * basis.nao
            second = p * nq * nr * s
            half = p * q * nr * s
            scratch = 8 * (3*slab + first + second + 3*half)
            if scratch <= available:
                score = (len(q_groups)*len(r_groups), len(r_groups), scratch)
                if best is None or score < best[0]:
                    best = score, q_groups, r_groups, nq, nr
    if best is None:
        raise ValueError('Minimum native AO->MO shell tile exceeds the working-memory budget')
    return best[1:]


def transform_eri_block(coefficients, basis, *, max_memory):
    """Transform four coefficient spaces without a resident AO ERI tensor.

    Real float64 native integrals are used. Coordinate derivatives through
    order two and all JAX MO-coefficient derivatives remain live. Primitive
    exponent/contraction derivatives retain the native explicit rejection.
    """
    original = tuple(coefficients)
    arrays, projections = list(original), []
    # Plan projected/promoted coefficient storage from metadata before any
    # matrix multiplication or array conversion can allocate those buffers.
    while isinstance(basis, ProjectedNativeDirectBasis):
        transform = basis.transform
        if len(transform.shape) != 2 or any(c.shape[0] != transform.shape[1] for c in coefficients):
            raise ValueError('Projected basis transform must match coefficient AO dimensions')
        if jnp.issubdtype(transform.dtype, jnp.complexfloating):
            raise NotImplementedError('Native AO->MO requires a real projected basis transform')
        projections.append(transform)
        dtypes = tuple(jnp.result_type(transform.dtype, c.dtype) for c in coefficients)
        arrays.extend(jax.ShapeDtypeStruct(transform.shape, dtype)
                      for dtype in dict.fromkeys(dtypes) if dtype != transform.dtype)
        arrays.extend(jax.ShapeDtypeStruct(c.shape, dtype)
                      for c, dtype in zip(coefficients, dtypes) if dtype != c.dtype)
        coefficients = tuple(jax.ShapeDtypeStruct((transform.shape[0], c.shape[1]),
            dtype) for c, dtype in zip(coefficients, dtypes))
        arrays.extend((transform, *coefficients))
        basis = basis.primitive_basis
    if not isinstance(basis, NativeDirectBasis) or basis.plan.backend != 'native':
        raise TypeError('direct_basis must be a native direct integral basis')
    if any(c.shape[0] != basis.nao for c in coefficients):
        raise ValueError('Coefficient AO dimensions differ from direct_basis')
    coefficients = tuple(c if c.dtype == jnp.float64 else
                         jax.ShapeDtypeStruct(c.shape, jnp.float64) for c in coefficients)
    # ERI pair symmetry puts the smaller spaces on the first/last AO legs.
    permutation = [0, 1, 2, 3]
    if coefficients[1].shape[1] < coefficients[0].shape[1]:
        permutation[:2] = [1, 0]
    if coefficients[2].shape[1] < coefficients[3].shape[1]:
        permutation[2:] = [3, 2]
    sizes = [coefficients[i].shape[1] for i in permutation]
    if sizes[2]*sizes[3]*sizes[0] < sizes[0]*sizes[1]*sizes[3]:
        permutation.reverse()
    coefficients = tuple(coefficients[i] for i in permutation)
    fixed = replace(basis.parameters, centers=jnp.zeros_like(basis.parameters.centers),
                    nuclear_coords=jnp.zeros_like(basis.parameters.nuclear_coords))
    atm, bas, env = basis.plan.pack(fixed)
    coords = jnp.concatenate((basis.parameters.nuclear_coords, basis.parameters.centers))
    plan = _slab_plan(basis, coefficients, arrays, env, coords, max_memory)
    shape = tuple(c.shape[1] for c in coefficients)
    if plan is None:
        return jnp.zeros(shape, dtype=jnp.float64).transpose(np.argsort(permutation))
    coefficients = original
    for transform in projections:
        dtypes = tuple(jnp.result_type(transform.dtype, c.dtype) for c in coefficients)
        promoted = {dtype: jnp.asarray(transform, dtype=dtype) for dtype in dict.fromkeys(dtypes)}
        coefficients = tuple(promoted[dtype] @ jnp.asarray(c, dtype=dtype)
                             for c, dtype in zip(coefficients, dtypes))
    coefficients = tuple(jnp.asarray(coefficients[i], dtype=jnp.float64) for i in permutation)
    q_groups, r_groups, nq, nr = plan
    q_groups, r_groups = jnp.asarray(q_groups), jnp.asarray(r_groups)
    nshell = len(basis.plan.topology.angular_momenta)
    c, d, e, f = coefficients

    @jax.checkpoint
    def r_contribution(ir, env, coords, c, d, e, f):
        r_start, r_stop, r_ao = r_groups[ir]

        @jax.checkpoint
        def q_contribution(iq, env, coords, c, d, f):
            q_start, q_stop, q_ao = q_groups[iq]
            bounds = jnp.asarray((0, nshell, q_start, q_stop,
                                  r_start, r_stop, 0, nshell), dtype=jnp.int32)
            block = evaluate_eri_slab(atm, bas, env, coords, bounds,
                                      (basis.nao, nq, nr, basis.nao),
                                      cart=basis.plan.topology.cart)
            first = jnp.einsum('pqrs,pP->Pqrs', block, c, precision='highest')
            second = jnp.einsum('Pqrs,sS->PqrS', first, f, precision='highest')
            # take(fill) preserves tail AO offsets; dynamic_slice would clamp.
            rows = jnp.take(d, q_ao + jnp.arange(nq), axis=0, mode='fill', fill_value=0.)
            return jnp.einsum('PqrS,qQ->PQrS', second, rows, precision='highest')

        half = jax.lax.fori_loop(0, q_groups.shape[0], lambda iq, acc: acc +
            q_contribution(iq, env, coords, c, d, f),
            jnp.zeros((shape[0], shape[1], nr, shape[3]), dtype=jnp.float64))
        rows = jnp.take(e, r_ao + jnp.arange(nr), axis=0, mode='fill', fill_value=0.)
        return jnp.einsum('PQrS,rR->PQRS', half, rows, precision='highest')

    result = jax.lax.fori_loop(0, r_groups.shape[0], lambda ir, acc: acc +
        r_contribution(ir, env, coords, c, d, e, f), jnp.zeros(shape, dtype=jnp.float64))
    return result.transpose(np.argsort(permutation))
