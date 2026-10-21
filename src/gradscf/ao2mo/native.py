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


def _slab_plan(basis, coefficients, arrays, env, coords, max_memory, requests,
               reserved_bytes):
    """Budget resident arrays, three buffers/output and unique slab/half buffers.

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
    available = workspace_bytes(max_memory, resident_bytes=resident + reserved_bytes,
                                device=device)
    sizes = tuple(c.shape[1] for c in coefficients)
    available -= 3 * sum(prod(sizes[i] for i in request) for request in requests) * 8
    if available < 0:
        raise ValueError('Native AO->MO output buffers exceed the working-memory budget')
    requests = tuple(dict.fromkeys(r for r in requests if all(sizes[i] for i in r)))
    if not requests:
        return None
    available -= native_jk_workspace_bytes(basis, 1, 0.)
    # Logarithmic width candidates keep planning cheap even for many shells.
    widths = [max(counts)]
    while widths[-1] < basis.nao:
        widths.append(min(basis.nao, 2*widths[-1]))
    first_keys = tuple(dict.fromkeys(r[0] for r in requests))
    second_keys = tuple(dict.fromkeys((r[0], r[3]) for r in requests))
    half_keys = tuple(dict.fromkeys((r[0], r[1], r[3]) for r in requests))
    best = None
    for q_limit in widths:
        q_groups, nq = _shell_groups(counts, q_limit)
        for r_limit in widths:
            r_groups, nr = _shell_groups(counts, r_limit, split=True)
            slab = basis.nao**2 * nq * nr
            first = sum(sizes[i] for i in first_keys) * nq * nr * basis.nao
            second = sum(sizes[i] * sizes[j] for i, j in second_keys) * nq * nr
            half = sum(sizes[i] * sizes[j] * sizes[k] for i, j, k in half_keys) * nr
            scratch = 8 * (3*slab + first + second + 3*half)
            if scratch <= available:
                score = (len(q_groups)*len(r_groups), len(r_groups), scratch)
                if best is None or score < best[0]:
                    best = score, q_groups, r_groups, nq, nr
    if best is None:
        raise ValueError('Minimum native AO->MO shell tile exceeds the working-memory budget')
    return best[1:]


def _request_permutation(request, coefficients):
    """Native physical ERI symmetry; prefer small halves, then stable space IDs."""
    permutations = ((0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
                    (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0))

    def cost(permutation):
        labels = tuple(request[i] for i in permutation)
        p, q, _, s = (coefficients[i].shape[1] for i in labels)
        return p*q*s, p*s, p, labels

    return min(permutations, key=cost)


def transform_eri_block(coefficients, basis, *, max_memory, reserved_bytes=0):
    """Single-request entry to the shared native slab/half transformation."""
    return transform_eri_blocks(coefficients, ((0, 1, 2, 3),), basis,
        max_memory=max_memory, reserved_bytes=reserved_bytes)[0]


def transform_eri_blocks(coefficients, requests, basis, *, max_memory, reserved_bytes=0):
    """Transform requested spaces with one native traversal and local half reuse.

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
    permutations = tuple(_request_permutation(r, coefficients) for r in requests)
    canonical = tuple(tuple(r[i] for i in permutation)
                      for r, permutation in zip(requests, permutations))
    fixed = replace(basis.parameters, centers=jnp.zeros_like(basis.parameters.centers),
                    nuclear_coords=jnp.zeros_like(basis.parameters.nuclear_coords))
    atm, bas, env = basis.plan.pack(fixed)
    coords = jnp.concatenate((basis.parameters.nuclear_coords, basis.parameters.centers))
    plan = _slab_plan(basis, coefficients, arrays, env, coords, max_memory,
                      canonical, reserved_bytes)
    sizes = tuple(c.shape[1] for c in coefficients)
    if plan is None:
        return tuple(jnp.zeros(tuple(sizes[i] for i in r), dtype=jnp.float64)
                     for r in requests)
    coefficients = original
    for transform in projections:
        dtypes = tuple(jnp.result_type(transform.dtype, c.dtype) for c in coefficients)
        promoted = {dtype: jnp.asarray(transform, dtype=dtype) for dtype in dict.fromkeys(dtypes)}
        coefficients = tuple(promoted[dtype] @ jnp.asarray(c, dtype=dtype)
                             for c, dtype in zip(coefficients, dtypes))
    coefficients = tuple(jnp.asarray(c, dtype=jnp.float64) for c in coefficients)
    q_groups, r_groups, nq, nr = plan
    q_groups, r_groups = jnp.asarray(q_groups), jnp.asarray(r_groups)
    nshell = len(basis.plan.topology.angular_momenta)
    keys = tuple(dict.fromkeys(r for r in canonical if all(sizes[i] for i in r)))
    first_keys = tuple(dict.fromkeys(r[0] for r in keys))
    second_keys = tuple(dict.fromkeys((r[0], r[3]) for r in keys))
    half_keys = tuple(dict.fromkeys((r[0], r[1], r[3]) for r in keys))

    @jax.checkpoint
    def r_contribution(ir, env, coords, cs):
        r_start, r_stop, r_ao = r_groups[ir]

        @jax.checkpoint
        def q_contribution(iq, env, coords, cs):
            q_start, q_stop, q_ao = q_groups[iq]
            bounds = jnp.asarray((0, nshell, q_start, q_stop,
                                  r_start, r_stop, 0, nshell), dtype=jnp.int32)
            block = evaluate_eri_slab(atm, bas, env, coords, bounds,
                                      (basis.nao, nq, nr, basis.nao),
                                      cart=basis.plan.topology.cart)
            first = {i: jnp.einsum('pqrs,pP->Pqrs', block, cs[i], precision='highest')
                     for i in first_keys}
            second = {(i, k): jnp.einsum('Pqrs,sS->PqrS', first[i], cs[k],
                                        precision='highest') for i, k in second_keys}
            # take(fill) preserves tail AO offsets; dynamic_slice would clamp.
            rows = {i: jnp.take(cs[i], q_ao + jnp.arange(nq), axis=0,
                                mode='fill', fill_value=0.)
                    for i in dict.fromkeys(r[1] for r in half_keys)}
            return tuple(jnp.einsum('PqrS,qQ->PQrS', second[i, k], rows[j],
                                    precision='highest') for i, j, k in half_keys)

        half = jax.lax.fori_loop(0, q_groups.shape[0], lambda iq, acc:
            jax.tree.map(jnp.add, acc, q_contribution(iq, env, coords, cs)),
            tuple(jnp.zeros((sizes[i], sizes[j], nr, sizes[k]), dtype=jnp.float64)
                  for i, j, k in half_keys))
        half = dict(zip(half_keys, half))
        rows = {i: jnp.take(cs[i], r_ao + jnp.arange(nr), axis=0,
                            mode='fill', fill_value=0.) for i in dict.fromkeys(r[2] for r in keys)}
        return tuple(jnp.einsum('PQrS,rR->PQRS', half[i, j, l], rows[k],
                                precision='highest') for i, j, k, l in keys)

    result = jax.lax.fori_loop(0, r_groups.shape[0], lambda ir, acc:
        jax.tree.map(jnp.add, acc, r_contribution(ir, env, coords, coefficients)),
        tuple(jnp.zeros(tuple(sizes[i] for i in key), dtype=jnp.float64) for key in keys))
    result = dict(zip(keys, result))
    return tuple(result[key].transpose(np.argsort(permutation)) if key in result else
                 jnp.zeros(tuple(sizes[i] for i in request), dtype=jnp.float64)
                 for key, permutation, request in zip(canonical, permutations, requests))
