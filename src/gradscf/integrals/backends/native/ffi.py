"""Forward-only CPU float64 integrals through GradSCF's private libcint FFI.

No runtime PySCF dependency. Coordinates in ``env`` are in Bohr; output units
are atomic units. ``dipole`` is the position operator r, without electron charge.
The caller supplies libcint-normalized contraction coefficients. Tables follow
the documented libcint/PySCF layout and are static under ``jax.jit``.
"""

import operator as operator_module

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.integrals._native import register_integrals

_OPERATORS = {"overlap": 0, "kinetic": 1, "nuclear": 2, "dipole": 3, "eri": 4}


def _check_eri_capacity(basis, cart, shell_slice=None):
    """Reject guaranteed native workspace overflow before declaring outputs."""
    if shell_slice is not None:
        selected = np.unique(np.concatenate([np.arange(first, last)
                             for first, last in zip(shell_slice[::2], shell_slice[1::2])]))
        basis = basis[selected]
    angular = basis[:, 1].astype(np.int64)
    cart_dims = basis[:, 3] * ((angular + 1)*(angular + 2)//2)
    shell_dims = cart_dims if cart else basis[:, 3] * (2*angular + 1)
    if int(np.max(shell_dims))**4 > np.iinfo(np.int32).max:
        raise ValueError("ERI shell workspace exceeds int32 indexing")
    # CINT2e_drv contracts in Cartesian space even for spherical output. Its
    # diagonal cache needs at least 3*(ncart*nctr)^4 doubles; native queries
    # additionally check the complete recurrence/primitive workspace.
    if 3 * int(np.max(cart_dims))**4 >= np.iinfo(np.int32).max:
        raise ValueError("ERI internal Cartesian cache exceeds int32 indexing")


def _eri_shell_slice(basis, cart, shell_slice):
    """Validate a static shell window and derive its four local AO dimensions."""
    if shell_slice is None:
        return None, None
    if isinstance(shell_slice, jax.core.Tracer):
        raise TypeError("shell_slice must be static under JIT.")
    try:
        endpoints = tuple(shell_slice)
    except TypeError as exc:
        raise TypeError("shell_slice must be a static sequence of eight integers.") from exc
    if len(endpoints) != 8:
        raise ValueError("shell_slice must have eight endpoints.")
    values = []
    for endpoint in endpoints:
        if isinstance(endpoint, jax.core.Tracer):
            raise TypeError("shell_slice endpoints must be static under JIT.")
        if isinstance(endpoint, (bool, np.bool_)):
            raise TypeError("shell_slice endpoints must be integers, not booleans.")
        try:
            values.append(operator_module.index(endpoint))
        except TypeError as exc:
            raise TypeError("shell_slice endpoints must be static integers.") from exc
    for first, last in zip(values[::2], values[1::2]):
        if not 0 <= first < last <= len(basis):
            raise ValueError("shell_slice requires 0 <= start < stop <= nshell.")
    _check_eri_capacity(basis, cart, values)
    angular = basis[:, 1].astype(np.int64)
    counts = basis[:, 3] * ((angular + 1)*(angular + 2)//2 if cart else 2*angular + 1)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    shape = tuple(int(offsets[last] - offsets[first])
                  for first, last in zip(values[::2], values[1::2]))
    if max(shape[0]*shape[1], shape[2]*shape[3]) > np.iinfo(np.int32).max:
        raise ValueError("ERI shell_slice pair dimensions exceed int32 indexing.")
    return tuple(values), shape


def _eri_slab_inputs(atm, bas, env, bounds, shape, cart):
    """Check static slab metadata without inspecting dynamic shell endpoints."""
    if not isinstance(cart, (bool, np.bool_)):
        raise TypeError("cart must be a static bool")
    tables = []
    for name, value, width in (("atm", atm, 6), ("bas", bas, 8)):
        try:
            array = np.asarray(value)
        except (jax.errors.TracerArrayConversionError, TypeError) as exc:
            raise TypeError(f"{name} must be a static int32 array") from exc
        if array.dtype != np.int32 or array.ndim != 2 or array.shape[1] != width or not len(array):
            raise ValueError(f"{name} must be a nonempty int32 array with shape (n, {width})")
        tables.append(np.ascontiguousarray(array))
    env, bounds = jnp.asarray(env), jnp.asarray(bounds)
    if env.dtype != jnp.float64 or env.ndim != 1 or env.size < 20:
        raise ValueError("env must be a float64 vector of at least 20 entries; enable JAX x64")
    if bounds.dtype != jnp.int32 or bounds.shape != (8,):
        raise ValueError("ERI slab bounds must be an int32 vector with shape (8,)")
    if isinstance(shape, jax.core.Tracer):
        raise TypeError("ERI slab shape must be static under JIT")
    try:
        dimensions = tuple(shape)
    except TypeError as exc:
        raise TypeError("ERI slab shape must contain four static positive integers") from exc
    if len(dimensions) != 4:
        raise ValueError("ERI slab shape must contain four dimensions")
    result, size = [], 1
    for dimension in dimensions:
        if isinstance(dimension, (jax.core.Tracer, bool, np.bool_)):
            raise TypeError("ERI slab shape dimensions must be static positive integers")
        try:
            dimension = operator_module.index(dimension)
        except TypeError as exc:
            raise TypeError("ERI slab shape dimensions must be static positive integers") from exc
        if dimension <= 0:
            raise ValueError("ERI slab shape dimensions must be positive")
        if dimension > np.iinfo(np.intp).max // 8 // size:
            raise ValueError("ERI slab shape exceeds addressable float64 storage")
        size *= dimension
        result.append(dimension)
    return tables[0], tables[1], env, bounds, tuple(result)


def evaluate_eri_slab(atm, bas, env, bounds, shape, cart=True):
    """Evaluate a dynamic shell window into a static, zero-padded AO slab.

    ``bounds`` is an int32 vector of eight half-open shell endpoints. The four
    actual AO dimensions must fit ``shape``; native code validates them at run
    time. This raw-ENV entry is value-only; geometry AD uses the matching
    ``autodiff.geometry.evaluate_eri_slab`` entry.
    """
    atoms, basis, env, bounds, shape = _eri_slab_inputs(atm, bas, env, bounds, shape, cart)
    register_integrals()
    call = jax.ffi.ffi_call("gradscf_eri_slab_cpu_v1",
                            jax.ShapeDtypeStruct(shape, jnp.float64),
                            vmap_method="sequential")
    return call(jnp.asarray(atoms), jnp.asarray(basis), env, bounds, cart=np.int32(cart))


def evaluate(operator, atm, bas, env, nao, cart=True, *, shell_slice=None):
    """Evaluate full Cartesian/spherical AO integrals, eagerly or inside JIT.

    ``atm`` and ``bas`` must be static int32 arrays with shapes ``(natm, 6)``
    and ``(nshell, 8)``. ``env`` is a dynamic float64 vector. Only point nuclei,
    scalar Gaussian orbitals (l <= 12), and Coulomb ERIs are supported.
    For ERIs only, positive ``env[8]`` selects erf-Coulomb long-range values;
    negative omega and nonzero omega on one-electron operators are rejected.
    Returns ``(nao, nao)``; dipole returns ``(3, nao, nao)`` and ERI returns
    ``(nao, nao, nao, nao)`` in chemists' order, with no symmetry compression.
    A static eight-endpoint ``shell_slice`` selects a native s1 ERI window;
    its local AO axes follow the four half-open shell ranges.

    This raw-ENV entry point is value-only. IntegralPlan.evaluate separates
    geometry from basis data and adds native coordinate JVP/VJP. Arbitrary
    ENV differentiation raises; no finite differences or callbacks are hidden.
    """
    if operator not in _OPERATORS:
        raise ValueError(f"Unknown native integral operator {operator!r}")
    if shell_slice is not None and operator != "eri":
        raise ValueError("shell_slice applies only to ERI.")
    if not isinstance(cart, (bool, np.bool_)):
        raise TypeError("cart must be a static bool")
    nao = operator_module.index(nao)
    if nao < 1:
        raise ValueError("nao must be positive")
    tables = []
    for name, value, width in [("atm", atm, 6), ("bas", bas, 8)]:
        try:
            array = np.asarray(value)
        except (jax.errors.TracerArrayConversionError, TypeError) as exc:
            raise TypeError(f"{name} must be a static int32 array") from exc
        if array.dtype != np.int32 or array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"{name} must be int32 with shape (n, {width})")
        if len(array) < 1:
            raise ValueError(f"{name} must not be empty")
        tables.append(np.ascontiguousarray(array))
    atoms, basis = tables
    env = jnp.asarray(env)
    if env.dtype != jnp.float64 or env.ndim != 1 or env.size < 20:
        raise ValueError("env must be a float64 vector of at least 20 entries; enable JAX x64")
    # Validate topology before entering native code (also checked at the FFI boundary).
    if np.any(basis[:, 1] < 0) or np.any(basis[:, 1] > 12):
        raise ValueError("native angular momentum must lie in 0..12")
    l = basis[:, 1].astype(np.int64)
    shell_dims = basis[:, 3] * ((l + 1)*(l + 2)//2 if cart else 2*l + 1)
    expected = int(np.sum(shell_dims))
    if expected != nao:
        raise ValueError(f"nao={nao} differs from basis AO count {expected}")
    shell_slice, slice_shape = _eri_shell_slice(basis, cart, shell_slice)
    if operator == "eri" and shell_slice is None:
        _check_eri_capacity(basis, cart)
    register_integrals()
    if shell_slice is not None:
        call = jax.ffi.ffi_call("gradscf_eri_slice_cpu_v1",
                               jax.ShapeDtypeStruct(slice_shape, jnp.float64),
                               vmap_method="sequential")
        return call(jnp.asarray(atoms), jnp.asarray(basis), env,
                    cart=np.int32(cart), shell_slice=np.asarray(shell_slice, dtype=np.int32))
    shape = (nao,)*4 if operator == "eri" else ((3, nao, nao) if operator == "dipole" else (nao, nao))
    call = jax.ffi.ffi_call("gradscf_integrals_cpu_v1",
                            jax.ShapeDtypeStruct(shape, jnp.float64),
                            vmap_method="sequential")
    return call(jnp.asarray(atoms), jnp.asarray(basis), env,
                operator=np.int32(_OPERATORS[operator]), cart=np.int32(cart))


def evaluate_compact(atm, bas, env, shape, *, layout, cart, split=0):
    register_integrals()
    if jnp.asarray(env).dtype != jnp.float64:
        raise ValueError("Native compact integrals require float64.")
    call = jax.ffi.ffi_call(
        "gradscf_compact_cpu_v1",
        jax.ShapeDtypeStruct(shape, jnp.float64),
        vmap_method="sequential",
    )
    return call(
        jnp.asarray(atm),
        jnp.asarray(bas),
        env,
        layout=np.int32(layout),
        cart=np.int32(cart),
        split=np.int32(split),
    )
