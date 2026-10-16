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


def evaluate(operator, atm, bas, env, nao, cart=True):
    """Evaluate full Cartesian/spherical AO integrals, eagerly or inside JIT.

    ``atm`` and ``bas`` must be static int32 arrays with shapes ``(natm, 6)``
    and ``(nshell, 8)``. ``env`` is a dynamic float64 vector. Only point nuclei,
    scalar Gaussian orbitals (l <= 12), and Coulomb ERIs are supported.
    For ERIs only, positive ``env[8]`` selects erf-Coulomb long-range values;
    negative omega and nonzero omega on one-electron operators are rejected.
    Returns ``(nao, nao)``; dipole returns ``(3, nao, nao)`` and ERI returns
    ``(nao, nao, nao, nao)`` in chemists' order, with no symmetry compression.

    This raw-ENV entry point is value-only. IntegralPlan.evaluate separates
    geometry from basis data and adds native coordinate JVP/VJP. Arbitrary
    ENV differentiation raises; no finite differences or callbacks are hidden.
    """
    if operator not in _OPERATORS:
        raise ValueError(f"Unknown native integral operator {operator!r}")
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
    if operator == "eri" and int(np.max(shell_dims))**4 > np.iinfo(np.int32).max:
        raise ValueError("ERI shell workspace exceeds int32 indexing")
    if operator == "eri":
        # CINT2e_drv contracts in Cartesian space even for spherical output.
        # Its queried cache contains at least 3*nc doubles, where diagonal
        # quartets have nc=(ncart*nctr)^4. Reject this guaranteed overflow
        # before JAX allocates the full output. The native query checks the
        # complete cache expression, including recurrence/primitive buffers.
        cart_dims = basis[:, 3] * ((l + 1)*(l + 2)//2)
        if 3 * int(np.max(cart_dims))**4 >= np.iinfo(np.int32).max:
            raise ValueError("ERI internal Cartesian cache exceeds int32 indexing")
    register_integrals()
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
