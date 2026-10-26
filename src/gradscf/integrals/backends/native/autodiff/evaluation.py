"""Combined native value evaluation and differentiation dispatch.

Static metadata and independent active parameter storage are validated here.
Coefficient, exponent and geometry products remain owned by their AD modules;
JAX owns primitive and contraction normalization before this entry point.
"""
from functools import lru_cache, partial
import operator as operator_module

import jax
from jax.custom_derivatives import SymbolicZero
import jax.numpy as jnp
import numpy as np

from ..ffi import evaluate as evaluate_values
from .geometry import _geometry_function
from .coefficients import coefficient_products
from .exponents import exponent_products


_OPERATORS = {"overlap": 0, "kinetic": 1, "nuclear": 2, "dipole": 3,
              "three_center": 6}


@lru_cache(maxsize=64)
def _integral_function(operator, atoms_key, basis_key, nao, cart, split):
    atoms = np.asarray(atoms_key, dtype=np.int32)
    basis = np.asarray(basis_key, dtype=np.int32)
    offsets = atoms[:, 1, None] + np.arange(3)[None, :]
    if np.unique(offsets).size != offsets.size:
        raise ValueError("Native plans require distinct coordinate storage per atom/center")
    active = basis[:split] if operator == "three_center" else basis
    coefficient_offsets = np.concatenate([
        np.arange(int(b[6]), int(b[6])+int(b[2])*int(b[3])) for b in active
    ])
    ncoeff = len(coefficient_offsets)
    inactive = basis[split:] if operator == "three_center" else ()
    inactive_offsets = (np.concatenate([
        np.arange(int(b[6]), int(b[6])+int(b[2])*int(b[3])) for b in inactive
    ]) if len(inactive) else np.empty(0, dtype=np.int64))
    exponent_offsets = np.concatenate([
        np.arange(int(b[5]), int(b[5])+int(b[2])) for b in basis
    ])
    active_exponents = np.concatenate([
        np.arange(int(b[5]), int(b[5])+int(b[2])) for b in active
    ])
    geometry_offsets = np.concatenate((offsets.ravel(), np.arange(1, 4)))
    # Shadow contractions assume independent active coefficient slots. A
    # shared PTR_COEFF would make the raw primal overwrite another shell or
    # fixed auxiliary data, while the products use independent shell copies.
    if (np.unique(coefficient_offsets).size != ncoeff
            or any(np.intersect1d(coefficient_offsets, other).size
                   for other in (inactive_offsets, exponent_offsets, geometry_offsets))):
        raise ValueError("Native active coefficient storage must not overlap coefficient, exponent, or geometry storage")
    inactive_exponents = exponent_offsets[len(active_exponents):]
    if (np.unique(active_exponents).size != len(active_exponents)
            or any(np.intersect1d(active_exponents, other).size for other in
                   (inactive_exponents, coefficient_offsets, inactive_offsets, geometry_offsets))):
        raise ValueError("Native active exponent storage must not overlap exponent, coefficient, or geometry storage")

    if operator == "three_center":
        cart_angular = [(int(b[1])+1)*(int(b[1])+2)//2 for b in basis]
        auxiliary_dim = max(m*int(b[3]) for m, b in zip(cart_angular[split:], basis[split:]))
        from ..density_fitting import evaluate as evaluate_compact
        angular = basis[:, 1].astype(np.int64)
        dims = basis[:, 3]*((angular+1)*(angular+2)//2 if cart else 2*angular+1)
        norb, naux = int(dims[:split].sum()), int(dims[split:].sum())
        shape = (naux, norb*(norb+1)//2)
    else:
        shape = (3, nao, nao) if operator == "dipole" else (nao, nao)

    def pack_coefficients(env, coefficients):
        return env.at[coefficient_offsets].set(coefficients, unique_indices=True)

    def pack_geometry(env, coords, origin):
        env = env.at[offsets].set(coords, unique_indices=True)
        return env.at[1:4].set(origin)

    linear, adjoint, second, second_adjoint = coefficient_products(
        operator, atoms, basis, ncoeff, shape, cart, split, _OPERATORS[operator])

    exponent_linear, exponent_adjoint = exponent_products(
        operator, atoms, basis, active, active_exponents, shape, cart, split,
        _OPERATORS[operator], auxiliary_dim if operator == "three_center" else 1)

    def pack_exponents(env, exponents):
        return env.at[active_exponents].set(exponents, unique_indices=True)

    @jax.custom_jvp
    def value(env, coefficients, coords, origin, exponents):
        env = pack_exponents(env, exponents)
        packed = pack_geometry(pack_coefficients(env, coefficients), coords, origin)
        if operator == "three_center":
            return evaluate_compact(atoms, basis, packed, shape, layout=3, cart=cart, split=split)
        return evaluate_values(operator, atoms, basis, packed, nao, cart)

    @partial(value.defjvp, symbolic_zeros=True)
    def value_jvp(primals, tangents):
        env, coefficients, coords, origin, exponents = primals
        denv, dcoefficients, dcoords, dorigin, dexponents = tangents
        if not isinstance(denv, SymbolicZero):
            raise NotImplementedError("Native basis AD requires fixed control data and auxiliary parameters")
        geometry_active = not (isinstance(dcoords, SymbolicZero) and isinstance(dorigin, SymbolicZero))
        if operator == "three_center" and geometry_active:
            raise NotImplementedError("Native three-center coefficient AD does not support geometry derivatives")
        primal = value(*primals)
        env = pack_exponents(env, exponents)
        terms = []
        if not isinstance(dcoefficients, SymbolicZero):
            terms.append(linear.bind(pack_geometry(env, coords, origin), coefficients, dcoefficients))
        if not isinstance(dexponents, SymbolicZero):
            packed = pack_geometry(pack_coefficients(env, coefficients), coords, origin)
            terms.append(exponent_linear.bind(packed, dexponents))
        if geometry_active:
            dcoords = jnp.zeros_like(coords) if isinstance(dcoords, SymbolicZero) else dcoords
            dorigin = jnp.zeros_like(origin) if isinstance(dorigin, SymbolicZero) else dorigin
            geometry = _geometry_function(operator, atoms_key, basis_key, nao, cart)
            fixed = pack_coefficients(env, coefficients)
            terms.append(jax.jvp(lambda r, o: geometry(fixed, r, o),
                                 (coords, origin), (dcoords, dorigin))[1])
        tangent = sum(terms) if terms else jnp.zeros_like(primal)
        return primal, tangent

    def from_environment(env, coefficients, coords, origin, exponents=None):
        if exponents is None:
            exponents = jax.lax.stop_gradient(env[active_exponents])
        return value(env, coefficients, coords, origin, exponents)

    return from_environment


def evaluate_differentiable(operator, atm, bas, fixed_env, coefficients, coords, origin,
                          nao, cart=True, split=0, *, exponents=None):
    """Evaluate with coefficient JVP/VJP/second products and exponent JVP/VJP.

    ``fixed_env`` retains fixed control/auxiliary data, with active coefficient
    slots zero. Supply active ``exponents`` separately to differentiate them;
    otherwise they are read from the fixed environment. Active coefficients
    are flattened shell by shell, contraction major. ``nao`` counts all supplied
    shells; ``split`` separates orbital and auxiliary shells for three-center
    output ``(naux, norb*(norb+1)//2)``.
    """
    if operator not in _OPERATORS:
        raise ValueError(f"Unknown native coefficient operator {operator!r}")
    if not isinstance(cart, (bool, np.bool_)):
        raise TypeError("cart must be a static bool")
    nao, split = operator_module.index(nao), operator_module.index(split)
    atoms, basis = np.asarray(atm), np.asarray(bas)
    for name, array, width in (("atm", atoms, 6), ("bas", basis, 8)):
        if array.dtype != np.int32 or array.ndim != 2 or array.shape[1] != width or len(array) < 1:
            raise ValueError(f"{name} must be a nonempty int32 array with shape (n, {width})")
    if operator == "three_center":
        if not 1 <= split < len(basis):
            raise ValueError("Three-center coefficient AD requires an orbital/auxiliary shell split")
    elif split != 0:
        raise ValueError("Dense coefficient AD requires split=0")
    angular = basis[:, 1].astype(np.int64)
    if np.any(angular < 0) or np.any(angular > 12):
        raise ValueError("Native angular momentum must lie in 0..12")
    dims = basis[:, 3]*((angular+1)*(angular+2)//2 if cart else 2*angular+1)
    if nao < 1 or int(dims.sum()) != nao:
        raise ValueError("nao differs from the supplied basis AO count")
    fixed_env, coefficients, coords, origin = map(jnp.asarray, (fixed_env, coefficients, coords, origin))
    active = basis[:split] if operator == "three_center" else basis
    ncoeff = int(np.sum(active[:, 2].astype(np.int64)*active[:, 3]))
    for name, array, shape in (("coefficients", coefficients, (ncoeff,)),
                               ("coords", coords, (len(atoms), 3)), ("origin", origin, (3,))):
        if array.dtype != jnp.float64 or array.shape != shape:
            raise ValueError(f"{name} must have float64 dtype and shape {shape}")
    if fixed_env.dtype != jnp.float64 or fixed_env.ndim != 1 or fixed_env.size < 20:
        raise ValueError("fixed_env must be a float64 vector of at least 20 entries")
    if exponents is not None:
        exponents = jnp.asarray(exponents)
        if exponents.dtype != jnp.float64 or exponents.shape != (int(active[:, 2].sum()),):
            raise ValueError("Exponents must be a float64 vector matching the active primitives")
    atoms_key, basis_key = tuple(map(tuple, atoms.tolist())), tuple(map(tuple, basis.tolist()))
    return _integral_function(operator, atoms_key, basis_key, nao, bool(cart), split)(
        fixed_env, coefficients, coords, origin, exponents)
