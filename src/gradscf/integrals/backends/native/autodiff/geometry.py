"""Geometry AD using native shell contractions and JAX FFI.

Primal basis data are separate from differentiable coordinates so unsupported
basis derivatives cannot silently become zero. Coordinate Hessians use native
shell contractions for bilinear JVPs and their adjoints, without constructing
an integral-coordinate Jacobian. Cartesian S/T/nuclear/full-Coulomb ERIs also
have cubic/quartic products; higher orders, spherical and dipole higher
products, and mixed basis-parameter derivatives remain explicitly unsupported.
"""
from functools import lru_cache, partial

import jax
from jax.custom_derivatives import SymbolicZero
from jax.extend import core
from jax.interpreters import ad, batching, mlir
import jax.numpy as jnp
import numpy as np

from ..ffi import evaluate as evaluate_values, _OPERATORS, _eri_shell_slice
from .common import batch_product, register_cpu_product


@lru_cache(maxsize=64)
def _geometry_function(operator, atoms_key, basis_key, nao, cart, shell_slice=None):
    atoms = np.asarray(atoms_key, dtype=np.int32)
    basis = np.asarray(basis_key, dtype=np.int32)
    offsets = atoms[:, 1, None]+np.arange(3)[None, :]
    if np.unique(offsets).size != offsets.size:
        raise ValueError("Native geometry plans require distinct coordinate storage per atom/center")
    shape = (nao,)*4 if operator == "eri" else ((3, nao, nao) if operator == "dipole" else (nao, nao))
    slice_attrs = {}
    if shell_slice is not None:
        shell_slice, shape = _eri_shell_slice(basis, cart, shell_slice)
        slice_attrs = {"shell_slice": np.asarray(shell_slice, dtype=np.int32)}

    def register(primitive, call, output_shape):
        if shell_slice is not None:
            register_cpu_product(primitive, call, output_shape)
        else:
            primitive.def_impl(call)
            primitive.def_abstract_eval(lambda *args: jax.core.ShapedArray(output_shape(*args), jnp.float64))
            mlir.register_lowering(primitive, mlir.lower_fun(call, multiple_results=False), platform="cpu")

    def target(name):
        return name.replace("gradscf_", "gradscf_eri_slice_", 1) if shell_slice is not None else name

    def pack(env, coords, origin):
        env = env.at[offsets].set(coords, unique_indices=True)
        return env.at[1:4].set(origin)

    def ffi_product(target, env, vector, output_shape):
        call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(output_shape, jnp.float64),
                                vmap_method="sequential")
        return call(jnp.asarray(atoms), jnp.asarray(basis), env, vector,
                    operator=np.int32(_OPERATORS[operator]), cart=np.int32(cart), **slice_attrs)

    # At fixed env these are mutually adjoint linear maps. Keep both behind
    # primitives so mixed AD differentiates their vector arguments without
    # reaching the raw FFI or materializing an integral Jacobian.
    linear = core.Primitive(f"gradscf_geometry_jvp_{operator}")
    adjoint = core.Primitive(f"gradscf_geometry_vjp_{operator}")
    product = lambda env, denv: ffi_product(target("gradscf_geometry_jvp_cpu_v1"), env, denv, shape)
    adjoint_product = lambda env, cot: ffi_product(target("gradscf_geometry_vjp_cpu_v1"), env, cot, env.shape)
    register(linear, product, lambda env, denv: shape)
    register(adjoint, adjoint_product, lambda env, cot: env.shape)

    second = core.Primitive(f"gradscf_geometry_hessian_jvp_{operator}")
    second_adjoint = core.Primitive(f"gradscf_geometry_hessian_vjp_{operator}")

    def second_product(target, env, left, right, output_shape):
        call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(output_shape, jnp.float64),
                                vmap_method="sequential")
        return call(jnp.asarray(atoms), jnp.asarray(basis), env, left, right,
                    operator=np.int32(_OPERATORS[operator]), cart=np.int32(cart), **slice_attrs)

    hessian_product = lambda env, u, v: second_product(
        target("gradscf_geometry_hessian_jvp_cpu_v1"), env, u, v, shape)
    hessian_adjoint = lambda env, u, cot: second_product(
        target("gradscf_geometry_hessian_vjp_cpu_v1"), env, u, cot, env.shape)
    register(second, hessian_product, lambda env, u, v: shape)
    register(second_adjoint, hessian_adjoint, lambda env, u, cot: env.shape)

    products, adjoints = {2: second}, {2: second_adjoint}

    def higher_primitives(order):
        forward = core.Primitive(f"gradscf_geometry_order{order}_jvp_{operator}")
        reverse = core.Primitive(f"gradscf_geometry_order{order}_vjp_{operator}")

        def call(target, env, direction, vector, extra, output_shape):
            function = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(output_shape, jnp.float64),
                                        vmap_method="sequential")
            return function(jnp.asarray(atoms), jnp.asarray(basis), env, direction, vector, extra,
                            operator=np.int32(_OPERATORS[operator]), cart=np.int32(cart), order=np.int32(order))

        def push(env, *vectors):
            return call("gradscf_geometry_higher_jvp_cpu_v1", env, vectors[0], vectors[1],
                        jnp.stack(vectors[2:]), shape)

        def pull(env, *vectors):
            directions, cotangent = vectors[:-1], vectors[-1]
            return call("gradscf_geometry_higher_vjp_cpu_v1", env, directions[0], cotangent,
                        jnp.stack(directions[1:]), env.shape)

        forward.def_impl(push)
        reverse.def_impl(pull)
        forward.def_abstract_eval(lambda env, *vectors: jax.core.ShapedArray(shape, jnp.float64))
        reverse.def_abstract_eval(lambda env, *vectors: jax.core.ShapedArray(env.shape, jnp.float64))
        mlir.register_lowering(forward, mlir.lower_fun(push, multiple_results=False), platform="cpu")
        mlir.register_lowering(reverse, mlir.lower_fun(pull, multiple_results=False), platform="cpu")
        return forward, reverse

    for order in (() if shell_slice is not None else (3, 4)):
        products[order], adjoints[order] = higher_primitives(order)

    def transpose(other, cotangent, env, vector):
        if ad.is_undefined_primal(env):
            raise NotImplementedError("Geometry transposition requires a fixed environment")
        if not ad.is_undefined_primal(vector):
            return None, None
        if isinstance(cotangent, ad.Zero):
            return None, ad.Zero(vector.aval)
        return None, other.bind(env, cotangent)

    def derivative(primitive, primals, tangents):
        env, vector = primals
        denv, dvector = tangents
        primal = primitive.bind(env, vector)
        terms = []
        if not isinstance(denv, ad.Zero):
            hessian = second if primitive is linear else second_adjoint
            terms.append(hessian.bind(env, denv, vector))
        if not isinstance(dvector, ad.Zero):
            terms.append(primitive.bind(env, dvector))
        tangent = sum(terms) if terms else ad.Zero.from_primal_value(primal)
        return primal, tangent

    def multilinear_derivative(order, is_adjoint, primals, tangents):
        env, *vectors = primals
        denv, *dvectors = tangents
        if not isinstance(denv, ad.Zero):
            if shell_slice is not None:
                raise NotImplementedError("Native shell_slice third geometry derivatives are not supported")
            if order == 4:
                raise NotImplementedError("Native fifth geometry derivatives are not supported")
            if not cart or operator not in ("overlap", "kinetic", "nuclear", "eri"):
                raise NotImplementedError("Native third geometry derivatives require Cartesian S/T/nuclear/full ERIs")
        family = adjoints if is_adjoint else products
        primitive = family[order]
        primal = primitive.bind(*primals)
        terms = []
        if not isinstance(denv, ad.Zero):
            terms.append(family[order+1].bind(env, denv, *vectors))
        for index, tangent in enumerate(dvectors):
            if not isinstance(tangent, ad.Zero):
                changed = list(vectors)
                changed[index] = tangent
                terms.append(primitive.bind(env, *changed))
        return primal, sum(terms) if terms else ad.Zero.from_primal_value(primal)

    def multilinear_transpose(order, is_adjoint, cotangent, env, *vectors):
        if ad.is_undefined_primal(env):
            raise NotImplementedError("Geometry transposition requires a fixed environment")
        unknown = [index for index, vector in enumerate(vectors) if ad.is_undefined_primal(vector)]
        if len(unknown) > 1:
            raise NotImplementedError("A multilinear geometry transpose requires one unknown argument")
        outputs = [None]*(len(vectors)+1)
        if not unknown:
            return tuple(outputs)
        index = unknown[0]
        if isinstance(cotangent, ad.Zero):
            outputs[index+1] = ad.Zero(vectors[index].aval)
        elif not is_adjoint:
            fixed = [vector for position, vector in enumerate(vectors) if position != index]
            outputs[index+1] = adjoints[order].bind(env, *fixed, cotangent)
        elif index == len(vectors)-1:
            # Transposing the integral-cotangent slot returns the forward product.
            outputs[index+1] = products[order].bind(env, *vectors[:-1], cotangent)
        else:
            # Coordinate derivative slots are symmetric, including the free
            # coordinate output of an adjoint. Keep the integral cotangent last.
            fixed = [vector for position, vector in enumerate(vectors[:-1]) if position != index]
            outputs[index+1] = adjoints[order].bind(env, *fixed, cotangent, vectors[-1])
        return tuple(outputs)

    for primitive, other in ((linear, adjoint), (adjoint, linear)):
        ad.primitive_transposes[primitive] = partial(transpose, other)
        ad.primitive_jvps[primitive] = partial(derivative, primitive)
        batching.primitive_batchers[primitive] = partial(batch_product, primitive)
    for order in products:
        for primitive, is_adjoint in ((products[order], False), (adjoints[order], True)):
            ad.primitive_transposes[primitive] = partial(multilinear_transpose, order, is_adjoint)
            ad.primitive_jvps[primitive] = partial(multilinear_derivative, order, is_adjoint)
            batching.primitive_batchers[primitive] = partial(batch_product, primitive)

    if shell_slice is not None:
        sliced_value = core.Primitive("gradscf_eri_slice_value")
        register(sliced_value, lambda env: evaluate_values(
            operator, atoms, basis, env, nao, cart, shell_slice=shell_slice), lambda env: shape)
        batching.primitive_batchers[sliced_value] = partial(batch_product, sliced_value)

    @jax.custom_jvp
    def value(env, coords, origin):
        packed = pack(env, coords, origin)
        return (evaluate_values(operator, atoms, basis, packed, nao, cart) if shell_slice is None
                else sliced_value.bind(packed))

    @partial(value.defjvp, symbolic_zeros=True)
    def value_jvp(primals, tangents):
        env, coords, origin = primals
        denv, dcoords, dorigin = tangents
        if not isinstance(denv, SymbolicZero):
            raise NotImplementedError("Native geometry AD does not yet support basis exponent/coefficient derivatives")
        if isinstance(dcoords, SymbolicZero):
            dcoords = jnp.zeros_like(coords)
        if isinstance(dorigin, SymbolicZero):
            dorigin = jnp.zeros_like(origin)
        primal = value(env, coords, origin)
        tangent = linear.bind(pack(env, coords, origin),
                              pack(jnp.zeros_like(env), dcoords, dorigin))
        return primal, tangent

    return value


def evaluate_geometry(operator, atm, bas, env, coords, origin, nao, cart=True, *, shell_slice=None):
    """Evaluate with geometry AD; env contains only fixed nongeometry values."""
    atoms_key = tuple(tuple(int(x) for x in row) for row in atm)
    basis_key = tuple(tuple(int(x) for x in row) for row in bas)
    shell_slice, _ = _eri_shell_slice(np.asarray(bas), cart, shell_slice)
    return _geometry_function(operator, atoms_key, basis_key, nao, cart, shell_slice)(env, coords, origin)
