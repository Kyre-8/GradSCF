"""First/second coordinate response of shell-contracted native J/K."""
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from ...._native import register_integrals
from .common import batch_product


@lru_cache(maxsize=32)
def geometry_products(atoms_key, basis_key, cart, with_j, with_k, *, workers=1):
    from ..jk import _density_transpose, _zero_basis_tangent

    atoms = np.asarray(atoms_key, dtype=np.int32)
    basis = np.asarray(basis_key, dtype=np.int32)
    nops = int(with_j) + int(with_k)
    products, adjoints = {}, {}

    def register(primitive, call, shape):
        primitive.def_impl(call)
        primitive.def_abstract_eval(lambda *args: jax.core.ShapedArray(shape(*args), jnp.float64))
        mlir.register_lowering(primitive, mlir.lower_fun(call, multiple_results=False), platform="cpu")
        host_call = jax.jit(call)

        def callback(*args):
            cpu = jax.local_devices(backend="cpu")[0]
            return np.asarray(host_call(*(jax.device_put(x, cpu) for x in args)))

        def cuda(*args):
            return jax.pure_callback(callback, jax.ShapeDtypeStruct(shape(*args), jnp.float64),
                                      *args, vmap_method="sequential")

        mlir.register_lowering(primitive, mlir.lower_fun(cuda, multiple_results=False), platform="cuda")
        batching.primitive_batchers[primitive] = partial(batch_product, primitive)

    for order in (1, 2):
        push = core.Primitive(f"gradscf_direct_jk_geometry_order{order}_jvp")
        pull = core.Primitive(f"gradscf_direct_jk_geometry_order{order}_vjp")
        products[order], adjoints[order] = push, pull

        def call(transpose, order, env, density, *vectors):
            register_integrals()
            empty = jnp.zeros((0,), dtype=env.dtype)
            direction = vectors[0] if (order == 2 or not transpose) else empty
            vector = vectors[-1] if (order == 2 or transpose) else empty
            shape = env.shape if transpose else (nops, *density.shape)
            target = "gradscf_direct_jk_geometry_vjp_cpu_v2" if transpose else "gradscf_direct_jk_geometry_jvp_cpu_v1"
            function = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(shape, jnp.float64),
                                         vmap_method="sequential")
            options = dict(workers=np.int32(workers)) if transpose else {}
            return function(jnp.asarray(atoms), jnp.asarray(basis), env, density, direction, vector,
                            order=np.int32(order), cart=np.int32(cart),
                            with_j=np.int32(with_j), with_k=np.int32(with_k), **options)

        register(push, partial(call, False, order), lambda env, density, *vectors: (nops, *density.shape))
        register(pull, partial(call, True, order), lambda env, density, *vectors: env.shape)

    def derivative(order, transpose, primals, tangents):
        env, density, *vectors = primals
        denv, ddensity, *dvectors = tangents
        family = adjoints if transpose else products
        primitive = family[order]
        primal = primitive.bind(*primals)
        terms = []
        if not isinstance(denv, ad.Zero):
            if order == 2:
                # Permit instantiated closure zeros, but no genuine third
                # coordinate derivative. This guard has its own linear rule.
                zero = _zero_basis_tangent("Native direct J/K third geometry derivatives are not supported").bind(denv)
                terms.append(jnp.zeros_like(primal) + zero)
            else:
                terms.append(family[2].bind(env, density, denv, *vectors))
        if not isinstance(ddensity, ad.Zero):
            terms.append(primitive.bind(env, ddensity, *vectors))
        for index, tangent in enumerate(dvectors):
            if not isinstance(tangent, ad.Zero):
                changed = list(vectors)
                changed[index] = tangent
                terms.append(primitive.bind(env, density, *changed))
        return primal, sum(terms) if terms else ad.Zero.from_primal_value(primal)

    def transpose_rule(order, is_adjoint, cotangent, env, density, *vectors):
        if ad.is_undefined_primal(env):
            raise NotImplementedError("Direct J/K geometry transposition requires its primal environment")
        arguments = (density, *vectors)
        unknown = [index for index, value in enumerate(arguments) if ad.is_undefined_primal(value)]
        result = [None] * (len(arguments)+1)
        if not unknown:
            return tuple(result)
        if len(unknown) != 1:
            raise NotImplementedError("A direct geometry transpose needs one unknown multilinear argument")
        index = unknown[0]
        if isinstance(cotangent, ad.Zero):
            result[index+1] = ad.Zero(arguments[index].aval)
        elif not is_adjoint:
            if index == 0:
                result[1] = _density_transpose(
                    lambda dm: products[order].bind(env, dm, *vectors), cotangent)
            else:
                fixed = [value for position, value in enumerate(vectors) if position != index-1]
                result[index+1] = adjoints[order].bind(env, density, *fixed, cotangent)
        elif index == 0:
            result[1] = _density_transpose(
                lambda dm: products[order].bind(env, dm, *vectors[:-1], cotangent), vectors[-1])
        elif index == len(arguments)-1:
            result[index+1] = products[order].bind(env, density, *vectors[:-1], cotangent)
        else:
            # Coordinate slots of a Hessian are symmetric; its output adjoint
            # supplies the free slot when transposing the fixed direction.
            result[index+1] = adjoints[order].bind(env, density, cotangent, vectors[-1])
        return tuple(result)

    for order in (1, 2):
        for primitive, is_adjoint in ((products[order], False), (adjoints[order], True)):
            ad.primitive_jvps[primitive] = partial(derivative, order, is_adjoint)
            ad.primitive_transposes[primitive] = partial(transpose_rule, order, is_adjoint)
    return products[1]
