"""Native contraction-coefficient JVP/VJP and second derivative products.

Products contract shell-local integrals without constructing an AO Jacobian.
Mixed basis-parameter or geometry derivatives remain unsupported.
"""
from functools import partial

import jax
from jax.extend import core
from jax.interpreters import ad, batching, mlir
import jax.numpy as jnp
import numpy as np

from ...._native import register_integrals
from .common import batch_product


def coefficient_products(operator, atoms, basis, ncoeff, shape, cart, split, operator_id):
    """Construct the coefficient tangent, adjoint and second-product primitives."""
    def check_workspace(identity=False):
        if operator == "three_center":
            # libcint contracts in Cartesian space even for spherical output.
            # Adjoint products replace exactly one orbital contraction count
            # by its primitive count for the identity coefficient contraction.
            left = identity_dim if identity else orbital_dim
            if 3*left*orbital_dim*auxiliary_dim >= np.iinfo(np.int32).max:
                label = "identity contraction" if identity else "Cartesian"
                raise ValueError(f"Native three-center internal {label} cache exceeds int32 indexing")

    if operator == "three_center":
        cart_angular = [(int(b[1])+1)*(int(b[1])+2)//2 for b in basis]
        orbital_dim = max(m*int(b[3]) for m, b in zip(cart_angular[:split], basis[:split]))
        identity_dim = max(m*int(b[2]) for m, b in zip(cart_angular[:split], basis[:split]))
        auxiliary_dim = max(m*int(b[3]) for m, b in zip(cart_angular[split:], basis[split:]))
        check_workspace()

    def ffi_product(target, env, coefficients, *vectors, output_shape):
        register_integrals()
        call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(output_shape, jnp.float64),
                                vmap_method="sequential")
        return call(jnp.asarray(atoms), jnp.asarray(basis), env, coefficients, *vectors,
                    operator=np.int32(operator_id), cart=np.int32(cart),
                    split=np.int32(split))

    linear = core.Primitive(f"gradscf_coefficient_jvp_{operator}")
    adjoint = core.Primitive(f"gradscf_coefficient_vjp_{operator}")
    second = core.Primitive(f"gradscf_coefficient_hessian_jvp_{operator}")
    second_adjoint = core.Primitive(f"gradscf_coefficient_hessian_vjp_{operator}")
    products = (
        (linear, "jvp", shape), (adjoint, "vjp", (ncoeff,)),
        (second, "hessian_jvp", shape), (second_adjoint, "hessian_vjp", (ncoeff,)),
    )
    for primitive, suffix, output_shape in products:
        def product(env, coefficients, *vectors, suffix=suffix, output_shape=output_shape):
            check_workspace(identity=suffix.endswith("vjp"))
            return ffi_product(f"gradscf_coefficient_{suffix}_cpu_v1", env, coefficients,
                               *vectors, output_shape=output_shape)

        def abstract_eval(*args, suffix=suffix, output_shape=output_shape):
            check_workspace(identity=suffix.endswith("vjp"))
            return jax.core.ShapedArray(output_shape, jnp.float64)

        primitive.def_impl(product)
        primitive.def_abstract_eval(abstract_eval)
        mlir.register_lowering(primitive, mlir.lower_fun(product, multiple_results=False), platform="cpu")

    def derivative(primitive, primals, tangents):
        env, coefficients, vector = primals
        denv, dcoefficients, dvector = tangents
        if not isinstance(denv, ad.Zero):
            raise NotImplementedError("Native coefficient products require fixed exponents/environment and geometry")
        primal = primitive.bind(*primals)
        terms = []
        if not isinstance(dcoefficients, ad.Zero):
            hessian = second if primitive is linear else second_adjoint
            terms.append(hessian.bind(env, coefficients, dcoefficients, vector))
        if not isinstance(dvector, ad.Zero):
            terms.append(primitive.bind(env, coefficients, dvector))
        return primal, sum(terms) if terms else ad.Zero(jax.typeof(primal).to_tangent_aval())

    def transpose(primitive, other, cotangent, env, coefficients, vector):
        if ad.is_undefined_primal(env):
            raise NotImplementedError("Native coefficient transposition requires fixed environment/geometry")
        unknown_c, unknown_v = map(ad.is_undefined_primal, (coefficients, vector))
        if unknown_c and unknown_v:
            raise NotImplementedError("A bilinear coefficient transpose requires one fixed argument")
        if isinstance(cotangent, ad.Zero):
            return (None, ad.Zero(coefficients.aval) if unknown_c else None,
                    ad.Zero(vector.aval) if unknown_v else None)
        if unknown_c:
            zero = jnp.zeros(coefficients.aval.shape, coefficients.aval.dtype)
            left, right = (vector, cotangent) if primitive is linear else (cotangent, vector)
            return None, second_adjoint.bind(env, zero, left, right), None
        return None, None, other.bind(env, coefficients, cotangent) if unknown_v else None

    def second_derivative(primitive, primals, tangents):
        env, coefficients, left, right = primals
        denv, _, dleft, dright = tangents
        if not isinstance(denv, ad.Zero):
            raise NotImplementedError("Native mixed coefficient/exponent or coefficient/geometry derivatives are not supported")
        primal = primitive.bind(*primals)
        terms = []
        # Degree two in normalized coefficients means that the Hessian has
        # exactly zero coefficient derivative, at any coefficient values.
        if not isinstance(dleft, ad.Zero):
            terms.append(primitive.bind(env, coefficients, dleft, right))
        if not isinstance(dright, ad.Zero):
            terms.append(primitive.bind(env, coefficients, left, dright))
        return primal, sum(terms) if terms else ad.Zero(jax.typeof(primal).to_tangent_aval())

    def second_transpose(primitive, cotangent, env, coefficients, left, right):
        if ad.is_undefined_primal(env):
            raise NotImplementedError("Native mixed coefficient/geometry derivatives are not supported")
        unknown_left, unknown_right = map(ad.is_undefined_primal, (left, right))
        if unknown_left and unknown_right:
            raise NotImplementedError("A bilinear Hessian transpose requires one fixed argument")
        # The native second products do not depend on primal coefficients.
        czero = ad.Zero(coefficients.aval) if ad.is_undefined_primal(coefficients) else None
        if ad.is_undefined_primal(coefficients):
            coefficients = jnp.zeros(coefficients.aval.shape, coefficients.aval.dtype)
        if isinstance(cotangent, ad.Zero):
            return (None, czero, ad.Zero(left.aval) if unknown_left else None,
                    ad.Zero(right.aval) if unknown_right else None)
        if primitive is second:
            return (None, czero,
                    second_adjoint.bind(env, coefficients, right, cotangent) if unknown_left else None,
                    second_adjoint.bind(env, coefficients, left, cotangent) if unknown_right else None)
        return (None, czero,
                second_adjoint.bind(env, coefficients, cotangent, right) if unknown_left else None,
                second.bind(env, coefficients, left, cotangent) if unknown_right else None)

    for primitive, other in ((linear, adjoint), (adjoint, linear)):
        ad.primitive_jvps[primitive] = partial(derivative, primitive)
        ad.primitive_transposes[primitive] = partial(transpose, primitive, other)
        batching.primitive_batchers[primitive] = partial(batch_product, primitive)
    for primitive in (second, second_adjoint):
        ad.primitive_jvps[primitive] = partial(second_derivative, primitive)
        ad.primitive_transposes[primitive] = partial(second_transpose, primitive)
        batching.primitive_batchers[primitive] = partial(batch_product, primitive)

    return linear, adjoint, second, second_adjoint
