"""First-order native Gaussian-exponent products; higher and mixed AD fail closed."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from ...._native import register_integrals
from .common import batch_product


def exponent_products(operator, atoms, basis, active, active_exponents, shape, cart, split, operator_id, auxiliary_dim=1):
    # Exponent derivatives act on bare Gaussians. The normalized coefficient
    # tangent is a separate term, supplied by JAX's normalization chain.
    exponent_linear = core.Primitive(f"gradscf_exponent_jvp_{operator}")
    exponent_adjoint = core.Primitive(f"gradscf_exponent_vjp_{operator}")

    def exponent_workspace():
        if np.any(active[:, 1] > 10):
            raise NotImplementedError("Native exponent angular raising requires orbital l <= 10")
        raised = max((int(b[1])+3)*(int(b[1])+4)//2*int(b[3]) for b in active)
        ordinary = max((int(b[1])+1)*(int(b[1])+2)//2*int(b[3]) for b in active)
        size = raised*ordinary
        if operator == "three_center":
            size *= 3*auxiliary_dim
        elif operator == "dipole":
            size *= 3
        if size >= np.iinfo(np.int32).max:
            raise ValueError("Exponent raised Cartesian cache exceeds int32 indexing")

    for primitive, suffix, result_shape in ((exponent_linear, "jvp", shape),
            (exponent_adjoint, "vjp", (len(active_exponents),))):
        def apply(env, vector, suffix=suffix, result_shape=result_shape):
            exponent_workspace()
            register_integrals()
            call = jax.ffi.ffi_call(f"gradscf_exponent_{suffix}_cpu_v1",
                jax.ShapeDtypeStruct(result_shape, jnp.float64), vmap_method="sequential")
            return call(jnp.asarray(atoms), jnp.asarray(basis), env, vector,
                operator=np.int32(operator_id), cart=np.int32(cart), split=np.int32(split))

        def abstract(env, vector, result_shape=result_shape):
            exponent_workspace()
            return jax.core.ShapedArray(result_shape, jnp.float64)

        primitive.def_impl(apply)
        primitive.def_abstract_eval(abstract)
        mlir.register_lowering(primitive, mlir.lower_fun(apply, multiple_results=False), platform="cpu")

    def exponent_jvp(primitive, primals, tangents):
        env, vector = primals
        denv, dvector = tangents
        if not isinstance(denv, ad.Zero):
            raise NotImplementedError("Native higher or mixed exponent derivatives are not implemented")
        primal = primitive.bind(env, vector)
        tangent = (ad.Zero(jax.typeof(primal).to_tangent_aval()) if isinstance(dvector, ad.Zero)
                   else primitive.bind(env, dvector))
        return primal, tangent

    def exponent_transpose(other, cotangent, env, vector):
        if ad.is_undefined_primal(env):
            raise NotImplementedError("Native higher or mixed exponent derivatives are not implemented")
        if not ad.is_undefined_primal(vector):
            return None, None
        if isinstance(cotangent, ad.Zero):
            return None, ad.Zero(vector.aval)
        return None, other.bind(env, cotangent)

    for primitive, other in ((exponent_linear, exponent_adjoint), (exponent_adjoint, exponent_linear)):
        ad.primitive_jvps[primitive] = partial(exponent_jvp, primitive)
        ad.primitive_transposes[primitive] = partial(exponent_transpose, other)
        batching.primitive_batchers[primitive] = partial(batch_product, primitive)

    return exponent_linear, exponent_adjoint
