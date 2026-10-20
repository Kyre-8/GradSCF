"""Native direct and packed J/K operators and bound direct-SCF providers."""

from functools import lru_cache
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.interpreters import ad, batching, mlir
from ..._native import register_integrals


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NativeDirectBasis:
    """Native AO action with static topology and dynamic basis parameters.

    J/K density derivatives are supported at a fixed basis. Basis/geometry
    derivatives remain explicitly rejected by the underlying native operator.
    """

    plan: object
    parameters: object

    def tree_flatten(self):
        return (self.parameters,), (self.plan.topology, self.plan.backend)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        from ...plan import make_plan

        topology, backend = metadata
        return cls(make_plan(topology, backend=backend), children[0])

    @property
    def nao(self):
        return self.plan.topology.nao

    def get_jk(self, density, *, screening_threshold=0.0):
        return self.plan.get_jk(
            self.parameters, density, screening_threshold=screening_threshold
        )

    def response_eri_pair_matrix(self):
        return self.plan.evaluate("eri", self.parameters, aosym="s4")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ProjectedNativeDirectBasis:
    """Exact contracted J/K from a fixed primitive shell-direct operator.

    The native call is differentiated only with respect to the primitive AO
    density.  JAX owns both contractions with ``transform``, so contraction
    coefficient derivatives do not require basis derivatives in libcint.
    """

    primitive_basis: NativeDirectBasis
    transform: object

    def tree_flatten(self):
        return (self.primitive_basis, self.transform), None

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*children)

    @property
    def nao(self):
        return self.transform.shape[1]

    def get_jk(self, density, *, screening_threshold=0.0):
        transform = jnp.asarray(self.transform)
        primitive_density = jnp.einsum(
            "pi,...ij,qj->...pq", transform, density, transform
        )
        fixed_parameters = jax.tree.map(
            jax.lax.stop_gradient, self.primitive_basis.parameters
        )
        primitive_j, primitive_k = self.primitive_basis.plan.get_jk(
            fixed_parameters, primitive_density,
            screening_threshold=screening_threshold,
        )
        project = lambda matrix: jnp.einsum(
            "pi,...pq,qj->...ij", transform, matrix, transform
        )
        return project(primitive_j), project(primitive_k)


@lru_cache(maxsize=1)
def _zero_basis_tangent():
    """Accept materialized zero tangents, never an active basis derivative.

    custom_root can instantiate symbolic zeros in closed-over constants.
    This partial linear operation is defined only on the zero tangent; its
    transpose explicitly rejects differentiation with respect to the basis.
    """
    primitive = core.Primitive("gradscf_direct_jk_zero_basis_tangent")

    def check(tangent):
        if np.any(np.asarray(tangent) != 0):
            raise NotImplementedError(
                "Direct J/K supports density AD only; basis/geometry derivatives are not implemented."
            )
        return np.asarray(0., dtype=tangent.dtype)

    primitive.def_impl(check)
    primitive.def_abstract_eval(lambda tangent: jax.core.ShapedArray((), tangent.dtype))
    mlir.register_lowering(primitive, mlir.lower_fun(
        lambda tangent: jax.pure_callback(check, jax.ShapeDtypeStruct((), tangent.dtype),
                                          tangent, vmap_method="sequential"),
        multiple_results=False))

    def jvp(primals, tangents):
        value = primitive.bind(primals[0])
        tangent = (ad.Zero(jax.core.ShapedArray((), value.dtype))
                   if isinstance(tangents[0], ad.Zero) else primitive.bind(tangents[0]))
        return value, tangent

    def transpose(cotangent, tangent):
        if not isinstance(cotangent, ad.Zero) and ad.is_undefined_primal(tangent):
            raise NotImplementedError("Direct J/K basis/geometry VJP is not implemented.")
        return (ad.Zero(tangent.aval) if ad.is_undefined_primal(tangent) else None,)

    ad.primitive_jvps[primitive] = jvp
    ad.primitive_transposes[primitive] = transpose
    batching.primitive_batchers[primitive] = lambda args, axes: (
        jax.lax.map(primitive.bind, batching.bdim_at_front(args[0], axes[0], args[0].shape[axes[0]])), 0)
    return primitive


@lru_cache(maxsize=32)
def _direct_operator(atoms_key, basis_key, cart, cutoff):
    atoms = np.asarray(atoms_key, dtype=np.int32)
    basis = np.asarray(basis_key, dtype=np.int32)
    primitive = core.Primitive("gradscf_direct_jk")

    def apply(env, density):
        register_integrals()
        call = jax.ffi.ffi_call(
            "gradscf_direct_jk_cpu_v1",
            jax.ShapeDtypeStruct((2, *density.shape), jnp.float64),
            vmap_method="sequential",
        )
        return call(
            jnp.asarray(atoms),
            jnp.asarray(basis),
            env,
            density,
            cart=np.int32(cart),
            cutoff=np.float64(cutoff),
        )

    primitive.def_impl(apply)
    primitive.def_abstract_eval(
        lambda env, d: jax.core.ShapedArray((2, *d.shape), jnp.float64)
    )
    mlir.register_lowering(
        primitive, mlir.lower_fun(apply, multiple_results=False), platform="cpu"
    )

    # Keep the shell engine on CPU when response arrays live on CUDA. Only
    # bounded density/J/K batches cross devices, never an AO ERI tensor.
    host_apply = jax.jit(apply)

    def host_callback(env, density):
        cpu = jax.local_devices(backend="cpu")[0]
        return np.asarray(host_apply(jax.device_put(env, cpu), jax.device_put(density, cpu)))

    def cuda_apply(env, density):
        shape = jax.ShapeDtypeStruct((2, *density.shape), jnp.float64)
        return jax.pure_callback(host_callback, shape, env, density, vmap_method="sequential")

    mlir.register_lowering(primitive, mlir.lower_fun(cuda_apply, multiple_results=False), platform="cuda")

    def jvp(primals, tangents):
        env, d = primals
        de, dd = tangents
        basis_zero = None if isinstance(de, ad.Zero) else _zero_basis_tangent().bind(de)
        value = primitive.bind(env, d)
        tangent = (
            ad.Zero(jax.core.ShapedArray(value.shape, value.dtype))
            if isinstance(dd, ad.Zero)
            else primitive.bind(env, dd)
        )
        if basis_zero is not None:
            tangent = (jnp.zeros_like(value) if isinstance(tangent, ad.Zero) else tangent) + basis_zero
        return value, tangent

    def transpose(cot, env, d):
        if ad.is_undefined_primal(env):
            raise NotImplementedError(
                "Direct J/K basis/geometry VJP is not implemented."
            )
        if not ad.is_undefined_primal(d):
            return None, None
        if isinstance(cot, ad.Zero):
            return None, ad.Zero(d.aval)
        nset = cot.shape[1]
        # Both Coulomb and exchange are self-adjoint linear density maps.
        value = primitive.bind(env, jnp.concatenate((cot[0], cot[1]), axis=0))
        return None, value[0, :nset] + value[1, nset:]

    def batch(args, axes):
        size = next(x.shape[a] for x, a in zip(args, axes) if a is not None)
        arrays = tuple(batching.bdim_at_front(x, a, size) for x, a in zip(args, axes))
        return jax.lax.map(lambda xs: primitive.bind(*xs), arrays), 0

    ad.primitive_jvps[primitive] = jvp
    ad.primitive_transposes[primitive] = transpose
    batching.primitive_batchers[primitive] = batch
    return primitive


def direct_jk(atm, bas, env, density, *, cart, cutoff=0.0):
    d = jnp.asarray(density)
    n = d.shape[-1]
    if d.shape[-2:] != (n, n) or d.dtype not in (jnp.float64, jnp.complex128):
        raise ValueError("Direct J/K requires (...,n,n) float64/complex128 density.")
    env = jnp.asarray(env)
    if env.dtype != jnp.float64:
        raise ValueError("Direct J/K requires float64 environment.")
    operator = _direct_operator(
        tuple(map(tuple, np.asarray(atm).tolist())),
        tuple(map(tuple, np.asarray(bas).tolist())),
        bool(cart),
        float(cutoff),
    )
    flat = d.reshape(-1, n, n)
    count = len(flat)
    real = (
        jnp.concatenate((flat.real, flat.imag), axis=0) if jnp.iscomplexobj(d) else flat
    )
    out = operator.bind(env, real)
    if jnp.iscomplexobj(d):
        out = out[:, :count] + 1j * out[:, count:]
    out = out.reshape(2, *d.shape)
    return out[0], out[1]


@lru_cache(maxsize=1)
def _packed_operator():
    from ...molecular.jk import _build_jk_from_packed_jax

    primitive = core.Primitive("gradscf_packed_jk")
    reference = lambda e, d: jnp.stack(_build_jk_from_packed_jax(e, d))

    def apply(eri, density):
        register_integrals()
        call = jax.ffi.ffi_call(
            "gradscf_packed_jk_cpu_v1",
            jax.ShapeDtypeStruct((2, *density.shape), jnp.float64),
            vmap_method="sequential",
        )
        return call(eri, density)

    primitive.def_impl(apply)
    primitive.def_abstract_eval(
        lambda e, d: jax.core.ShapedArray((2, *d.shape), jnp.float64)
    )
    mlir.register_lowering(primitive, mlir.lower_fun(reference, multiple_results=False))
    mlir.register_lowering(
        primitive, mlir.lower_fun(apply, multiple_results=False), platform="cpu"
    )

    def jvp(primals, tangents):
        e, d = primals
        de, dd = tangents
        value = primitive.bind(e, d)
        tangent = ad.Zero(jax.core.ShapedArray(value.shape, value.dtype))
        if not isinstance(de, ad.Zero):
            tangent = primitive.bind(de, d)
        if not isinstance(dd, ad.Zero):
            part = primitive.bind(e, dd)
            tangent = part if isinstance(tangent, ad.Zero) else tangent + part
        return value, tangent

    def transpose(cot, e, d):
        if ad.is_undefined_primal(e):
            if ad.is_undefined_primal(d):
                raise ValueError("A bilinear J/K transpose needs one known primal.")
            if isinstance(cot, ad.Zero):
                return ad.Zero(e.aval), None
            # The ERI cotangent uses the portable bounded scatter path;
            # density derivatives use the fast self-adjoint native operator.
            _, pullback = jax.vjp(
                lambda x: reference(x, d), jnp.zeros(e.aval.shape, e.aval.dtype)
            )
            return pullback(cot)[0], None
        if not ad.is_undefined_primal(d):
            return None, None
        if isinstance(cot, ad.Zero):
            return None, ad.Zero(d.aval)
        count = cot.shape[1]
        value = primitive.bind(e, jnp.concatenate((cot[0], cot[1]), axis=0))
        return None, value[0, :count] + value[1, count:]

    def batch(args, axes):
        size = next(x.shape[a] for x, a in zip(args, axes) if a is not None)
        arrays = tuple(batching.bdim_at_front(x, a, size) for x, a in zip(args, axes))
        return jax.lax.map(lambda xs: primitive.bind(*xs), arrays), 0

    ad.primitive_jvps[primitive] = jvp
    ad.primitive_transposes[primitive] = transpose
    batching.primitive_batchers[primitive] = batch
    return primitive


def packed_jk(eri, density):
    """Native s8 value/density AD; ERI VJP uses bounded JAX scatters."""
    if eri.ndim != 1:
        raise ValueError("The native packed J/K operator requires s8 ERIs.")
    n = density.shape[-1]
    flat = density.reshape(-1, n, n)
    count = len(flat)
    d = (
        jnp.concatenate((flat.real, flat.imag), axis=0)
        if jnp.iscomplexobj(flat)
        else flat
    )
    out = _packed_operator().bind(eri, d)
    if jnp.iscomplexobj(flat):
        out = out[:, :count] + 1j * out[:, count:]
    out = out.reshape(2, *density.shape)
    return out[0], out[1]
