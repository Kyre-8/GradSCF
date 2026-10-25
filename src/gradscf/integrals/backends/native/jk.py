"""Native direct and packed J/K operators and bound direct-SCF providers."""

from functools import lru_cache
from dataclasses import dataclass
from numbers import Integral
import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.interpreters import ad, batching, mlir
from ..._native import register_integrals


def native_jk_workspace_bytes(provider, workers, cutoff):
    """Bound one native call's batch-independent optimizer and shell arrays.

    The pinned native ABI uses float64, int32 and 64-bit pointers. Return
    None for dimension-only placeholders without a complete shell topology.
    SCF and response share this estimate; no integrals are evaluated here.
    """
    topology = provider.plan.topology
    names = ("angular_momenta", "primitive_counts", "contraction_counts",
             "nuclear_charges", "cart")
    if any(not hasattr(topology, name) for name in names):
        return None
    angular = topology.angular_momenta
    primitive = topology.primitive_counts
    contractions = topology.contraction_counts
    shells = len(angular)
    workers = min(workers, shells)
    total_primitive = sum(primitive)
    total_coefficients = sum(p*c for p, c in zip(primitive, contractions))
    max_l, max_p = max(angular), max(primitive)
    cart_dims = tuple((l+1)*(l+2)//2 for l in angular)
    max_cart = max(f*c for f, c in zip(cart_dims, contractions))
    max_output = max(c*(f if topology.cart else 2*l+1)
                     for l, f, c in zip(angular, cart_dims, contractions))
    nf, nc = max(cart_dims)**4, max_cart**4
    g_size = (2*max_l+1)**3 * (max_l+1)**2
    pair_scratch = (10*max_p**2 + 4*max(p*c for p, c in zip(primitive, contractions))
                    + 8*max_p + 3*nf)
    cache = max(6*g_size + nf + 3*nc + pair_scratch, nc + 4*nf)
    # Retained vector capacities can exceed the current shell's logical size.
    shapes = len(set(zip(angular, primitive, contractions)))
    scratch = 16*(max_output**4 + cache) + 128*shapes**4
    # The shared geometry family has up to nine output components. Derivative
    # angular increments total two: either +2 on one leg or +1 on two legs.
    # Include that shell/cache envelope as well as the ordinary value kernel.
    cart0 = max(cart_dims)
    cart1, cart2 = (max_l+2)*(max_l+3)//2, (max_l+3)*(max_l+4)//2
    derivative_nf = max(cart2*cart0**3, cart1**2*cart0**2)
    derivative_g = (2*max_l+3)**3*(max_l+2)**2
    derivative_cache = max(6*derivative_g+9*derivative_nf+3*nc+pair_scratch,
                           nc+4*derivative_nf)
    scratch = max(scratch, 16*(9*max_output**4+derivative_cache)+128*shapes**4)
    allowed_l = min(max_l, 6)
    cumulative_cart = (allowed_l+1)*(allowed_l+2)*(allowed_l+3)//6
    optimizer = (48 + 12*cumulative_cart**4 + 8*(max_l+1)*16**3
                 + 24*shells + 12*total_primitive + 4*total_coefficients)
    if total_primitive <= 2048:
        optimizer += 40*total_primitive**2 + 8*shells**2
    atoms = len(topology.nuclear_charges) + shells
    tables = (4*(6*atoms + 8*shells + shells+1)
              + 8*(20 + 3*atoms + total_primitive + total_coefficients))
    bounds = 8*shells**2 if cutoff > 0 else 0
    # Coordinate VJPs reduce one ENV-sized vector per worker. Charging another
    # table bounds those vectors without introducing worker-sized J/K copies.
    return optimizer + bounds + (2*workers+1)*tables + workers*scratch


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NativeDirectBasis:
    """Native AO action with static topology and dynamic basis parameters.

    J/K density response and first/second coordinate response are supported.
    Coordinate AD requires zero screening cutoff; primitive exponent and
    coefficient derivatives remain explicitly unsupported.
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

    def get_jk(self, density, *, screening_threshold=0.0, workers=1,
               with_j=True, with_k=True, omega=0.):
        from ...plan import _validate_omega

        omega = _validate_omega(omega)
        range_options = {"omega": omega} if omega else {}
        return self.plan.get_jk(
            self.parameters, density, screening_threshold=screening_threshold,
            workers=workers, with_j=with_j, with_k=with_k, **range_options,
        )

    def response_eri_pair_matrix(self):
        return self.plan.evaluate("eri", self.parameters, aosym="s4")

    @property
    def eri_cache_nbytes(self):
        """Exact s8 storage; no numerical integrals are held by the provider."""
        pairs = self.nao * (self.nao + 1) // 2
        return 8 * pairs * (pairs + 1) // 2

    def prepare_jk(self, *, cache_bytes=0, screening_threshold=0.0, workers=1,
                   with_j=True, with_k=True, omega=0.):
        """Bind one CPU response action, optionally retaining exact s8 ERIs.

        The caller pays for both the cache and contraction workspaces. The
        returned closure owns the cache; neither the reference nor a global
        cache is mutated. Traced coordinates use the direct geometry action.
        Exponent/coefficient AD has the same strict boundary as direct J/K.
        Nonzero omega always uses direct J/K; the cached s8 values are ordinary
        Coulomb integrals and must not be reused for a long-range operator.
        """
        from ...plan import _validate_omega

        omega = _validate_omega(omega)
        range_options = {"omega": omega} if omega else {}
        direct = lambda density: self.get_jk(
            density, screening_threshold=screening_threshold, workers=workers,
            with_j=with_j, with_k=with_k, **range_options)
        coordinates_traced = any(isinstance(value, jax.core.Tracer) for value in
                                 jax.tree.leaves((self.parameters.centers, self.parameters.nuclear_coords)))
        if (omega or coordinates_traced or not (with_j or with_k) or cache_bytes < self.eri_cache_nbytes
                or screening_threshold > 0):
            return direct

        @jax.custom_jvp
        def evaluate(parameters):
            return self.plan.evaluate("eri", parameters, aosym="s8")

        def jvp(primals, tangents):
            value = evaluate(primals[0])
            zero = jnp.asarray(0., dtype=value.dtype)
            for tangent in jax.tree.leaves(tangents[0]):
                if not isinstance(tangent, (ad.Zero, jax.custom_derivatives.SymbolicZero)):
                    zero = zero + _zero_basis_tangent().bind(tangent)
            return value, jnp.zeros_like(value) + zero

        evaluate.defjvp(jvp, symbolic_zeros=True)
        eri = evaluate(self.parameters)
        # If a prepared CPU action is later moved to CUDA, retain the bounded
        # CPU shell callback instead of the generic packed GPU row gather.
        return lambda density: jax.lax.platform_dependent(
            density, cpu=lambda d: packed_jk(eri, d, with_j=with_j, with_k=with_k),
            default=lambda d: self.get_jk(d, screening_threshold=screening_threshold,
                                          workers=min(workers, 8),
                                          with_j=with_j, with_k=with_k))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ProjectedNativeDirectBasis:
    """Exact contracted J/K with coordinate and contraction-matrix response.

    JAX owns both contractions with ``transform``. Primitive AO coordinates
    retain their native geometry response; primitive exponents/coefficients
    are explicitly unsupported differentiation inputs.
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

    def get_jk(self, density, *, screening_threshold=0.0, workers=1,
               with_j=True, with_k=True, omega=0.):
        from ...plan import _validate_omega

        omega = _validate_omega(omega)
        range_options = {"omega": omega} if omega else {}
        if not (with_j or with_k):
            return jnp.zeros_like(density), jnp.zeros_like(density)
        transform = jnp.asarray(self.transform)
        primitive_density = jnp.einsum(
            "pi,...ij,qj->...pq", transform, density, transform
        )
        primitive_j, primitive_k = self.primitive_basis.plan.get_jk(
            self.primitive_basis.parameters, primitive_density,
            screening_threshold=screening_threshold,
            workers=workers, with_j=with_j, with_k=with_k, **range_options,
        )
        project = lambda matrix: jnp.einsum(
            "pi,...pq,qj->...ij", transform, matrix, transform
        )
        return project(primitive_j), project(primitive_k)


@lru_cache(maxsize=4)
def _zero_basis_tangent(message="Direct J/K supports density AD only; basis/geometry derivatives are not implemented."):
    """Accept materialized zero tangents, never an active basis derivative.

    custom_root can instantiate symbolic zeros in closed-over constants.
    This partial linear operation is defined only on the zero tangent; its
    transpose explicitly rejects differentiation with respect to the basis.
    """
    primitive = core.Primitive("gradscf_direct_jk_zero_basis_tangent")

    def check(tangent):
        if np.any(np.asarray(tangent) != 0):
            raise NotImplementedError(message)
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
            raise NotImplementedError(message)
        return (ad.Zero(tangent.aval) if ad.is_undefined_primal(tangent) else None,)

    ad.primitive_jvps[primitive] = jvp
    ad.primitive_transposes[primitive] = transpose
    batching.primitive_batchers[primitive] = lambda args, axes: (
        jax.lax.map(primitive.bind, batching.bdim_at_front(args[0], axes[0], args[0].shape[axes[0]])), 0)
    return primitive


@lru_cache(maxsize=32)
def _direct_operator(atoms_key, basis_key, cart, cutoff, workers, with_j, with_k):
    from .autodiff.common import register_cpu_product

    atoms = np.asarray(atoms_key, dtype=np.int32)
    basis = np.asarray(basis_key, dtype=np.int32)
    primitive = core.Primitive("gradscf_direct_jk")
    nops = int(with_j) + int(with_k)

    def apply(env, density):
        register_integrals()
        call = jax.ffi.ffi_call(
            "gradscf_direct_jk_cpu_v1",
            jax.ShapeDtypeStruct((nops, *density.shape), jnp.float64),
            vmap_method="sequential",
        )
        return call(
            jnp.asarray(atoms),
            jnp.asarray(basis),
            env,
            density,
            cart=np.int32(cart),
            cutoff=np.float64(cutoff),
            workers=np.int32(workers),
            with_j=np.int32(with_j),
            with_k=np.int32(with_k),
        )

    # Keep the shell engine on CPU when response arrays live on CUDA. Only
    # bounded density/J/K batches cross devices, never an AO ERI tensor.
    register_cpu_product(primitive, apply, lambda env, d: (nops, *d.shape))

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
        return None, _density_transpose(lambda d: primitive.bind(env, d), cot)

    def batch(args, axes):
        size = next(x.shape[a] for x, a in zip(args, axes) if a is not None)
        arrays = tuple(batching.bdim_at_front(x, a, size) for x, a in zip(args, axes))
        return jax.lax.map(lambda xs: primitive.bind(*xs), arrays), 0

    ad.primitive_jvps[primitive] = jvp
    ad.primitive_transposes[primitive] = transpose
    batching.primitive_batchers[primitive] = batch
    return primitive


@lru_cache(maxsize=32)
def _direct_geometry_value(atoms_key, basis_key, cart, cutoff, workers, with_j, with_k):
    from functools import partial
    from .autodiff.common import batch_product
    from .autodiff.direct_jk import geometry_products

    atoms = np.asarray(atoms_key, dtype=np.int32)
    offsets = atoms[:, 1, None] + np.arange(3)[None, :]
    operator = _direct_operator(atoms_key, basis_key, cart, cutoff, workers, with_j, with_k)
    geometry = geometry_products(atoms_key, basis_key, cart, with_j, with_k, workers=workers)

    def pack(env, coordinates):
        return env.at[offsets].set(coordinates, unique_indices=True)

    value = core.Primitive("gradscf_direct_jk_coordinates")

    def apply(env, coordinates, density):
        return operator.bind(pack(env, coordinates), density)

    value.def_impl(apply)
    value.def_abstract_eval(lambda env, coordinates, density:
        jax.core.ShapedArray((int(with_j)+int(with_k), *density.shape), jnp.float64))
    mlir.register_lowering(value, mlir.lower_fun(apply, multiple_results=False))

    def value_jvp(primals, tangents):
        env, coordinates, density = primals
        denv, dcoordinates, ddensity = tangents
        primal = value.bind(env, coordinates, density)
        terms = []
        if not isinstance(denv, ad.Zero):
            zero = _zero_basis_tangent(
                "Direct J/K basis exponent/coefficient derivatives are not implemented").bind(denv)
            terms.append(jnp.zeros_like(primal) + zero)
        if not isinstance(ddensity, ad.Zero):
            # The recursive value retains the geometry response of a density
            # tangent under mixed and higher differentiation.
            terms.append(value.bind(env, coordinates, ddensity))
        if not isinstance(dcoordinates, ad.Zero):
            direction = pack(jnp.zeros_like(env), dcoordinates)
            if cutoff > 0:
                zero = _zero_basis_tangent(
                    "Screened direct J/K geometry AD requires cutoff=0").bind(direction)
                terms.append(jnp.zeros_like(primal) + zero)
            else:
                terms.append(geometry.bind(pack(env, coordinates), density, direction))
        return primal, sum(terms) if terms else ad.Zero.from_primal_value(primal)

    def transpose(cotangent, env, coordinates, density):
        if ad.is_undefined_primal(env) or ad.is_undefined_primal(coordinates):
            raise NotImplementedError("A direct J/K density transpose requires its primal basis and coordinates")
        if not ad.is_undefined_primal(density):
            return None, None, None
        if isinstance(cotangent, ad.Zero):
            return None, None, ad.Zero(density.aval)
        # Keep coordinates explicit in the transposed action. Returning a raw
        # ENV action here would lose geometry response of a density VJP.
        return None, None, _density_transpose(
            lambda d: value.bind(env, coordinates, d), cotangent)

    ad.primitive_jvps[value] = value_jvp
    ad.primitive_transposes[value] = transpose
    batching.primitive_batchers[value] = partial(batch_product, value)
    return value.bind


def _density_transpose(apply, cotangent):
    """J and K are self-adjoint linear maps on nonsymmetric AO densities."""
    if cotangent.shape[0] == 1:
        return apply(cotangent[0])[0]
    count = cotangent.shape[1]
    value = apply(jnp.concatenate((cotangent[0], cotangent[1]), axis=0))
    return value[0, :count] + value[1, count:]


def _validate_selection(with_j, with_k):
    if not all(isinstance(flag, (bool, np.bool_)) for flag in (with_j, with_k)):
        raise ValueError("with_j and with_k must be static booleans.")


def direct_jk(atm, bas, env, density, *, cart, cutoff=0.0, workers=1,
              with_j=True, with_k=True, coordinates=None):
    """Contract J/K; IntegralPlan supplies coordinates separately for geometry AD.

    A raw packed ENV remains a fixed-basis input. Separating fixed ENV and
    coordinate leaves prevents unsupported exponent/coefficient VJPs from
    being silently returned as zeros.
    """
    _validate_selection(with_j, with_k)
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    d = jnp.asarray(density)
    n = d.shape[-1]
    if d.shape[-2:] != (n, n) or d.dtype not in (jnp.float64, jnp.complex128):
        raise ValueError("Direct J/K requires (...,n,n) float64/complex128 density.")
    env = jnp.asarray(env)
    if env.dtype != jnp.float64:
        raise ValueError("Direct J/K requires float64 environment.")
    if not (with_j or with_k):
        return jnp.zeros_like(d), jnp.zeros_like(d)
    operator = _direct_operator(
        tuple(map(tuple, np.asarray(atm).tolist())),
        tuple(map(tuple, np.asarray(bas).tolist())),
        bool(cart),
        float(cutoff),
        int(workers),
        bool(with_j), bool(with_k),
    )
    flat = d.reshape(-1, n, n)
    count = len(flat)
    real = (
        jnp.concatenate((flat.real, flat.imag), axis=0) if jnp.iscomplexobj(d) else flat
    )
    if coordinates is None:
        out = operator.bind(env, real)
    else:
        coordinates = jnp.asarray(coordinates, dtype=env.dtype)
        if coordinates.shape != (len(atm), 3):
            raise ValueError("Direct J/K coordinates must contain every nuclear and basis-center atom.")
        out = _direct_geometry_value(
            tuple(map(tuple, np.asarray(atm).tolist())),
            tuple(map(tuple, np.asarray(bas).tolist())), bool(cart), float(cutoff),
            int(workers), bool(with_j), bool(with_k))(env, coordinates, real)
    if jnp.iscomplexobj(d):
        out = out[:, :count] + 1j * out[:, count:]
    out = out.reshape(int(with_j) + int(with_k), *d.shape)
    return (out[0] if with_j else jnp.zeros_like(d),
            out[-1] if with_k else jnp.zeros_like(d))


@lru_cache(maxsize=4)
def _packed_operator(with_j=True, with_k=True):
    from ...molecular.jk import _build_jk_from_packed_jax
    from .autodiff.common import register_eager_dispatch

    primitive = core.Primitive("gradscf_packed_jk")
    nops = int(with_j) + int(with_k)

    def reference(e, d):
        coulomb, exchange = _build_jk_from_packed_jax(e, d)
        return jnp.stack([v for v, include in ((coulomb, with_j), (exchange, with_k))
                          if include])

    def apply(eri, density):
        register_integrals()
        call = jax.ffi.ffi_call(
            "gradscf_packed_jk_cpu_v1",
            jax.ShapeDtypeStruct((nops, *density.shape), jnp.float64),
            vmap_method="sequential",
        )
        return call(eri, density, with_j=np.int32(with_j), with_k=np.int32(with_k))

    # Eager calls must honor the CPU FFI / portable CUDA lowering selection.
    register_eager_dispatch(primitive)
    primitive.def_abstract_eval(
        lambda e, d: jax.core.ShapedArray((nops, *d.shape), jnp.float64)
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
        return None, _density_transpose(lambda d: primitive.bind(e, d), cot)

    def batch(args, axes):
        size = next(x.shape[a] for x, a in zip(args, axes) if a is not None)
        arrays = tuple(batching.bdim_at_front(x, a, size) for x, a in zip(args, axes))
        return jax.lax.map(lambda xs: primitive.bind(*xs), arrays), 0

    ad.primitive_jvps[primitive] = jvp
    ad.primitive_transposes[primitive] = transpose
    batching.primitive_batchers[primitive] = batch
    return primitive


def packed_jk(eri, density, *, with_j=True, with_k=True):
    """Native s8 value/density AD; ERI VJP uses bounded JAX scatters."""
    if eri.ndim != 1:
        raise ValueError("The native packed J/K operator requires s8 ERIs.")
    _validate_selection(with_j, with_k)
    if not (with_j or with_k):
        return jnp.zeros_like(density), jnp.zeros_like(density)
    n = density.shape[-1]
    flat = density.reshape(-1, n, n)
    count = len(flat)
    d = (
        jnp.concatenate((flat.real, flat.imag), axis=0)
        if jnp.iscomplexobj(flat)
        else flat
    )
    out = _packed_operator(bool(with_j), bool(with_k)).bind(eri, d)
    if jnp.iscomplexobj(flat):
        out = out[:, :count] + 1j * out[:, count:]
    out = out.reshape(int(with_j) + int(with_k), *density.shape)
    return (out[0] if with_j else jnp.zeros_like(density),
            out[-1] if with_k else jnp.zeros_like(density))


def _selected_jk(provider, density, *, with_j=True, with_k=True,
                 screening_threshold=0., workers=1, omega=0.):
    """Select native operators while preserving legacy provider signatures."""
    from ...plan import _validate_omega

    _validate_selection(with_j, with_k)
    omega = _validate_omega(omega)
    if type(provider) in (NativeDirectBasis, ProjectedNativeDirectBasis):
        range_options = {"omega": omega} if omega else {}
        return provider.get_jk(density, with_j=with_j, with_k=with_k,
                               screening_threshold=screening_threshold, workers=workers,
                               **range_options)
    if omega:
        raise NotImplementedError("A nonzero omega requires a native long-range J/K provider.")
    j_mat, k_mat = provider.get_jk(density)
    return (j_mat if with_j else jnp.zeros_like(j_mat),
            k_mat if with_k else jnp.zeros_like(k_mat))
