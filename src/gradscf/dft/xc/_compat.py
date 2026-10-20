"""Upstream parameter bindings and stable JAX-kernel compatibility fixes."""
from functools import lru_cache, update_wrapper
from types import FunctionType

import numpy as np
import ctypes
import inspect
import re


@lru_cache(maxsize=16)
def pw_parameter_factory(factory):
    """Preserve PW's three parameter sets across legacy ``get_p`` conversion.

    The jax_xc 0.0.9 converter observed in validation repeats element zero of
    each PW vector. The factory itself supplies all 22 correct scalar values.
    Reconstruct vectors from those actual arguments, including user overrides,
    while retaining the upstream generated formula and its derivative graph.
    A private globals dictionary avoids monkey-patching the installed module.
    """
    if not isinstance(factory, FunctionType) or "get_p" not in factory.__globals__:
        # Factories that do not use the legacy converter have no such boundary.
        return factory
    upstream_get_p = factory.__globals__["get_p"]
    fields = ("pp", "a", "alpha1", "beta1", "beta2", "beta3", "beta4")

    def get_p(name, polarized, *values):
        p = upstream_get_p(name, polarized, *values)
        if name != "lda_c_pw":
            return p
        if len(values) != 22 or not hasattr(p, "params"):
            raise RuntimeError("Unsupported upstream lda_c_pw parameter layout")
        if not all(hasattr(p.params, field) for field in (*fields, "fz20")):
            raise RuntimeError("Unsupported upstream lda_c_pw parameter fields")
        arrays = {field: np.asarray(values[3*i:3*i+3], dtype=np.float64)
                  for i, field in enumerate(fields)}
        return p._replace(params=p.params._replace(**arrays, fz20=np.asarray(values[21], dtype=np.float64)))

    namespace = dict(factory.__globals__, get_p=get_p)
    wrapped = FunctionType(factory.__code__, namespace, factory.__name__,
                           factory.__defaults__, factory.__closure__)
    wrapped.__kwdefaults__ = factory.__kwdefaults__
    return update_wrapper(wrapped, factory)
"""Factored local omegaB97 form used only by the omegaB97X-D adapter.

Chai and Head-Gordon, JCP 128, 084106 (2008), Eqs. 6--7 and 11--26:
https://doi.org/10.1063/1.2834918. Coefficients remain supplied by the upstream
omegaB97X-D factory (PCCP 10, 6615, https://doi.org/10.1039/B810189B).

The generated jax_xc expression expands powers of the bounded B97 variable
into separate large density reciprocals. Their third derivatives overflow
at active molecular quadrature points. Horner evaluation below is the same
polynomial, without these intermediate powers. The SR-LDA attenuation uses
the same 1.35 branch and eight-term asymptotic series as LibXC/jax_xc.
"""
import jax
import jax.numpy as jnp


def _polynomial(coefficients, u):
    value = coefficients[-1]
    for coefficient in coefficients[-2::-1]:
        value = coefficient + u*value
    return value


def _bounded_gradient(q):
    # The second form avoids cancellation between O(1/q) terms in the
    # derivative of q/(1+q) when the reduced gradient is large.
    return jnp.where(q > 1., 1-1/(1+q), q/(1+q))


def _pw92_component(rs, a, alpha1, beta1, beta2, beta3, beta4):
    """PW92 Eq. 10, using the parameters present in LibXC's WB97 kernel."""
    root = jnp.sqrt(rs)
    denominator = 2*a*root*(beta1 + root*(beta2 + root*(beta3 + root*beta4)))
    return -2*a*(1 + alpha1*rs)*jnp.log1p(1/denominator)


def _pw92_polarized(rs):
    return _pw92_component(rs, .015545, .20548, 14.1189, 6.1977, 3.3662, .62517)


def _pw92_total(rs, zeta, zeta_threshold):
    unpolarized = _pw92_component(rs, .031091, .21370, 7.5957, 3.5876, 1.6382, .49294)
    polarized = _pw92_polarized(rs)
    stiffness = -_pw92_component(rs, .016887, .11125, 10.357, 3.6231, .88026, .49671)
    spin_factors = jnp.maximum(jnp.asarray([1+zeta, 1-zeta]), zeta_threshold)
    f = (jnp.sum(spin_factors*jnp.cbrt(spin_factors))-2)/(2**(4/3)-2)
    # Preserve the PW92/LibXC parameter, including its published rounding.
    # Replacing it by the analytic limit would change this named functional.
    f_second_zero = 1.709921
    zeta4 = zeta**4
    return unpolarized + f*(zeta4*(polarized-unpolarized)
                            + (1-zeta4)*stiffness/f_second_zero)


def _sr_attenuation(a):
    # Keep both branches in-domain under vmap and higher-order differentiation.
    # At equality the selected series gets its right-branch derivative. The
    # upstream nested (> then >=) clamps accidentally gave zero AD there.
    small = jnp.where(a < 1.35, a, 1.35)
    large = jnp.where(a >= 1.35, a, 1.35)
    inverse_square = (1/large)**2
    series = inverse_square*_polynomial(
        (1/36, -1/960, 1/26880, -1/829440, 1/28385280,
         -1/1073479680, 1/44590694400, -1/2021444812800), inverse_square)
    exponential = jnp.exp(-1/(4*small**2))
    direct = 1-(8/3)*small*(jnp.sqrt(jnp.pi)*jax.lax.erf(1/(2*small))
                            + 2*small*(exponential-1.5-2*small**2*(exponential-1)))
    return jnp.where(a >= 1.35, series, direct)


def _polarized_epsilon(p, density, sigma):
    total = jnp.sum(density)
    spin_factors = 2*density/total
    active = (density > p.dens_threshold) & (spin_factors > p.zeta_threshold)
    # Retain every positive spin density, including values below the masking
    # threshold. Only an exactly absent spin needs a finite unused branch.
    safe_density = jnp.where(density > 0., density, total/2)
    root = jnp.cbrt(safe_density)
    reduced_squared = sigma/(safe_density*root)**2
    u_x = _bounded_gradient(.004*reduced_squared)
    u_ss = _bounded_gradient(.2*reduced_squared)
    u_ab_numerator = .003*jnp.sum(reduced_squared)
    u_ab = _bounded_gradient(u_ab_numerator)
    a = p.cam_omega/(2*jnp.cbrt(6*jnp.pi**2)*root)
    exchange = (-1.5*jnp.cbrt(3/(4*jnp.pi))*safe_density*root
                *_sr_attenuation(a)*_polynomial(p.params.c_x, u_x))
    rs_spin = jnp.cbrt(3/(4*jnp.pi))/root
    same_spin = safe_density*_pw92_polarized(rs_spin)
    exchange = jnp.where(active, exchange, 0.)
    same_spin = jnp.where(active, same_spin, 0.)
    rs = jnp.cbrt(3/(4*jnp.pi*total))
    zeta = (density[0]-density[1])/total
    opposite_spin = total*_pw92_total(rs, zeta, p.zeta_threshold)-jnp.sum(same_spin)
    correlation = (jnp.sum(same_spin*_polynomial(p.params.c_ss, u_ss))
                   + opposite_spin*_polynomial(p.params.c_ab, u_ab))
    return (jnp.sum(exchange)+correlation)/total


def invoke(p, rho, r, mo=None):
    """Retain the upstream density-function interface and density threshold."""
    from jax_xc.impl.utils import rho_to_arguments

    density, sigma = rho_to_arguments(p, rho, r, mo)
    if p.nspin == 1:
        density = jnp.asarray([density/2, density/2])
        sigma = jnp.asarray([sigma/4, sigma/4])
    else:
        sigma = jnp.asarray([sigma[0], sigma[2]])
    return jax.lax.cond(jnp.sum(density) < p.dens_threshold,
                        lambda: jnp.asarray(0., dtype=density.dtype),
                        lambda: _polarized_epsilon(p, density, sigma))


def _exp1_asymptotic(x):
    inverse = 1/jnp.maximum(x, 50.)
    power = inverse
    value, derivative, coefficient = inverse, -inverse**2, 1.
    for index in range(1, 17):
        coefficient *= -index
        power *= inverse
        value += coefficient*power
        derivative -= (index+1)*coefficient*power*inverse
    return value, derivative


def _exp1_series(x):
    def series_term(index, state):
        term, total = state
        term = -x*term/index
        return term, total+term/index
    return jax.lax.fori_loop(1, 33, series_term,
                             (jnp.ones_like(x), jnp.zeros_like(x)))[1]


@jax.custom_jvp
def scaled_exp1(x):
    """exp(x) E1(x), with an analytic JVP avoiding expn's nested-JIT defect.

    The asymptotic branch avoids 0*inf for large positive arguments. This is a
    special-function compatibility rule, independent of any XC functional.
    """
    x = jnp.asarray(x)
    small_x = jnp.minimum(jnp.maximum(x, jnp.finfo(x.dtype).tiny), 1.)
    series = _exp1_series(small_x)
    small = jnp.exp(small_x)*(-.577215664901532860606512090082402431-jnp.log(small_x)-series)
    # exp1's vectorized internal continued fraction is not safe on the unused
    # tiny-x branch. Only pass it [1,50]; use the convergent series below 1.
    large_x = jnp.minimum(jnp.maximum(x, 1.), 50.)
    direct = jnp.where(x <= 1., small,
                       jax.scipy.special.exp1(large_x)*jnp.exp(large_x))
    asymptotic, _ = _exp1_asymptotic(x)
    return jnp.where(x > 50., asymptotic, direct)


@scaled_exp1.defjvp
def _scaled_exp1_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    value = scaled_exp1(x)
    _, asymptotic_derivative = _exp1_asymptotic(x)
    derivative = jnp.where(x > 50., asymptotic_derivative,
                           value-1/jnp.maximum(x, jnp.finfo(jnp.asarray(x).dtype).tiny))
    return value, derivative*dx


def _erfcx_asymptotic(x):
    inverse = 1/jnp.maximum(x, 20.)
    power = inverse
    value, derivative, coefficient = inverse, -inverse**2, 1.
    for index in range(1, 13):
        coefficient *= -(2*index-1)/2
        power *= inverse**2
        value += coefficient*power
        derivative -= (2*index+1)*coefficient*power*inverse
    return value/jnp.sqrt(jnp.pi), derivative/jnp.sqrt(jnp.pi)


@jax.custom_jvp
def scaled_erfcx(x):
    """Stable exp(x²) erfc(x) on the nonnegative screened-exchange domain.

    WPBE evaluates erfcx near x=29.46 even in its unused branch. The upstream
    product overflows there. Both expressions below remain in domain under
    vmap/AD, and the large-x series avoids that inf*0 product entirely.
    """
    direct_x = jnp.minimum(x, 20.)
    direct = jnp.exp(direct_x**2)*jax.scipy.special.erfc(direct_x)
    asymptotic, _ = _erfcx_asymptotic(x)
    return jnp.where(x > 20., asymptotic, direct)


@scaled_erfcx.defjvp
def _scaled_erfcx_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    value = scaled_erfcx(x)
    _, asymptotic_derivative = _erfcx_asymptotic(x)
    derivative = jnp.where(x > 20., asymptotic_derivative,
                           2*x*value-2/jnp.sqrt(jnp.pi))
    return value, derivative*dx


def _exp1_plus_log(x):
    """Stable exp(x) E1(x)+log(x), including positive x far below 1e-30."""
    safe = jnp.maximum(x, jnp.finfo(jnp.asarray(x).dtype).tiny)
    small_x = jnp.minimum(safe, 1.)
    small = (-.577215664901532860606512090082402431*jnp.exp(small_x)
             -jnp.log(small_x)*jnp.expm1(small_x)
             -jnp.exp(small_x)*_exp1_series(small_x))
    large_x = jnp.maximum(safe, 1.)
    return jnp.where(safe <= 1., small, scaled_exp1(large_x)+jnp.log(large_x))


def _wpbe_auxiliary(s):
    square = s*s
    fourth = square*square
    hole = (.00979681*square+.0410834*fourth)/(1+.187440*fourth+.00120824*fourth*s+.0347188*fourth*square)
    return square, hole, 6.4753871*hole+.47965830, square*hole, .57786348+square*hole


def _wpbe_zero_enhancement(s):
    # Exact w=0 specialization of Libxc 6.0.0's existing Maple expression:
    # https://gitlab.com/libxc/libxc/-/raw/6.0.0/maple/gga_exc/gga_x_wpbeh.mpl
    # SHA256 429835517c3382aeb890b98245f3961e2c6b022c09ce5e5456afdaa5a724385b.
    # term3/4/5 and the w^1..w^8 terms vanish algebraically. Removing them
    # before AD avoids 0*inf from their tiny-gradient denominators. This is
    # the native WPBEH zero-omega limit, not a replacement by PBE exchange.
    a, b, c, d, e = 1.0161144, -.37170836, -.077215461, .57786348, -.051955731
    square, _, f, delta, auxiliary = _wpbe_auxiliary(s)
    large_s = jnp.maximum(s, .08)
    lsquare, _, lf, ldelta, laux = _wpbe_auxiliary(large_s)
    ga = (jnp.sqrt(jnp.pi)*(15*e+6*c*(1+lf*lsquare)*laux+4*b*laux**2+8*a*laux**3)
          /(16*laux**3.5)-3*jnp.pi/4*jnp.sqrt(a)*scaled_erfcx(jnp.sqrt(9*ldelta/(4*a))))
    gb = 15*jnp.sqrt(jnp.pi)*lsquare/(16*laux**3.5)
    eg_large = -(3*jnp.pi/4+ga)/gb
    ssquare = jnp.minimum(s, .08)**2
    eg_small = -.02628417880-.07117647788*ssquare+.08534541323*ssquare**2
    eg = jnp.where(s > .08, eg_large, eg_small)
    term1 = a/2*(_exp1_plus_log(9*delta/(4*a))-jnp.log(9/(4*a))-jnp.log(auxiliary))
    term2 = (auxiliary**2*b+auxiliary*c+2*e+auxiliary*square*c*f+2*square*eg)/(2*auxiliary**3)
    return -8/9*(term1+term2)


def _wpbe_zero_omega_invoke(p, rho, r, mo=None):
    from jax_xc.impl.utils import rho_to_arguments

    density, sigma = rho_to_arguments(p, rho, r, mo)
    threshold = p.dens_threshold
    if p.nspin == 1:
        density = jnp.asarray([density/2, density/2])
        sigma = jnp.asarray([sigma/4, sigma/4])
        threshold /= 2
    else:
        sigma = jnp.asarray([sigma[0], sigma[2]])
    total = jnp.sum(density)
    safe_total = jnp.where(total > 0., total, 1.)
    active = (density > threshold) & (2*density/safe_total > p.zeta_threshold)
    safe_density = jnp.where(active, density, 1.)
    safe_sigma = jnp.where(sigma > 0., sigma, 1e-60)
    reduced = jnp.sqrt(safe_sigma)/(2*jnp.cbrt(6*jnp.pi**2)*safe_density*jnp.cbrt(safe_density))
    capped = jnp.minimum(reduced, 15.)
    smooth = capped-jnp.log1p(jnp.exp(jnp.maximum(capped, 1.)-8.572844))
    # The old tiny-s floor protected log(delta) but flattened its transverse
    # curvature. The combined analytic log limit is finite, so retain the
    # physical small-gradient limit instead of differentiating that floor.
    s = jnp.where(reduced < 1., reduced,
                   jnp.where(reduced > 15., 8.572844, smooth))
    energy = -.75*jnp.cbrt(6/jnp.pi)*safe_density*jnp.cbrt(safe_density)*_wpbe_zero_enhancement(s)
    return jnp.sum(jnp.where(active, energy, 0.))/safe_total


@lru_cache(maxsize=128)
def _safe_kernel_invoke(module):
    """Clone one generated kernel namespace, retaining all upstream formulas."""
    namespace = dict(vars(module))
    for name, function in tuple(namespace.items()):
        if isinstance(function, FunctionType) and function.__globals__ is vars(module):
            clone = FunctionType(function.__code__, namespace, function.__name__,
                                 function.__defaults__, function.__closure__)
            clone.__kwdefaults__ = function.__kwdefaults__
            namespace[name] = clone
    namespace["xc_E1_scaled"] = scaled_exp1
    namespace["xc_erfcx"] = scaled_erfcx
    if module.__name__.rsplit(".", 1)[-1] == "gga_x_wpbeh":
        original_cond = namespace["lax_cond"]

        def positive_gradient_cond(condition, lower, value):
            # Libxc 6.0.0 Maple f_wpbeh floors s at 1e-15 to protect logs.
            # For positive omega all log arguments already have a positive
            # omega² term. Preserve every positive s, including its finite
            # zero-gradient curvature; retain safety at exactly absent s.
            # This constant occurs only at the three generated s-floor sites.
            # The static omega=0 graph is simplified separately above.
            if isinstance(lower, float) and lower == 1e-15:
                return jnp.where(value > 0., value, lower)
            return original_cond(condition, lower, value)

        namespace["lax_cond"] = positive_gradient_cond
    return namespace["invoke"]


def _invoke_compatible(p, rho, r, mo=None):
    from jax_xc import impl

    if p.maple_name == "gga_x_wpbeh" and not isinstance(p.cam_omega, jax.core.Tracer) and p.cam_omega == 0.:
        return _wpbe_zero_omega_invoke(p, rho, r, mo)
    if p.maple_name == "":
        return sum(coefficient*_invoke_compatible(child, rho, r, mo)
                   for child, coefficient in zip(p.func_aux, p.mix_coef))
    return _safe_kernel_invoke(getattr(impl, p.maple_name))(p, rho, r, mo)


class _XCHeader(ctypes.Structure):
    # The public xc_func_type prefix used by the upstream pylibxc binding.
    _fields_ = [("info", ctypes.c_void_p), ("nspin", ctypes.c_int),
                ("n_func_aux", ctypes.c_int), ("func_aux", ctypes.c_void_p),
                ("mix_coef", ctypes.POINTER(ctypes.c_double))]


class _B97MVParameters(ctypes.Structure):
    # Verified against the active parameter library, Libxc 6.0.0:
    # https://gitlab.com/libxc/libxc/-/raw/6.0.0/src/mgga_xc_b97mv.c
    # Source SHA256 4cdde56c8c75551ad7e28869f34e082f24ec9e2abeb53750b6fa94c7fcb790ff.
    # Read actual C arrays; no replacement coefficient table is maintained.
    _fields_ = [(name, ctypes.c_double*5) for name in ("c_x", "c_ss", "c_os")]


def _repair_b97mv_parameters(data, pointer):
    from jax_xc.libxc import structs
    from jax_xc.libxc.core import core

    version_function = ctypes.CFUNCTYPE(ctypes.c_char_p)(("xc_version_string", core))
    if version_function() != b"6.0.0":
        raise RuntimeError("B97M-V legacy parameter-array repair requires the verified Libxc 6.0.0 ABI.")
    functional = ctypes.cast(pointer, ctypes.POINTER(structs.xc_func_type)).contents
    if not functional.params:
        raise RuntimeError("B97M-V parameter storage is absent.")
    parameters = ctypes.cast(functional.params, ctypes.POINTER(_B97MVParameters)).contents
    for name in ("c_x", "c_ss", "c_os"):
        data["params"][name] = np.ctypeslib.as_array(getattr(parameters, name)).copy()


def _repair_mixing_tree(data, pointer):
    if data.get("name") == "mgga_xc_b97m_v":
        vectors = [np.asarray(data["params"][name]) for name in ("c_x", "c_ss", "c_os")]
        if all(vector.shape == (5,) and np.all(vector == vector[0]) for vector in vectors):
            _repair_b97mv_parameters(data, pointer)
    header = ctypes.cast(pointer, ctypes.POINTER(_XCHeader)).contents
    children = data.get("func_aux", ())
    if not children:
        return data
    if header.n_func_aux != len(children):
        raise RuntimeError("Unsupported upstream XC auxiliary-functional layout")
    if bool(header.mix_coef):
        data["mix_coef"] = np.asarray([header.mix_coef[index]
                                       for index in range(header.n_func_aux)])
    child_pointers = ctypes.cast(header.func_aux, ctypes.POINTER(ctypes.c_void_p))
    for child, child_pointer in zip(children, child_pointers[:header.n_func_aux]):
        _repair_mixing_tree(child, child_pointer)
    return data


@lru_cache(maxsize=128)
def _parameter_reader(upstream):
    """Repair the legacy C-array conversion while the source object is alive."""
    if not isinstance(upstream, FunctionType) or "libxc" not in upstream.__globals__:
        return upstream
    binding = upstream.__globals__["libxc"]

    class Reader:
        @staticmethod
        def get_p(pointer):
            return _repair_mixing_tree(binding.get_p(pointer), pointer)

    namespace = dict(upstream.__globals__, libxc=Reader)
    return FunctionType(upstream.__code__, namespace, upstream.__name__,
                        upstream.__defaults__, upstream.__closure__)


@lru_cache(maxsize=256)
def _default_parameter_values(name, polarized):
    import jax_xc
    factory = getattr(jax_xc, name, None)
    if not isinstance(factory, FunctionType) or "get_p" not in factory.__globals__:
        return ()
    values = []
    upstream = factory.__globals__["get_p"]

    def capture(key, spin, *arguments):
        values.extend(arguments)
        return upstream(key, spin, *arguments)

    clone = FunctionType(factory.__code__, dict(factory.__globals__, get_p=capture),
                         factory.__name__, factory.__defaults__, factory.__closure__)
    clone.__kwdefaults__ = factory.__kwdefaults__
    clone(polarized=polarized)
    return tuple(values)


@lru_cache(maxsize=256)
def _parameter_names(name):
    import jax_xc
    factory = getattr(jax_xc, name, None)
    return tuple(key for key in inspect.signature(factory).parameters if key != "polarized") if callable(factory) else ()


def _parameter_array(values):
    return (jnp.asarray(values) if any(isinstance(value, jax.core.Tracer) for value in values)
            else np.asarray(values, dtype=np.float64))


_VECTOR_PREFIXES = {"c_x": "_cx", "c_ss": "_css", "c_ab": "_cos",
                    "m08_a": "_a", "m08_b": "_b"}


def _repair_parameter_vectors(p, values, polarized, parameter_names=()):
    if hasattr(p, "params") and hasattr(p.params, "_fields"):
        arrays = {}
        for field in p.params._fields:
            array = np.asarray(getattr(p.params, field))
            if array.ndim != 1 or array.size < 2:
                continue
            prefix = _VECTOR_PREFIXES.get(field, "_"+field)
            group = [(int(key[len(prefix):]), index) for index, key in enumerate(parameter_names)
                     if key.startswith(prefix) and re.fullmatch(r"\d+", key[len(prefix):])]
            if len(group) == array.size:
                arrays[field] = _parameter_array([values[index] for _, index in sorted(group)])
                continue
            if not np.all(array == array[0]):
                continue
            # The converter repeats element zero. A unique occurrence in the
            # actual factory arguments identifies the complete contiguous C
            # vector without introducing a new set of physical coefficients.
            candidates = [index for index, value in enumerate(values)
                          if not isinstance(value, jax.core.Tracer) and value == array[0]
                          and index+array.size <= len(values)]
            if len(candidates) == 1:
                start = candidates[0]
                arrays[field] = np.asarray(values[start:start+array.size], dtype=np.float64)
        if p.maple_name == "hyb_gga_xc_wb97" and len(values) == 18:
            arrays.update({field: np.asarray(values[5*index:5*index+5])
                           for index, field in enumerate(("c_x", "c_ss", "c_ab"))})
        if p.maple_name == "hyb_mgga_x_m05" and len(values) == 14:
            arrays.update(cx=np.asarray(values[13]), csi_HF=np.asarray(values[12]))
        if arrays:
            p = p._replace(params=p.params._replace(**arrays))
    # These Libxc variants share the already-generated M06 correlation kernel;
    # the legacy binding omitted its Maple dispatch name while retaining the
    # complete M06 parameter layout. No new formula is introduced here.
    if p.name in {"mgga_c_m06", "mgga_c_m06_2x", "mgga_c_m06_l", "mgga_c_m06_hf"}:
        if all(hasattr(p.params, field) for field in ("css", "cab", "dss", "dab")):
            p = p._replace(maple_name="mgga_c_m06l")
    if hasattr(p, "func_aux"):
        children = [_repair_parameter_vectors(child, _default_parameter_values(child.name, polarized), polarized,
                                                _parameter_names(child.name))
                    for child in p.func_aux]
        p = p._replace(func_aux=children)
    return p


@lru_cache(maxsize=256)
def parameter_factory(factory):
    """Keep source-factory parameters and generated JAX formulas intact."""
    if not isinstance(factory, FunctionType) or "get_p" not in factory.__globals__:
        return factory
    reader = _parameter_reader(factory.__globals__["get_p"])
    parameter_names = tuple(name for name in inspect.signature(factory).parameters if name != "polarized")
    signature = inspect.signature(factory)

    def compatible(*args, **kwargs):
        supplied = signature.bind_partial(*args, **kwargs).arguments
        concrete_args = tuple(None if isinstance(value, jax.core.Tracer) else value for value in args)
        concrete_kwargs = {key: None if isinstance(value, jax.core.Tracer) else value
                           for key, value in kwargs.items()}

        def get_p(name, polarized, *values):
            actual = list(values)
            for index, key in enumerate(parameter_names):
                if index < len(actual) and supplied.get(key) is not None:
                    actual[index] = supplied[key]
            concrete = [values[index] if isinstance(value, jax.core.Tracer) else value
                        for index, value in enumerate(actual)]
            p = _repair_parameter_vectors(reader(name, polarized, *concrete), actual, polarized, parameter_names)
            # Only the current factory's caller overrides are put back into
            # scalar JAX fields. Child defaults are not overrides: resetting
            # them destroys custom composite parameters (e.g. PW6B95).
            mapped = {"polarized", "_omega"}
            scalar_updates = {}
            if hasattr(p, "params") and hasattr(p.params, "_fields"):
                for field in p.params._fields:
                    value = getattr(p.params, field)
                    ndim = getattr(value, "ndim", None)
                    ndim = np.ndim(value) if ndim is None else ndim
                    key = "_"+field
                    if ndim == 0 and key in supplied and supplied[key] is not None:
                        scalar_updates[field] = supplied[key]
                        mapped.add(key)
                    elif ndim == 1:
                        prefix = _VECTOR_PREFIXES.get(field, "_"+field)
                        mapped.update(key for key in supplied if key.startswith(prefix)
                                      and re.fullmatch(r"\d+", key[len(prefix):]))
                if scalar_updates:
                    p = p._replace(params=p.params._replace(**scalar_updates))
            unsupported = [key for key, value in supplied.items()
                           if isinstance(value, jax.core.Tracer) and key not in mapped]
            if unsupported:
                raise NotImplementedError(f"AD of factory parameters {unsupported!r} has no mapped JAX parameter field.")
            if "_omega" in supplied and supplied["_omega"] is not None:
                p = p._replace(cam_omega=supplied["_omega"])
            return p

        namespace = dict(factory.__globals__, get_p=get_p, invoke=_invoke_compatible)
        clone = FunctionType(factory.__code__, namespace, factory.__name__,
                             factory.__defaults__, factory.__closure__)
        clone.__kwdefaults__ = factory.__kwdefaults__
        return clone(*concrete_args, **concrete_kwargs)

    return update_wrapper(compatible, factory)
