from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache, update_wrapper
from types import FunctionType
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .registry import component_requires_laplacian, local_composite_definitions


class MissingJAXXCError(ImportError):
    """Raised when the configured environment does not provide ``jax_xc``."""


JAXXCStatus = Literal["strict", "wrapped", "experimental", "unavailable"]


@dataclass(frozen=True)
class JAXXCFunctionalInfo:
    name: str
    status: JAXXCStatus
    family: str
    reason: str
    children: tuple[str, ...] = ()


_DENSITY_FLOOR = 1e-12
_UNRESTRICTED_DERIVATIVE_FLOOR = 1e-15

_STRICT_JAX_XC_FUNCTIONALS = {
    "lda_x",
    "lda_c_pw",
    "lda_c_vwn",
    "lda_c_vwn_rpa",
    "gga_x_b88",
    "gga_x_pbe",
    "gga_x_wpbeh",
    "gga_c_lyp",
    "gga_c_pbe",
    "mgga_x_r2scan",
    "mgga_c_r2scan",
}

_HybridTerm = tuple[float, str, dict[str, float]]

SAFE_JAX_XC_WRAPPED_COMPOSITES: dict[str, tuple[_HybridTerm, ...]] = local_composite_definitions()

_EXPERIMENTAL_JAX_XC_FUNCTIONALS = {
    "gga_x_rpbe",
    "gga_x_wc",
    "gga_x_pw91",
    "hyb_gga_xc_b97",
    "hyb_gga_xc_b97_1",
    "hyb_gga_xc_wb97x",
}

# Legacy array-conversion mismatches are repaired by parameter_factory. Keep
# this explicit deny-list for independently demonstrated unresolved channels.
_KNOWN_MISMATCH_JAX_XC_FUNCTIONALS: set[str] = set()

_JAX_XC_FUNCTIONAL_PREFIXES = (
    "lda_",
    "gga_",
    "mgga_",
    "hyb_gga_",
    "hyb_mgga_",
)


def ensure_finite(value, *, context="XC evaluation"):
    """Check eager values; traced graphs retain failures for caller validation."""
    if not isinstance(value, jax.core.Tracer) and not np.all(np.isfinite(np.asarray(value))):
        raise FloatingPointError(f"{context} returned non-finite values.")
    return value


def _validate_runtime_omega(omega):
    """Require positive tunable omega when its primal value is concrete.

    Fully dynamic JIT callers must keep omega positive; validation introduces
    no host callback. Static local terms can separately fix omega exactly zero.
    """
    value = omega
    while isinstance(value, jax.core.Tracer) and hasattr(value, "primal"):
        value = value.primal
    if value is not None and not isinstance(value, jax.core.Tracer):
        if not np.isfinite(float(value)) or float(value) <= 0.:
            raise ValueError("Runtime omega must be positive; a static fixed-zero local term is a separate limit.")


def _is_mgga_name(name: str) -> bool:
    return name.startswith("mgga_") or name.startswith("hyb_mgga_")


def _is_kinetic_name(name: str) -> bool:
    return (
        name.startswith("lda_k_")
        or name.startswith("gga_k_")
        or name.startswith("mgga_k_")
    )


def _is_jax_xc_functional_name(name: str) -> bool:
    return name.startswith(_JAX_XC_FUNCTIONAL_PREFIXES)


def _family_from_name(name: str) -> str:
    if name.startswith("lda_"):
        return "LDA"
    if _is_mgga_name(name):
        return "MGGA"
    if name.startswith("gga_"):
        return "GGA"
    if name.startswith("hyb_gga_"):
        return "HYB_GGA"
    return "unknown"


def load_jax_xc() -> tuple[Any, str]:
    """Load the installed ``jax_xc`` package.

    GradSCF no longer carries a generated or local fallback for traditional
    XC formulas. The active runtime environment must provide ``jax_xc``.
    """

    try:
        module = importlib.import_module("jax_xc")
    except (ImportError, OSError) as exc:
        causes = []
        current = exc
        while current is not None and len(causes) < 5:
            causes.append(f"{type(current).__name__}: {current}")
            current = current.__cause__ or current.__context__
        raise MissingJAXXCError(
            "Traditional XC evaluation requires the optional package 'jax_xc' "
            "to load in the active Python environment. " + " Caused by: ".join(causes)
        ) from exc
    return _JAXXCModule(module), "upstream"


def _active_jax_xc_has(name: str) -> bool:
    try:
        module, _ = load_jax_xc()
    except MissingJAXXCError:
        return False
    return hasattr(module, name)


def _active_jax_xc_names() -> set[str]:
    try:
        module, _ = load_jax_xc()
    except MissingJAXXCError:
        return set()
    raw_module = getattr(module, "_module", module)
    names = {name for name in dir(raw_module) if not name.startswith("_")}
    names.update(SAFE_JAX_XC_WRAPPED_COMPOSITES)
    return names


def jax_xc_functional_info(name: str) -> JAXXCFunctionalInfo:
    canonical = str(name).strip().lower()
    if canonical == "hyb_gga_xc_wb97x_d":
        return JAXXCFunctionalInfo(
            name=canonical,
            status="wrapped",
            family="GGA",
            reason=(
                "Local omegaB97X-D channel at fixed omega=0.2 bohr^-1; "
                "B97 coefficient vectors are preserved by a parameter wrapper. "
                "Energy, potential and kernel are checked against LibXC. "
                "Exact exchange and CHG dispersion must be supplied separately."
            ),
        )
    if canonical in _STRICT_JAX_XC_FUNCTIONALS:
        return JAXXCFunctionalInfo(
            name=canonical,
            status="strict",
            family=_family_from_name(canonical),
            reason="Provided by the installed jax_xc package and allowed by default.",
        )
    if canonical in SAFE_JAX_XC_WRAPPED_COMPOSITES:
        children = tuple(child for _, child, _ in SAFE_JAX_XC_WRAPPED_COMPOSITES[canonical])
        return JAXXCFunctionalInfo(
            name=canonical,
            status="wrapped",
            family=_family_from_name(canonical),
            reason="Resolved into safe semilocal child components from the installed jax_xc package.",
            children=children,
        )
    is_active = _active_jax_xc_has(canonical)
    if is_active and _is_kinetic_name(canonical):
        return JAXXCFunctionalInfo(
            name=canonical,
            status="unavailable",
            family=_family_from_name(canonical),
            reason=(
                "This is a kinetic-energy functional exposed by jax_xc, not an XC "
                "energy-density component."
            ),
        )
    if is_active and (
        canonical in _EXPERIMENTAL_JAX_XC_FUNCTIONALS
        or _is_jax_xc_functional_name(canonical)
    ):
        reason = "Available from installed jax_xc but not validated for Neural XC training."
        if canonical in _KNOWN_MISMATCH_JAX_XC_FUNCTIONALS:
            reason = (
                "This local channel has an unresolved independent PySCF/Libxc reference mismatch; "
                "explicit experimental opt-in is required."
            )
        elif _is_mgga_name(canonical):
            reason = (
                "MGGA functional is available from jax_xc and requires an orbital-derived "
                "tau/mo_fn bridge; explicit experimental opt-in is required."
            )
            if canonical.startswith("hyb_mgga_"):
                reason = (
                    "Hybrid MGGA functional is available from jax_xc as an experimental "
                    "local channel; GradSCF does not infer exact-exchange fractions "
                    "from this name."
                )
        elif canonical.startswith("hyb_gga_"):
            reason = (
                "Hybrid GGA functional is available from jax_xc as an experimental "
                "local channel; GradSCF does not infer exact-exchange fractions "
                "from this name."
            )
        return JAXXCFunctionalInfo(
            name=canonical,
            status="experimental",
            family=_family_from_name(canonical),
            reason=reason,
        )
    return JAXXCFunctionalInfo(
        name=canonical,
        status="unavailable",
        family=_family_from_name(canonical),
        reason="No allowed jax_xc implementation is available.",
    )


def list_jax_xc_functionals(status: JAXXCStatus | None = None) -> tuple[JAXXCFunctionalInfo, ...]:
    active_functionals = {
        name
        for name in _active_jax_xc_names()
        if _is_jax_xc_functional_name(name)
    }
    names = sorted(
        _STRICT_JAX_XC_FUNCTIONALS
        | set(SAFE_JAX_XC_WRAPPED_COMPOSITES)
        | _EXPERIMENTAL_JAX_XC_FUNCTIONALS
        | {"hyb_gga_xc_wb97x_d"}
        | active_functionals
    )
    infos = tuple(jax_xc_functional_info(name) for name in names)
    if status is None:
        return infos
    return tuple(info for info in infos if info.status == status)


def _raise_if_not_allowed(
    info: JAXXCFunctionalInfo,
    *,
    allow_experimental_jax_xc: bool,
) -> None:
    if component_requires_laplacian(info.name):
        raise NotImplementedError(f"XC component {info.name!r} requires a density Laplacian; rho/grad/tau alone are insufficient.")
    if info.status == "unavailable":
        raise KeyError(f"jax_xc functional {info.name!r} is unavailable: {info.reason}")
    if info.status == "experimental" and not bool(allow_experimental_jax_xc):
        raise ValueError(
            f"jax_xc functional {info.name!r} is experimental: {info.reason} "
            "Pass allow_experimental_jax_xc=True to evaluate it."
        )


@lru_cache(maxsize=4)
def _wb97xd_parameter_factory(factory):
    """Restore B97 coefficient vectors lost by the legacy jax_xc converter.

    In jax_xc 0.0.9, ``get_p`` repeats the first element of every C array.
    Use the factory's actual scalar arguments and a factored local kernel to
    avoid overflow in high derivatives of expanded B97 polynomials. As in the
    PW compatibility layer, the cloned function has private globals; the
    installed package is never modified.
    """
    if not isinstance(factory, FunctionType) or "get_p" not in factory.__globals__:
        return factory
    upstream_get_p = factory.__globals__["get_p"]

    def get_p(name, polarized, *values):
        p = upstream_get_p(name, polarized, *values)
        if name != "hyb_gga_xc_wb97x_d":
            return p
        fields = ("c_x", "c_ss", "c_ab")
        if len(values) != 18 or not hasattr(p, "params") or not all(
            hasattr(p.params, field) for field in fields
        ):
            raise RuntimeError("Unsupported upstream omegaB97X-D parameter layout")
        arrays = {field: np.asarray(values[5*i:5*i+5], dtype=np.float64)
                  for i, field in enumerate(fields)}
        return p._replace(params=p.params._replace(**arrays))

    from ._compat import invoke

    namespace = dict(factory.__globals__, get_p=get_p, invoke=invoke)
    wrapped = FunctionType(factory.__code__, namespace, factory.__name__,
                           factory.__defaults__, factory.__closure__)
    wrapped.__kwdefaults__ = factory.__kwdefaults__
    return update_wrapper(wrapped, factory)


class _JAXXCModule:
    """Proxy an installed jax_xc module and expose GradSCF composites."""

    def __init__(self, module: Any):
        self._module = module
        self.__version__ = getattr(module, "__version__", None)

    def __getattr__(self, name: str):
        from ._compat import parameter_factory

        if name in SAFE_JAX_XC_WRAPPED_COMPOSITES:
            return self._hybrid_factory(name)
        if name == "lda_c_pw":
            from ._compat import pw_parameter_factory
            return pw_parameter_factory(getattr(self._module, name))
        if name == "hyb_gga_xc_wb97x_d":
            return _wb97xd_parameter_factory(getattr(self._module, name))
        factory = getattr(self._module, name)
        return parameter_factory(factory)

    def _hybrid_factory(self, name: str):
        terms = SAFE_JAX_XC_WRAPPED_COMPOSITES[name]
        module = self

        def factory(*, polarized: bool = False):
            child_functionals = [
                (
                    coefficient,
                    getattr(module, child_name)(polarized=polarized, **child_params),
                )
                for coefficient, child_name, child_params in terms
            ]

            def functional(rho_fn, r, mo_fn=None):
                total = None
                for coefficient, child in child_functionals:
                    try:
                        value = child(rho_fn, r, mo_fn)
                    except TypeError:
                        value = child(rho_fn, r)
                    contribution = coefficient * _coerce_functional_value(value)
                    total = contribution if total is None else total + contribution
                if total is None:
                    return jnp.asarray(0.0)
                return total

            return functional

        return factory


_SafeJAXXCModule = _JAXXCModule


def _factory_kwargs(name: str, omega: Any | None) -> dict[str, Any]:
    if name == "hyb_gga_xc_wb97x_d" and omega is not None:
        if isinstance(omega, jax.core.Tracer) or float(omega) != 0.2:
            raise ValueError("omegaB97X-D local channel is validated only at fixed omega=0.2 bohr^-1")
    if name == "gga_x_wpbeh" and omega is not None:
        return {"_omega": omega}
    return {}


def _coerce_functional_value(value: Any) -> Any:
    if isinstance(value, tuple):
        value = value[0]
    return value


def _evaluate_factory_from_restricted_features(
    factory: Any,
    features: Any,
    *,
    family: str,
) -> jnp.ndarray:
    rho = jnp.maximum(jnp.asarray(features.rho), _DENSITY_FLOOR)
    raw_sigma = jnp.asarray(features.sigma)
    sigma = jnp.where(raw_sigma >= 0., raw_sigma, 0.)
    grad_mag = jnp.sqrt(sigma + _UNRESTRICTED_DERIVATIVE_FLOOR**2)
    tau = jnp.maximum(jnp.asarray(features.tau_a + features.tau_b), 0.0)
    origin = jnp.zeros((3,), dtype=rho.dtype)

    def point_eval(rho_value, grad_value, tau_value):
        def rho_fn(r):
            return rho_value + grad_value * r[0]

        def mo_fn(r):
            mo_grad = jnp.sqrt(jnp.maximum(2.0 * tau_value, 1e-30))
            return jnp.asarray([mo_grad * r[0]], dtype=rho.dtype)

        if family == "MGGA":
            value = factory(rho_fn, origin, mo_fn)
        else:
            value = factory(rho_fn, origin)
        return jnp.asarray(_coerce_functional_value(value), dtype=rho.dtype)

    if rho.ndim == 0:
        value = point_eval(rho, grad_mag, tau)
    else:
        value = jax.vmap(point_eval)(rho, grad_mag, tau)
    return ensure_finite(value, context="restricted local XC energy")


def _evaluate_factory_from_unrestricted_features(
    factory: Any,
    features: Any,
    *,
    family: str,
) -> jnp.ndarray:
    from .features import spin_gradients_from_features

    rho_a = jnp.maximum(jnp.asarray(features.rho_a), _DENSITY_FLOOR)
    rho_b = jnp.maximum(jnp.asarray(features.rho_b), _DENSITY_FLOOR)
    grad_a, grad_b = spin_gradients_from_features(features)
    tau_a = jnp.maximum(jnp.asarray(features.tau_a), 0.0)
    tau_b = jnp.maximum(jnp.asarray(features.tau_b), 0.0)
    origin = jnp.zeros((3,), dtype=rho_a.dtype)

    def point_eval(rho_a_value, rho_b_value, grad_a_value, grad_b_value, tau_a_value, tau_b_value):
        def rho_fn(r):
            return jnp.asarray(
                [
                    rho_a_value + jnp.dot(grad_a_value, r),
                    rho_b_value + jnp.dot(grad_b_value, r),
                ],
                dtype=rho_a.dtype,
            )

        def mo_fn(r):
            mo_grad_a = jnp.sqrt(jnp.maximum(2.0 * tau_a_value, 1e-30))
            mo_grad_b = jnp.sqrt(jnp.maximum(2.0 * tau_b_value, 1e-30))
            return jnp.asarray(
                [
                    [mo_grad_a * r[0]],
                    [mo_grad_b * r[0]],
                ],
                dtype=rho_a.dtype,
            )

        if family == "MGGA":
            value = factory(rho_fn, origin, mo_fn)
        else:
            value = factory(rho_fn, origin)
        return jnp.asarray(_coerce_functional_value(value), dtype=rho_a.dtype)

    if rho_a.ndim == 0:
        value = point_eval(rho_a, rho_b, grad_a, grad_b, tau_a, tau_b)
    else:
        value = jax.vmap(point_eval)(rho_a, rho_b, grad_a, grad_b, tau_a, tau_b)
    return ensure_finite(value, context="unrestricted local XC energy")


def eval_jax_xc_energy_density_from_unrestricted_density_gradients(
    name: str,
    rho_a: Any,
    rho_b: Any,
    grad_a: Any,
    grad_b: Any,
    *,
    tau_a: Any | None = None,
    tau_b: Any | None = None,
    omega: Any | None = None,
    allow_experimental_jax_xc: bool = False,
) -> jnp.ndarray:
    """Evaluate one polarized local XC contribution from spin densities and gradients."""

    info = jax_xc_functional_info(name)
    _raise_if_not_allowed(info, allow_experimental_jax_xc=allow_experimental_jax_xc)
    if info.family == "MGGA" and (tau_a is None or tau_b is None):
        raise ValueError("MGGA requires explicit orbital-derived tau_a and tau_b.")
    module, _ = load_jax_xc()
    try:
        factory = getattr(module, info.name)
    except AttributeError as exc:
        raise KeyError(f"Installed jax_xc does not expose functional {info.name!r}.") from exc
    functional = factory(
        polarized=True,
        **_factory_kwargs(info.name, omega),
    )

    # Evaluate the one-sided derivative where a spin density or its gradient vanishes.
    rho_a_eff = jnp.asarray(rho_a) + _UNRESTRICTED_DERIVATIVE_FLOOR
    rho_b_eff = jnp.asarray(rho_b) + _UNRESTRICTED_DERIVATIVE_FLOOR
    grad_a = jnp.asarray(grad_a)
    grad_b = jnp.asarray(grad_b)
    gradient_floor = jnp.zeros_like(grad_a).at[0].set(_UNRESTRICTED_DERIVATIVE_FLOOR)
    grad_a_eff = grad_a + gradient_floor
    grad_b_eff = grad_b + gradient_floor
    origin = jnp.zeros((3,), dtype=rho_a_eff.dtype)

    def rho_fn(displacement):
        return jnp.asarray(
            [
                rho_a_eff + jnp.dot(grad_a_eff, displacement),
                rho_b_eff + jnp.dot(grad_b_eff, displacement),
            ],
            dtype=rho_a_eff.dtype,
        )

    if info.family == "MGGA":
        # Synthetic orbital gradients reproduce the supplied positive kinetic
        # densities. rho_fn retains the full, nonparallel spin gradients.
        def mo_fn(displacement):
            slopes = jnp.sqrt(jnp.maximum(2*jnp.asarray([tau_a, tau_b]), 1e-30))
            return slopes[:, None]*displacement[0]
        epsilon_xc = functional(rho_fn, origin, mo_fn)
    else:
        epsilon_xc = functional(rho_fn, origin)
    return (rho_a_eff + rho_b_eff) * jnp.asarray(
        _coerce_functional_value(epsilon_xc),
        dtype=rho_a_eff.dtype,
    )


def eval_jax_xc_from_restricted_features(
    name: str,
    features: Any,
    *,
    omega: Any | None = None,
    allow_experimental_jax_xc: bool = False,
) -> jnp.ndarray:
    """Evaluate a jax_xc functional as epsilon_xc on restricted grid features."""

    info = jax_xc_functional_info(name)
    _raise_if_not_allowed(info, allow_experimental_jax_xc=allow_experimental_jax_xc)
    module, _ = load_jax_xc()
    try:
        factory = getattr(module, info.name)
    except AttributeError as exc:
        raise KeyError(f"Installed jax_xc does not expose functional {info.name!r}.") from exc
    functional = factory(
        polarized=False,
        **_factory_kwargs(info.name, omega),
    )
    return _evaluate_factory_from_restricted_features(
        functional,
        features,
        family=info.family,
    )


def eval_jax_xc_from_unrestricted_features(
    name: str,
    features: Any,
    *,
    omega: Any | None = None,
    allow_experimental_jax_xc: bool = False,
) -> jnp.ndarray:
    """Evaluate a jax_xc functional as spin-polarized epsilon_xc."""

    info = jax_xc_functional_info(name)
    _raise_if_not_allowed(info, allow_experimental_jax_xc=allow_experimental_jax_xc)
    module, _ = load_jax_xc()
    try:
        factory = getattr(module, info.name)
    except AttributeError as exc:
        raise KeyError(f"Installed jax_xc does not expose functional {info.name!r}.") from exc
    functional = factory(
        polarized=True,
        **_factory_kwargs(info.name, omega),
    )
    return _evaluate_factory_from_unrestricted_features(
        functional,
        features,
        family=info.family,
    )
