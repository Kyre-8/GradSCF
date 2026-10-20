"""Single pure-JAX local energy graph used by molecular and periodic consumers.

All values are energy per volume. Point variables are [rho_a,rho_b] (LDA),
then grad_a[3],grad_b[3] (GGA), then tau_a,tau_b (MGGA), in atomic units.
This module supplies only the semilocal contribution of a complete recipe.
"""

from functools import lru_cache
from importlib import metadata
import ctypes
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array

from . import _jax_xc
from ._jax_xc import MissingJAXXCError, ensure_finite, jax_xc_functional_info
from .features import restricted_feature_bundle_from_rho_grad_tau
from .parser import (_TERM_FAMILIES, parse_recipe, resolve_xc_component_name,
                     xc_type)
from .registry import get_standard_recipe
from .types import LocalXCTermSpec, RestrictedFeatureBundle

_DENSITY_FLOOR = 1e-12


def jax_xc_backend_info() -> dict[str, Any]:
    """Return diagnostics for the active installed jax_xc backend."""

    from ._jax_xc import load_jax_xc

    try:
        module, backend = load_jax_xc()
    except MissingJAXXCError as exc:
        return {
            "available": False,
            "backend": "missing",
            "module_version": None,
            "reason": str(exc),
        }
    try:
        distribution_version = metadata.version("jax-xc")
    except metadata.PackageNotFoundError:
        distribution_version = None
    parameter_version = parameter_path = None
    if getattr(getattr(module, "_module", module), "__file__", None) is not None:
        try:
            from jax_xc.libxc.core import core
            version_function = ctypes.CFUNCTYPE(ctypes.c_char_p)(("xc_version_string", core))
            parameter_version = version_function().decode()
            parameter_path = core._name
        except (ImportError, AttributeError, OSError):
            pass
    return {
        "available": True,
        "backend": backend,
        "module_version": getattr(module, "__version__", None) or distribution_version,
        "jax_xc_distribution_version": distribution_version,
        "parameter_library_version": parameter_version,
        "parameter_library_path": parameter_path,
        "formula_backend": "jax_xc generated JAX kernels",
    }


def _omega_for_term(
    term: LocalXCTermSpec,
    omega: Array | float | None,
) -> Array | float | None:
    if term.omega_mode == "runtime":
        _jax_xc._validate_runtime_omega(omega)
        return 0.4 if omega is None else omega
    if term.omega_mode == "fixed":
        if term.fixed_omega is None:
            raise ValueError(f"LocalXCTermSpec {term.name!r} is fixed-omega but fixed_omega is None.")
        return term.fixed_omega
    return None


def _local_recipe(spec, omega, allow_experimental):
    recipe = parse_recipe(spec, allow_experimental_jax_xc=allow_experimental)
    if recipe.needs_laplacian:
        raise NotImplementedError(f"XC recipe {recipe.name!r} requires a density Laplacian; the local field contract supplies rho, grad and tau only.")
    fixed_native = any(term.name.startswith("hyb_") for term in recipe.semilocal)
    if omega is not None and any(term.omega_mode == "runtime" for term in recipe.semilocal):
        _jax_xc._validate_runtime_omega(omega)
    if omega is not None and recipe.default_omega is not None and (fixed_native or recipe.name == "hse03"):
        if isinstance(omega, jax.core.Tracer) or abs(float(omega)-recipe.default_omega) > 1e-14:
            raise ValueError(f"XC recipe {recipe.name!r} has fixed omega={recipe.default_omega} bohr^-1.")
    for term in recipe.semilocal:
        if term.name in _jax_xc._KNOWN_MISMATCH_JAX_XC_FUNCTIONALS and not allow_experimental:
            raise NotImplementedError(f"XC component {term.name!r} has a known JAX/Libxc mismatch; the standard recipe is not available.")
    catalog = get_standard_recipe(spec) is not None if isinstance(spec, str) else True
    return recipe, recipe.default_omega if omega is None else omega, bool(allow_experimental or catalog)


def _component_allowed(name, allowed):
    return bool(allowed and _jax_xc.jax_xc_functional_info(name).status == "experimental")


def xc_type_from_term_specs(
    term_specs: Sequence[LocalXCTermSpec],
    *,
    allow_experimental_jax_xc: bool = False,
) -> str:
    if not term_specs:
        return "HF"
    families = set()
    for term in term_specs:
        term_name = resolve_xc_component_name(term.name)
        if term_name in _TERM_FAMILIES:
            families.add(_TERM_FAMILIES[term_name])
            continue
        info = jax_xc_functional_info(term_name)
        if info.status == "experimental" and not allow_experimental_jax_xc:
            raise ValueError(
                f"jax_xc functional {term_name!r} is experimental: {info.reason} "
                "Pass allow_experimental_jax_xc=True to evaluate it."
            )
        families.add(info.family)
    if not families:
        return "HF"
    if "MGGA" in families:
        return "MGGA"
    if "GGA" in families or "HYB_GGA" in families:
        return "GGA"
    return "LDA"


def eval_xc_term_specs_energy_density(
    term_specs: Sequence[LocalXCTermSpec],
    features: RestrictedFeatureBundle,
    *,
    omega: Array | float | None = None,
    allow_experimental_jax_xc: bool = False,
) -> Array:
    from . import _jax_xc

    values = []
    for term in term_specs:
        term_name = resolve_xc_component_name(term.name)
        eps = _jax_xc.eval_jax_xc_from_restricted_features(
            term_name,
            features,
            omega=_omega_for_term(term, omega),
            allow_experimental_jax_xc=allow_experimental_jax_xc,
        )
        values.append(term.coefficient * features.rho * eps)
    if not values:
        return jnp.zeros_like(features.rho)
    return ensure_finite(jnp.sum(jnp.stack(values, axis=0), axis=0), context="local XC energy")


def eval_xc_energy_density(
    spec: str,
    features: RestrictedFeatureBundle,
    *,
    omega: Array | float | None = None,
    allow_experimental_jax_xc: bool = False,
) -> Array:
    """Return the local XC grid contribution e_xc(r) for direct quadrature."""

    from . import _jax_xc

    recipe, omega, allowed = _local_recipe(spec, omega, allow_experimental_jax_xc)
    values = []
    for term in recipe.semilocal:
        term_omega = _omega_for_term(term, omega)
        eps = _jax_xc.eval_jax_xc_from_restricted_features(
            term.name,
            features,
            omega=term_omega,
            allow_experimental_jax_xc=_component_allowed(term.name, allowed),
        )
        values.append(term.coefficient * features.rho * eps)
    if not values:
        return jnp.zeros_like(features.rho)
    return ensure_finite(jnp.sum(jnp.stack(values, axis=0), axis=0), context="local XC energy")


def eval_xc_energy_density_unrestricted(
    spec: str,
    features: RestrictedFeatureBundle,
    *,
    omega: Array | float | None = None,
    allow_experimental_jax_xc: bool = False,
) -> Array:
    """Return spin-polarized local XC grid contribution e_xc(r)."""

    from . import _jax_xc

    recipe, omega, allowed = _local_recipe(spec, omega, allow_experimental_jax_xc)
    values = []
    for term in recipe.semilocal:
        term_omega = _omega_for_term(term, omega)
        eps = _jax_xc.eval_jax_xc_from_unrestricted_features(
            term.name,
            features,
            omega=term_omega,
            allow_experimental_jax_xc=_component_allowed(term.name, allowed),
        )
        values.append(term.coefficient * features.rho * eps)
    if not values:
        return jnp.zeros_like(features.rho)
    return ensure_finite(jnp.sum(jnp.stack(values, axis=0), axis=0), context="local XC energy")


def eval_xc_energy_density_unrestricted_from_density_gradients(
    spec: str,
    rho_a: Array,
    rho_b: Array,
    grad_a: Array,
    grad_b: Array,
    *,
    tau_a: Array | None = None,
    tau_b: Array | None = None,
    omega: Array | float | None = None,
    allow_experimental_jax_xc: bool = False,
) -> Array:
    """Return polarized local XC energy from explicit spin-density gradients."""

    from . import _jax_xc

    recipe, omega, allowed = _local_recipe(spec, omega, allow_experimental_jax_xc)
    values = []
    for term in recipe.semilocal:
        term_omega = _omega_for_term(term, omega)
        value = _jax_xc.eval_jax_xc_energy_density_from_unrestricted_density_gradients(
            term.name,
            rho_a,
            rho_b,
            grad_a,
            grad_b,
            tau_a=tau_a,
            tau_b=tau_b,
            omega=term_omega,
            allow_experimental_jax_xc=_component_allowed(term.name, allowed),
        )
        values.append(term.coefficient * value)
    if not values:
        return jnp.asarray(0.0, dtype=jnp.asarray(rho_a).dtype)
    return ensure_finite(jnp.sum(jnp.stack(values, axis=0), axis=0), context="local XC energy")


@lru_cache(maxsize=128)
def point_energy(spec, kind):
    """Return a cached scalar local energy, differentiable in spin fields.

    Negative spin densities/tau use the existing zero-side SCF convention;
    the backend adds 1e-15 to rho and one Cartesian gradient component to keep
    zero-spin and zero-gradient derivative branches in their valid domain.
    Tunable runtime omega must stay positive, including under JIT. Static
    fixed-zero local terms have a separate energy/field-derivative limit;
    this does not advertise a dynamic-zero omega derivative.
    """
    recipe = parse_recipe(spec)
    kind = str(kind).upper()
    if kind not in {"HF", "LDA", "GGA", "MGGA"}:
        raise ValueError(f"Unsupported local XC kind {kind!r}.")
    order = {"HF": 0, "LDA": 1, "GGA": 2, "MGGA": 3}
    if order[kind] < order[recipe.kind]:
        raise ValueError(f"XC recipe {recipe.name!r} requires {recipe.kind} fields, received {kind}.")

    def energy(variables, omega=None):
        # Validate fixed omega before invoking a backend even for unsupported
        # extra physical contributions: this is explicitly a local evaluator.
        _local_recipe(recipe, omega, False)
        variables = jnp.asarray(variables)
        if kind == "HF":
            return jnp.asarray(0., dtype=variables.dtype)
        rho_a = jnp.where(variables[0] >= 0., variables[0], 0.)
        rho_b = jnp.where(variables[1] >= 0., variables[1], 0.)
        zeros = jnp.zeros((3,), dtype=variables.dtype)
        grad_a, grad_b = (zeros, zeros) if kind == "LDA" else (variables[2:5], variables[5:8])
        tau_a = jnp.maximum(variables[8], 0.) if kind == "MGGA" else None
        tau_b = jnp.maximum(variables[9], 0.) if kind == "MGGA" else None
        return eval_xc_energy_density_unrestricted_from_density_gradients(
            recipe, rho_a, rho_b, grad_a, grad_b, tau_a=tau_a, tau_b=tau_b, omega=omega,
        )

    return energy


def _checked_grid_kernel(spec, kind, transform):
    scalar = point_energy(spec, kind)
    compiled = jax.jit(jax.vmap(transform(scalar), in_axes=(0, None)))

    def evaluate(variables, omega=None):
        recipe, _, _ = _local_recipe(spec, omega, False)
        fixed = recipe.name == "hse03" or any(term.name.startswith("hyb_") for term in recipe.semilocal)
        result = compiled(variables, None if fixed and omega is not None else omega)
        return jax.tree_util.tree_map(lambda value: ensure_finite(value, context="local XC grid derivative"), result)

    return evaluate


@lru_cache(maxsize=128)
def grid_energy(spec, kind):
    return _checked_grid_kernel(spec, kind, lambda energy: energy)


@lru_cache(maxsize=128)
def grid_value_and_grad(spec, kind):
    return _checked_grid_kernel(spec, kind, jax.value_and_grad)


@lru_cache(maxsize=128)
def grid_hessian(spec, kind):
    return _checked_grid_kernel(spec, kind, jax.hessian)


def eval_xc_response_tensor(
    spec: str,
    rho: Array,
    *,
    grad: Array | None = None,
    tau: Array | None = None,
    density_floor: float = _DENSITY_FLOOR,
    omega: Array | float | None = None,
    allow_experimental_jax_xc: bool = False,
) -> tuple[str, Array]:
    """Return the semilocal grid Hessian for a JAX libxc-like spec."""

    kind = xc_type(
        spec,
        allow_experimental_jax_xc=allow_experimental_jax_xc,
    )
    if kind == "HF":
        raise ValueError("Pure HF does not define a semilocal response tensor.")

    rho = jnp.maximum(jnp.asarray(rho), density_floor)
    dtype = rho.dtype
    zeros_grad = jnp.zeros(rho.shape + (3,), dtype=dtype)
    grad = zeros_grad if grad is None else jnp.asarray(grad, dtype=dtype)
    tau = jnp.zeros_like(rho) if tau is None else jnp.asarray(tau, dtype=dtype)

    if kind == "LDA":
        response_variables = rho[..., None]
    elif kind == "GGA":
        response_variables = jnp.concatenate([rho[..., None], grad], axis=-1)
    elif kind == "MGGA":
        response_variables = jnp.concatenate(
            [rho[..., None], grad, tau[..., None]],
            axis=-1,
        )
    else:
        raise ValueError(f"Unsupported XC response kind {kind!r}.")

    recipe, _, _ = _local_recipe(spec, omega, allow_experimental_jax_xc)
    # Equal-spin projection is a linear feature map, not a second local XC
    # graph. Its Hessian is T.T @ H_spin @ T at every quadrature point.
    indices = {"LDA": (0, 0), "GGA": (0, 0, 1, 2, 3, 1, 2, 3),
               "MGGA": (0, 0, 1, 2, 3, 1, 2, 3, 4, 4)}[kind]
    projection = .5*jnp.eye(response_variables.shape[-1], dtype=dtype)[jnp.asarray(indices)]
    spin_variables = response_variables @ projection.T
    spin_tensor = grid_hessian(recipe, kind)(spin_variables, omega)
    tensor = jnp.einsum("ia,pij,jb->abp", projection, spin_tensor, projection)
    tensor = ensure_finite(tensor, context="local XC Hessian")
    return kind, tensor
