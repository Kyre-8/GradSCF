"""Traditional XC definitions and one local JAX evaluator.

Named recipes include every required physical term. ``eval_xc_*`` and grid
kernels explicitly evaluate only their semilocal contribution; SCF consumers
must assemble HF/RSH and call ``require_complete_support`` for extra terms.
"""

from dataclasses import dataclass

from .types import (AdditionalTerm, ExactExchange, FunctionalRecipe,
                    LocalXCTermSpec, RestrictedFeatureBundle, RSHFunctionalPreset,
                    XCTerm)
from .registry import (canonical_rsh_preset_name, get_rsh_functional_preset,
                       list_functional_recipes, list_rsh_functional_presets)
from .parser import (FRIENDLY_XC_COMPONENT_ALIASES, b3lyp_component_basis,
                     b3lyp_component_coefficients, hybrid_coeff, parse_recipe,
                     parse_xc, resolve_semilocal_xc_specs,
                     resolve_xc_component_name, semilocal_terms, xc_type)
from .features import restricted_feature_bundle_from_rho_grad_tau
from ._jax_xc import (JAXXCFunctionalInfo, JAXXCStatus, MissingJAXXCError,
                      jax_xc_functional_info, list_jax_xc_functionals, load_jax_xc)
from .evaluate import (eval_xc_energy_density,
                       eval_xc_energy_density_unrestricted,
                       eval_xc_energy_density_unrestricted_from_density_gradients,
                       eval_xc_response_tensor, eval_xc_term_specs_energy_density,
                       grid_energy, grid_hessian, grid_value_and_grad,
                       jax_xc_backend_info, point_energy, xc_type_from_term_specs)

CLASSIC_XC_SPECS = {name: name for name in ("lda", "svwn", "pbe", "pbe0", "b3lyp")}


@dataclass(frozen=True)
class TraditionalXCFunctional:
    """Describe a complete recipe and expose its local contribution explicitly."""

    spec: str
    name: str | None = None

    def __post_init__(self):
        parse_recipe(self.spec)
        object.__setattr__(self, "name", self.spec if self.name is None else str(self.name))

    def __str__(self) -> str:
        return self.spec

    @property
    def recipe(self):
        return parse_recipe(self.spec)

    @property
    def exact_exchange_fraction(self):
        return self.recipe.exact_exchange.full_range

    @property
    def response_kind(self):
        return self.recipe.kind

    def terms(self):
        return tuple(parse_xc(self.spec))

    def semilocal_component_terms(self):
        return tuple(semilocal_terms(self.spec))

    def energy_density(self, features):
        """Legacy spelling for the semilocal energy per volume only."""
        return eval_xc_energy_density(self.spec, features)

    def local_energy_density(self, features, *, density_floor=1e-12):
        del density_floor
        return self.energy_density(features)

    def response_tensor(self, rho, *, grad=None, tau=None, density_floor=1e-12):
        return eval_xc_response_tensor(self.spec, rho, grad=grad, tau=tau,
                                       density_floor=density_floor)


def make_classic_xc_functional(spec, *, name=None):
    return TraditionalXCFunctional(spec, name=name)


def make_lda_functional():
    return make_classic_xc_functional("lda")


def make_pbe_functional():
    return make_classic_xc_functional("pbe")


def make_pbe0_functional():
    return make_classic_xc_functional("pbe0")


def make_b3lyp_functional():
    return make_classic_xc_functional("b3lyp")


def info(spec: str | FunctionalRecipe | TraditionalXCFunctional) -> dict:
    """Return complete metadata and actual local-backend availability.

    Factory presence is distinct from independent numerical validation and
    from support by a particular molecular/periodic method or derivative.
    Neural Functional objects retain their own metadata and are not standard
    traditional recipes.
    """
    from . import _jax_xc

    if not isinstance(spec, (str, FunctionalRecipe, TraditionalXCFunctional)):
        raise TypeError("info expects a traditional XC specification; neural Functional objects retain their own metadata.")
    recipe = parse_recipe(spec.spec if isinstance(spec, TraditionalXCFunctional) else spec)
    channels = []
    for term in recipe.semilocal:
        metadata = _jax_xc.jax_xc_functional_info(term.name)
        present = _jax_xc._active_jax_xc_has(term.name)
        channels.append({"name": term.name, "coefficient": term.coefficient,
                         "omega_mode": term.omega_mode, "fixed_omega": term.fixed_omega,
                         "factory_available": present, "status": metadata.status,
                         "reason": metadata.reason})
    mismatch = any(term.name in _jax_xc._KNOWN_MISMATCH_JAX_XC_FUNCTIONALS
                   for term in recipe.semilocal)
    backend_available = (not recipe.needs_laplacian and not mismatch
                         and all(channel["factory_available"] and channel["status"] != "unavailable"
                                 for channel in channels))
    return {"recipe": recipe, "local_channels": tuple(channels),
            "backend_available": backend_available,
            "extra_requirements": tuple(name for name in ("dispersion", "nlc", "pt2")
                                        if getattr(recipe, name) is not None),
            "unsupported_features": ("laplacian",) if recipe.needs_laplacian else (),
            "validation_status": "known_mismatch" if mismatch else recipe.validation_status}


def available(spec: str | FunctionalRecipe | TraditionalXCFunctional | None = None) -> bool | tuple[str, ...]:
    """Check local backend support, or list such standard names if spec is None.

    True does not mean that extra physical terms, every consuming method, or
    every derivative order have been implemented or numerically validated.
    """
    if spec is not None:
        return bool(info(spec)["backend_available"])
    return tuple(recipe.name for recipe in list_functional_recipes()
                 if info(recipe)["backend_available"])


__all__ = [
    "AdditionalTerm", "CLASSIC_XC_SPECS", "ExactExchange", "FunctionalRecipe", "available", "info",
    "FRIENDLY_XC_COMPONENT_ALIASES", "JAXXCFunctionalInfo", "JAXXCStatus",
    "LocalXCTermSpec", "MissingJAXXCError", "RestrictedFeatureBundle",
    "RSHFunctionalPreset", "TraditionalXCFunctional", "XCTerm",
    "b3lyp_component_basis", "b3lyp_component_coefficients",
    "canonical_rsh_preset_name", "eval_xc_energy_density",
    "eval_xc_energy_density_unrestricted",
    "eval_xc_energy_density_unrestricted_from_density_gradients",
    "eval_xc_response_tensor", "eval_xc_term_specs_energy_density",
    "get_rsh_functional_preset", "grid_energy", "grid_hessian",
    "grid_value_and_grad", "hybrid_coeff", "jax_xc_backend_info",
    "jax_xc_functional_info", "list_functional_recipes", "list_jax_xc_functionals",
    "list_rsh_functional_presets", "load_jax_xc", "make_b3lyp_functional",
    "make_classic_xc_functional", "make_lda_functional", "make_pbe0_functional",
    "make_pbe_functional", "parse_recipe", "parse_xc", "point_energy",
    "resolve_semilocal_xc_specs", "resolve_xc_component_name",
    "restricted_feature_bundle_from_rho_grad_tau", "semilocal_terms",
    "xc_type", "xc_type_from_term_specs",
]
