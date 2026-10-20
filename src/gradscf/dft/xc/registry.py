"""One static source for complete recipes and their compatibility views."""

from dataclasses import replace
from importlib.resources import files
import json

from .types import (AdditionalTerm, ExactExchange, FunctionalRecipe,
                    LocalXCTermSpec, RSHFunctionalPreset)

# This is reference metadata, not runtime availability or derivative validation.
_LIBXC_REFERENCE_DATA = json.loads(
    files(__package__).joinpath("_catalog.json").read_text(encoding="utf-8")
)["names"]


def canonical_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace("ω", "w").replace(" ", "")


def _from_reference(name, record):
    omega, alpha, beta = record["rsh"]
    has_rsh = omega != 0.0 and (alpha != 0.0 or beta != 0.0)
    exchange = (ExactExchange(short_range=alpha + beta, long_range=alpha, omega=omega)
                if has_rsh else ExactExchange(full_range=record["exact_exchange"]))
    terms = tuple(LocalXCTermSpec(term["name"], term["coefficient"])
                  for term in record["terms"])
    nlc = (AdditionalTerm("rVV10" if "rvv10" in name else "VV10",
                          coefficient=record["nlc"][0][1],
                          references=tuple(record["references"]),
                          parameters=tuple(zip(("b", "C"), record["nlc"][0][0])))
           if record["nlc"] else None)
    return FunctionalRecipe(name, terms, exchange, record["family"],
                            nlc=nlc, references=tuple(record["references"]),
                            needs_laplacian=record["needs_laplacian"])


RECIPES = {canonical_name(name): _from_reference(canonical_name(name), record)
           for name, record in _LIBXC_REFERENCE_DATA.items()}
RECIPES["lda"] = replace(RECIPES["lda,pw"], name="lda")
RECIPES["svwn_rpa"] = replace(
    RECIPES["svwn"], name="svwn_rpa",
    semilocal=(LocalXCTermSpec("lda_x"), LocalXCTermSpec("lda_c_vwn_rpa")),
)

# Documented safe local decompositions. HF/RSH coefficients stay exclusively in
# the complete reference recipe. HSE03 has different HF and local omega.
_LOCAL_DECOMPOSITIONS = {
    "pbe0": (LocalXCTermSpec("gga_x_pbe", .75), LocalXCTermSpec("gga_c_pbe")),
    "b3lyp": (LocalXCTermSpec("lda_x", .08), LocalXCTermSpec("gga_x_b88", .72),
              LocalXCTermSpec("lda_c_vwn_rpa", .19), LocalXCTermSpec("gga_c_lyp", .81)),
    "bhandhlyp": (LocalXCTermSpec("gga_x_b88", .5), LocalXCTermSpec("gga_c_lyp")),
    "lc_wpbe": (LocalXCTermSpec("gga_x_wpbeh", 1., "runtime"), LocalXCTermSpec("gga_c_pbe")),
    "hse03": (LocalXCTermSpec("gga_x_wpbeh", 1., "fixed", 0.),
              LocalXCTermSpec("gga_x_wpbeh", -.25, "fixed", .18898815748423098),
              LocalXCTermSpec("gga_c_pbe")),
    "hse06": (LocalXCTermSpec("gga_x_wpbeh", 1., "fixed", 0.),
              LocalXCTermSpec("gga_x_wpbeh", -.25, "runtime"), LocalXCTermSpec("gga_c_pbe")),
    "wb97x_d": (LocalXCTermSpec("hyb_gga_xc_wb97x_d", 1., "fixed", .2),),
}
for _name, _terms in _LOCAL_DECOMPOSITIONS.items():
    RECIPES[_name] = replace(RECIPES[_name], semilocal=_terms)
for _name, _form in (("b97d", "D2"), ("wb97x_d", "CHG"), ("wb97x_d3", "D3")):
    RECIPES[_name] = replace(RECIPES[_name], dispersion=AdditionalTerm(_form))

# Grimme B2PLYP, JCP 124, 034108, DOI 10.1063/1.2148954.
RECIPES["b2plyp"] = FunctionalRecipe(
    "b2plyp", (LocalXCTermSpec("gga_x_b88", .47), LocalXCTermSpec("gga_c_lyp", .73)),
    ExactExchange(full_range=.53), "GGA", pt2=AdditionalTerm("MP2", .27),
    references=("https://doi.org/10.1063/1.2148954",),
)
RECIPES["hf"] = FunctionalRecipe("hf", exact_exchange=ExactExchange(full_range=1.))
RECIPES["lc_wpbe_local"] = replace(
    RECIPES["lc_wpbe"], name="lc_wpbe_local", exact_exchange=ExactExchange(),
)

ALIASES = {
    "pbeh": "pbe0", "pbe1pbe": "pbe0", "b3lypg": "b3lyp",
    "lcwpbe": "lc_wpbe", "lc_wpbe_semilocal": "lc_wpbe_local",
    "lcwpbe_local": "lc_wpbe_local", "wb97xd": "wb97x_d",
    "omega_b97x_d": "wb97x_d", "omega_b97xd": "wb97x_d",
    "m06_l": "m06l", "m11_l": "m11l", "mn12_l": "mn12l",
    "mn15_l": "mn15l", "m06_hf": "m06hf", "pbe_sol": "pbesol",
    "rev_pbe": "revpbe", "rev_tpss": "revtpss",
}
for _name, _record in _LIBXC_REFERENCE_DATA.items():
    if len(_record["terms"]) == 1 and "_xc_" in _record["terms"][0]["name"]:
        ALIASES[_record["terms"][0]["name"]] = canonical_name(_name)


def get_standard_recipe(name: str) -> FunctionalRecipe | None:
    key = canonical_name(name)
    return RECIPES.get(ALIASES.get(key, key))


def list_functional_recipes() -> tuple[FunctionalRecipe, ...]:
    return tuple(RECIPES[name] for name in sorted(RECIPES))


def component_requires_laplacian(name: str) -> bool:
    return any(record["needs_laplacian"] and any(term["name"] == name for term in record["terms"])
               for record in _LIBXC_REFERENCE_DATA.values())


def local_composite_definitions():
    """Legacy local-only factories derived from the complete definitions."""
    result = {}
    for name in ("pbe0", "pbeh", "hyb_gga_xc_pbeh", "b3lyp", "hyb_gga_xc_b3lyp",
                 "bhandhlyp", "hyb_gga_xc_bhandhlyp", "hyb_gga_xc_hse03", "hyb_gga_xc_hse06"):
        recipe = get_standard_recipe(name)
        result[name] = tuple((term.coefficient, term.name,
                              {"_omega": term.fixed_omega} if term.omega_mode == "fixed"
                              else ({"_omega": recipe.default_omega} if term.omega_mode == "runtime" else {}))
                             for term in recipe.semilocal)
    return result


# Tuning bounds and legacy presentation choices only. Physical defaults and
# local terms below are always views of RECIPES, never parallel definitions.
_RSH_TUNING = {
    "lc_wpbe": dict(omega_bounds=(.05, .80), sr_hf_bounds=(0., 0.),
                    optxc_default_omega=.205, optxc_omega_bounds=(.13, .30),
                    jax_local_xc_spec="lc_wpbe_local", strict_jax_supported=True),
    "hse03": dict(omega_bounds=(.05, .80), strict_jax_supported=True),
    "hse06": dict(omega_bounds=(.05, .80), strict_jax_supported=True),
    "wb97x_d": dict(omega_bounds=(.05, .80), sr_hf_bounds=(0., .60),
                    optxc_default_omega=.164, optxc_omega_bounds=(.10, .24),
                    jax_local_xc_spec="hyb_gga_xc_wb97x_d"),
}
_RSH_PRESETS = {}
for _recipe in RECIPES.values():
    _ex = _recipe.exact_exchange
    if _ex.omega is None:
        continue
    _name = _recipe.name.replace("_", "-")
    _source = _LIBXC_REFERENCE_DATA[_recipe.name]
    _tuning = dict(omega_bounds=(_ex.omega, _ex.omega),
                   sr_hf_bounds=(_ex.full_range+_ex.short_range,)*2,
                   jax_local_xc_spec=_recipe.name)
    _tuning.update(_RSH_TUNING.get(_recipe.name, {}))
    _RSH_PRESETS[_name] = RSHFunctionalPreset(
        name=_name, canonical_xc_name=_recipe.name.upper(),
        libxc_id=_source["terms"][0]["name"].upper(),
        exchange_form="Explicit complete-recipe SR/LR exchange",
        correlation_form="Explicit recipe local terms",
        default_sr_hf_fraction=_ex.full_range+_ex.short_range,
        default_lr_hf_fraction=_ex.full_range+_ex.long_range,
        default_omega=_ex.omega,
        lr_hf_bounds=(_ex.full_range+_ex.long_range,)*2,
        has_dispersion=_recipe.dispersion is not None,
        dispersion_form=None if _recipe.dispersion is None else _recipe.dispersion.name,
        local_term_specs=_recipe.semilocal,
        monotonic_lr_hf=_ex.long_range >= _ex.short_range,
        notes="Complete recipe metadata; local kernel/method/derivative validation remains separate.",
        **_tuning,
    )


def canonical_rsh_preset_name(name: str) -> str:
    recipe = get_standard_recipe(name)
    if recipe is None or recipe.exact_exchange.omega is None:
        raise KeyError(f"Unsupported RSH preset {name!r}. Supported presets: {', '.join(sorted(_RSH_PRESETS))}.")
    return recipe.name.replace("_", "-")


def get_rsh_functional_preset(name: str) -> RSHFunctionalPreset:
    return _RSH_PRESETS[canonical_rsh_preset_name(name)]


def list_rsh_functional_presets() -> tuple[str, ...]:
    return tuple(sorted(_RSH_PRESETS))
