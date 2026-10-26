"""Complete-recipe parsing and separate Neural-XC component selection."""

from dataclasses import replace
import re
from typing import Sequence

from .registry import canonical_name, component_requires_laplacian, get_standard_recipe
from .types import ExactExchange, FunctionalRecipe, LocalXCTermSpec, XCTerm
from ._jax_xc import jax_xc_functional_info

_TERM_FAMILIES = {"hf": "HF", "lda_x": "LDA", "lda_c_pw": "LDA",
                  "lda_c_vwn": "LDA", "lda_c_vwn_rpa": "LDA",
                  "gga_x_b88": "GGA", "gga_x_pbe": "GGA",
                  "gga_x_wpbeh": "GGA", "gga_c_lyp": "GGA",
                  "gga_c_pbe": "GGA", "mgga_x_r2scan": "MGGA",
                  "mgga_c_r2scan": "MGGA"}

FRIENDLY_XC_COMPONENT_ALIASES: dict[str, str] = {
    "x_lda": "lda_x",
    "pw_c": "lda_c_pw",
    "c_pw": "lda_c_pw",
    "vwn_c": "lda_c_vwn",
    "c_vwn": "lda_c_vwn",
    "vwn_rpa_c": "lda_c_vwn_rpa",
    "c_vwn_rpa": "lda_c_vwn_rpa",
    "b88_x": "gga_x_b88",
    "x_b88": "gga_x_b88",
    "pbe_x": "gga_x_pbe",
    "x_pbe": "gga_x_pbe",
    "pbe_c": "gga_c_pbe",
    "c_pbe": "gga_c_pbe",
    "lyp_c": "gga_c_lyp",
    "c_lyp": "gga_c_lyp",
    "wpbeh_x": "gga_x_wpbeh",
    "x_wpbeh": "gga_x_wpbeh",
    "rpbe_x": "gga_x_rpbe",
    "x_rpbe": "gga_x_rpbe",
    "pw91_x": "gga_x_pw91",
    "x_pw91": "gga_x_pw91",
    "pw91_c": "gga_c_pw91",
    "c_pw91": "gga_c_pw91",
    "scan_x": "mgga_x_scan",
    "x_scan": "mgga_x_scan",
    "scan_c": "mgga_c_scan",
    "c_scan": "mgga_c_scan",
    "r2scan_x": "mgga_x_r2scan",
    "x_r2scan": "mgga_x_r2scan",
    "r2scan_c": "mgga_c_r2scan",
    "c_r2scan": "mgga_c_r2scan",
    "tpss_x": "mgga_x_tpss",
    "x_tpss": "mgga_x_tpss",
    "tpss_c": "mgga_c_tpss",
    "c_tpss": "mgga_c_tpss",
}

_AMBIGUOUS_XC_COMPONENT_ALIASES: dict[str, tuple[str, ...]] = {
    "pbe": ("pbe_x", "pbe_c", "gga_x_pbe", "gga_c_pbe"),
    "scan": ("scan_x", "scan_c", "mgga_x_scan", "mgga_c_scan"),
    "r2scan": ("r2scan_x", "r2scan_c", "mgga_x_r2scan", "mgga_c_r2scan"),
    "tpss": ("tpss_x", "tpss_c", "mgga_x_tpss", "mgga_c_tpss"),
    "pw91": ("pw91_x", "pw91_c", "gga_x_pw91", "gga_c_pw91"),
}

_CANONICAL_XC_COMPONENT_PREFIXES = (
    "lda_",
    "gga_",
    "mgga_",
    "hyb_gga_",
    "hyb_mgga_",
)


def _canonical_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _parse_coefficient(token: str) -> tuple[float, str]:
    token = token.strip()
    if "*" in token:
        left, right = token.split("*", 1)
        return float(left.replace(" ", "")), right.strip()
    parts = token.split()
    if len(parts) == 2:
        try:
            return float(parts[0]), parts[1]
        except ValueError:
            pass
    return 1.0, token


def _normalize_spec(spec: str) -> str:
    return _canonical_name(spec)


def _is_kinetic_component_name(name: str) -> bool:
    return (
        name.startswith("lda_k_")
        or name.startswith("gga_k_")
        or name.startswith("mgga_k_")
    )


def _resolve_family_qualified_component(name: str) -> str | None:
    if ":" not in name:
        return None
    family, component = (part.strip() for part in name.split(":", 1))
    family = _canonical_name(family)
    component = _canonical_name(component)
    valid_families = {"lda", "gga", "mgga", "hyb_gga", "hyb_mgga"}
    if family not in valid_families:
        raise ValueError(
            f"Unsupported XC component family qualifier {family!r}. "
            "Use one of lda, gga, mgga, hyb_gga, hyb_mgga."
        )
    if component in FRIENDLY_XC_COMPONENT_ALIASES:
        resolved = FRIENDLY_XC_COMPONENT_ALIASES[component]
        if resolved.startswith(f"{family}_"):
            return resolved
    if component.endswith(("_x", "_c", "_xc")):
        base, kind = component.rsplit("_", 1)
    elif component.startswith(("x_", "c_", "xc_")):
        kind, base = component.split("_", 1)
    else:
        raise ValueError(
            f"Family-qualified XC component {name!r} must include x/c/xc, "
            "for example 'gga:lyp_c' or 'mgga:scan_x'."
        )
    return f"{family}_{kind}_{base}"


def resolve_xc_component_name(name: str) -> str:
    """Resolve a user-facing XC component name to a canonical jax_xc name."""

    key = _canonical_name(str(name))
    qualified = _resolve_family_qualified_component(key)
    if qualified is not None:
        key = qualified
    elif key in FRIENDLY_XC_COMPONENT_ALIASES:
        key = FRIENDLY_XC_COMPONENT_ALIASES[key]
    elif key in _AMBIGUOUS_XC_COMPONENT_ALIASES:
        suggestions = ", ".join(_AMBIGUOUS_XC_COMPONENT_ALIASES[key])
        raise ValueError(
            f"XC component {name!r} is ambiguous. Use one of: {suggestions}."
        )

    if _is_kinetic_component_name(key):
        raise ValueError(
            f"XC component {name!r} resolves to {key!r}, which is a kinetic-energy "
            "functional exposed by jax_xc, not an XC energy-density component."
        )
    if key == "hf" or key.startswith(_CANONICAL_XC_COMPONENT_PREFIXES):
        return key
    return key


def b3lyp_component_basis() -> tuple[str, ...]:
    """Return the explicit semilocal basis channels of the B3LYP decomposition."""

    return tuple(term.name for term in get_standard_recipe("b3lyp").semilocal)


def b3lyp_component_coefficients() -> tuple[float, ...]:
    """Return B3LYP coefficients in ``b3lyp_component_basis`` order plus HF."""

    recipe = get_standard_recipe("b3lyp")
    return tuple(term.coefficient for term in recipe.semilocal) + (recipe.exact_exchange.full_range,)


def parse_xc(spec: str, *, allow_experimental_jax_xc: bool = False) -> list[XCTerm]:
    """Legacy weighted-term view; use ``parse_recipe`` for complete metadata."""
    recipe = parse_recipe(spec, allow_experimental_jax_xc=allow_experimental_jax_xc)
    terms = []
    if recipe.exact_exchange.full_range:
        terms.append(XCTerm("hf", recipe.exact_exchange.full_range, "HF", "hf"))
    terms.extend(XCTerm(term.name, term.coefficient, _component_family(term.name))
                 for term in recipe.semilocal)
    return terms


def xc_type(spec: str, *, allow_experimental_jax_xc: bool = False) -> str:
    return parse_recipe(spec, allow_experimental_jax_xc=allow_experimental_jax_xc).kind


def hybrid_coeff(spec: str) -> float:
    """Full-range HF fraction only; RSH consumers must use complete metadata."""
    return parse_recipe(spec).exact_exchange.full_range


def semilocal_terms(spec: str, *, allow_experimental_jax_xc: bool = False) -> list[XCTerm]:
    return [
        term
        for term in parse_xc(spec, allow_experimental_jax_xc=allow_experimental_jax_xc)
        if term.kind == "semilocal"
    ]


def resolve_semilocal_xc_specs(
    spec: str | Sequence[str],
    *,
    allow_experimental_jax_xc: bool = False,
) -> tuple[str, ...]:
    """Resolve aliases/composites into semilocal energy-density channel names."""

    if isinstance(spec, str):
        raw_specs = (spec,)
    else:
        raw_specs = tuple(str(value) for value in spec)
    if not raw_specs:
        raise ValueError("semilocal_xc must contain at least one functional specification.")

    resolved: list[str] = []
    for raw_spec in raw_specs:
        # This API selects neural-model basis channels, not complete methods.
        # Keep explicit experimental hybrid local channels available here while
        # parse_recipe rejects unknown complete hybrid definitions.
        component = resolve_xc_component_name(raw_spec) if get_standard_recipe(raw_spec) is None else None
        if component is not None and component.startswith("hyb_"):
            info = jax_xc_functional_info(component)
            if info.status == "experimental" and not allow_experimental_jax_xc:
                raise ValueError(f"Local XC component {component!r} is experimental; pass allow_experimental_jax_xc=True.")
            if info.status not in {"strict", "wrapped", "experimental"}:
                raise KeyError(f"Unsupported local XC component {component!r}.")
            resolved.append(component)
            continue
        for term in semilocal_terms(
            raw_spec,
            allow_experimental_jax_xc=allow_experimental_jax_xc,
        ):
            resolved.append(term.name)
    if not resolved:
        raise ValueError(f"XC specification {spec!r} contains no semilocal terms.")
    return tuple(resolved)


def _component_family(name):
    if "mgga_" in name:
        return "MGGA"
    if "gga_" in name:
        return "GGA"
    if name.startswith("lda_"):
        return "LDA"
    return "HF" if name == "hf" else "unknown"


def _split_expression(spec):
    # Commas inside RSH(...) are parameters; top-level commas separate X/C.
    text = str(spec)
    depth, start, tokens = 0, 0, []
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char in "+," and depth == 0:
            # A plus sign in an exponent is not a term separator.
            if char == "+" and index > 1 and text[index-1] in "eE" and text[index-2] in ".0123456789":
                continue
            tokens.append(text[start:index].strip())
            start = index + 1
        elif char == "-" and depth == 0 and index > start:
            if index > 1 and text[index-1] in "eE" and text[index-2] in ".0123456789":
                continue
            if text[index-1].isspace() or re.match(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*\*", text[index+1:]):
                tokens.append(text[start:index].strip())
                start = index
        if depth < 0:
            raise ValueError(f"Unbalanced XC expression {spec!r}.")
    if depth:
        raise ValueError(f"Unbalanced XC expression {spec!r}.")
    tokens.append(text[start:].strip())
    return tuple(token for token in tokens if token)


def parse_recipe(spec: str | FunctionalRecipe, *, allow_experimental_jax_xc=False) -> FunctionalRecipe:
    """Resolve complete standard definitions or explicit weighted local/HF terms.

    ``RSH(omega, alpha, beta)`` follows alpha=LR, alpha+beta=SR. A single
    screening parameter is allowed. Unknown hybrid names never acquire an HF
    fraction from their spelling or from an upstream local-energy factory.
    """
    if isinstance(spec, FunctionalRecipe):
        return spec
    known = get_standard_recipe(spec)
    if known is not None:
        return known
    local, full, short, long, omega, refs = [], 0., 0., 0., None, []
    requirements = {"dispersion": None, "nlc": None, "pt2": None}
    needs_laplacian = False

    def add_omega(value):
        nonlocal omega
        value = float(value)
        if value < 0. or (omega is not None and abs(value-omega) > 1e-14):
            raise ValueError("An XC recipe must use one nonnegative omega.")
        omega = value

    tokens = _split_expression(spec)
    if not tokens:
        raise ValueError("XC specification must not be empty.")
    for token in tokens:
        factor, raw = _parse_coefficient(token)
        raw = raw.strip()
        if raw.startswith("-"):
            factor, raw = -factor, raw[1:].strip()
        match = re.fullmatch(r"(?i)(RSH|SR_HF|LR_HF)\(([^()]*)\)", raw)
        if match:
            values = [float(value) for value in match[2].split(",")]
            if match[1].upper() == "RSH":
                if len(values) != 3:
                    raise ValueError("RSH expects (omega, alpha, beta).")
                if values[0] == 0.:
                    full += factor*(values[1]+values[2])
                    continue
                add_omega(values[0])
                short += factor*(values[1]+values[2])
                long += factor*values[1]
            else:
                if len(values) != 1:
                    raise ValueError("SR_HF/LR_HF expects one omega.")
                if values[0] == 0.:
                    if match[1].upper() == "SR_HF":
                        full += factor
                    continue
                add_omega(values[0])
                if match[1].upper() == "SR_HF":
                    short += factor
                else:
                    long += factor
            continue
        entry = get_standard_recipe(raw)
        if entry is not None:
            local.extend(replace(term, coefficient=factor*term.coefficient) for term in entry.semilocal)
            ex = entry.exact_exchange
            full += factor*ex.full_range
            short += factor*ex.short_range
            long += factor*ex.long_range
            if ex.omega is not None:
                add_omega(ex.omega)
            refs.extend(entry.references)
            needs_laplacian |= entry.needs_laplacian
            for label in requirements:
                value = getattr(entry, label)
                if value is not None:
                    if factor != 1. or requirements[label] is not None:
                        raise ValueError("Weighted or repeated nonlocal recipe terms need an explicit physical definition.")
                    requirements[label] = value
            continue
        name = resolve_xc_component_name(raw)
        if name.startswith("hyb_"):
            raise ValueError(f"Unknown complete hybrid recipe {raw!r}; select it explicitly as a local component only.")
        if name not in _TERM_FAMILIES:
            info = jax_xc_functional_info(name)
            if info.status == "experimental" and not allow_experimental_jax_xc:
                raise ValueError(f"jax_xc component {name!r} is experimental; pass allow_experimental_jax_xc=True.")
            if info.status not in {"strict", "wrapped", "experimental"}:
                raise KeyError(f"Unsupported JAX XC component {raw!r}: {info.reason}")
        local.append(LocalXCTermSpec(name, factor, "runtime" if name == "gga_x_wpbeh" else "none"))
        needs_laplacian |= component_requires_laplacian(name)
    families = {_component_family(term.name) for term in local}
    kind = "MGGA" if "MGGA" in families else ("GGA" if "GGA" in families else ("LDA" if local else "HF"))
    return FunctionalRecipe(str(spec), tuple(local), ExactExchange(full, short, long, omega), kind,
                            references=tuple(dict.fromkeys(refs)), needs_laplacian=needs_laplacian,
                            **requirements)
