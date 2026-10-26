"""Complete molecular XC recipe and exact range-separated integral contract."""
import math

import jax.numpy as jnp

from ..dft.xc import RSHFunctionalPreset, get_rsh_functional_preset, parse_recipe


def rsh_preset(xc_spec):
    """Recognize the supported complete method, not a local XC channel."""
    xc_spec = getattr(xc_spec, "spec", xc_spec)
    try:
        preset = get_rsh_functional_preset(xc_spec)
    except KeyError:
        if not isinstance(xc_spec, str):
            return None
        try:
            recipe = parse_recipe(xc_spec)
        except (KeyError, ValueError):
            return None
        coefficients = recipe.to_range_separated_hybrid_coefficients()
        if coefficients is None:
            return None
        omega, long_range, short_range = coefficients
        if omega <= 0.:
            raise ValueError("Range-separated molecular assembly requires positive omega; use full-range HF terms at zero.")
        # A compatibility view over the one authoritative parsed recipe.
        # The local evaluator keeps the complete expression, so no term or
        # coefficient is reconstructed from a display name or HF fraction.
        return RSHFunctionalPreset(
            name=recipe.name, canonical_xc_name=recipe.name, libxc_id="custom",
            exchange_form="Explicit recipe SR/LR exchange", correlation_form="Explicit recipe local terms",
            default_sr_hf_fraction=short_range, default_lr_hf_fraction=long_range,
            default_omega=omega, omega_bounds=(omega, omega),
            sr_hf_bounds=(short_range, short_range), lr_hf_bounds=(long_range, long_range),
            has_dispersion=recipe.dispersion is not None,
            dispersion_form=None if recipe.dispersion is None else recipe.dispersion.name,
            jax_local_xc_spec=xc_spec, local_term_specs=recipe.semilocal,
            monotonic_lr_hf=long_range >= short_range)
    return preset


def local_xc_spec(xc_spec):
    preset = rsh_preset(xc_spec)
    return (preset.jax_local_xc_spec or xc_spec) if preset is not None else xc_spec


def _require_direct_rsh_provider(provider):
    from ..integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis

    if type(provider) not in (NativeDirectBasis, ProjectedNativeDirectBasis):
        raise NotImplementedError("Direct range-separated J/K requires a canonical native provider.")


def _empty_eri(value):
    return value is None or int(jnp.size(value)) == 0


def _direct_rsh_jk(provider, density, coefficients, *, with_j=True, with_k=True,
                   screening_threshold=0., workers=1):
    """Full Coulomb and sr*K(full)+(lr-sr)*K(omega), for any density batch."""
    _require_direct_rsh_provider(provider)
    from ..integrals.backends.native.jk import _selected_jk, _validate_selection

    _validate_selection(with_j, with_k)
    if len(coefficients) != 3:
        raise ValueError("RSH J/K requires (omega, long_range, short_range) coefficients.")
    omega, long_range, short_range = tuple(float(value) for value in coefficients)
    if not all(math.isfinite(value) for value in (omega, long_range, short_range)) or omega <= 0.:
        raise ValueError("RSH J/K requires finite coefficients and positive omega.")
    density = jnp.asarray(density)
    coulomb = exchange = jnp.zeros_like(density)
    controls = dict(screening_threshold=screening_threshold, workers=workers)
    if with_j or (with_k and short_range != 0.):
        coulomb, full_k = _selected_jk(provider, density, with_j=with_j,
            with_k=with_k and short_range != 0., omega=0., **controls)
        if with_k:
            exchange = short_range*full_k
    if with_k and long_range != short_range:
        _, long_k = _selected_jk(provider, density, with_j=False, with_k=True,
                                 omega=omega, **controls)
        exchange = exchange + (long_range-short_range)*long_k
    return coulomb, exchange


def validate_rsh_inputs(xc_spec, eri, eri_lr, dispersion_energy, *, jk_backend="full", direct_basis=None):
    """Reject missing physical contributions before exposing a complete name."""
    recipe = parse_recipe(xc_spec)
    if recipe.needs_laplacian:
        raise NotImplementedError("Molecular standard XC currently supports tau-dependent MGGA, not density Laplacians.")
    if recipe.dispersion is not None and dispersion_energy is None:
        raise ValueError(f"Complete {xc_spec} requires explicit dispersion_energy (zero is allowed).")
    if recipe.dispersion is None and dispersion_energy is not None:
        raise ValueError("dispersion_energy requires a complete XC recipe with a dispersion term.")
    recipe.require_complete_support(provided=() if dispersion_energy is None else ("dispersion",))
    if dispersion_energy is not None and jnp.ndim(dispersion_energy) != 0:
        raise ValueError("dispersion_energy must be a scalar in Hartree.")
    preset = rsh_preset(xc_spec)
    if preset is None:
        if recipe.exact_exchange.omega is not None:
            raise NotImplementedError("Molecular range-separated assembly requires a complete named XC recipe.")
        if eri_lr is not None:
            raise ValueError("eri_lr requires a complete range-separated XC recipe.")
        return None
    if jk_backend == "direct":
        _require_direct_rsh_provider(direct_basis)
        if not _empty_eri(eri) or not _empty_eri(eri_lr):
            raise ValueError("Direct range-separated XC requires empty eri and eri_lr leaves.")
        return preset
    if jk_backend != "full":
        raise NotImplementedError("Range-separated XC requires full ERIs or canonical native direct J/K.")
    if eri_lr is None:
        raise ValueError(f"Complete {xc_spec} requires explicit eri_lr.")
    if _empty_eri(eri) or jnp.ndim(eri) != 4 or len(set(jnp.shape(eri))) != 1:
        raise ValueError("Complete range-separated XC requires a full four-index Coulomb eri.")
    if jnp.shape(eri_lr) != jnp.shape(eri):
        raise ValueError("eri_lr must have the same full AO shape as eri.")
    return preset


def response_rsh_coefficients(source, *, xc_spec=None):
    """Validate exact AO exchange leaves; a coefficient tuple is not a local recipe.

    A named TD adapter separately checks its complete recipe against this tuple.
    Legacy omegaB97X-D array references retain the required dispersion leaf.
    """
    coefficients = getattr(source, "range_separated_coefficients", None)
    eri_lr = getattr(source, "eri_lr", None)
    if coefficients is None and eri_lr is None:
        return None
    if coefficients is None or len(coefficients) != 3:
        raise ValueError("Range-separated response requires (omega, long_range, short_range) coefficients.")
    coefficients = tuple(float(value) for value in coefficients)
    if not all(math.isfinite(value) for value in coefficients) or coefficients[0] <= 0.:
        raise ValueError("Range-separated response requires finite coefficients and positive omega.")
    eri = getattr(source, "rep_tensor", None)
    direct_basis = getattr(source, "direct_basis", None)
    if direct_basis is not None:
        _require_direct_rsh_provider(direct_basis)
        if not _empty_eri(eri) or not _empty_eri(eri_lr):
            raise ValueError("Direct range-separated response requires empty eri and eri_lr leaves.")
    else:
        if _empty_eri(eri) or jnp.ndim(eri) != 4 or len(set(jnp.shape(eri))) != 1:
            raise ValueError("Range-separated response requires full AO ERIs or a canonical native provider.")
        if eri_lr is None or jnp.shape(eri_lr) != jnp.shape(eri):
            raise ValueError("eri_lr must have the same full AO shape as eri.")
    dispersion = getattr(source, "dispersion_energy", None)
    if xc_spec is None:
        xc_spec = getattr(source, "xc", None) or getattr(source, "xc_spec", None)
    if xc_spec is None and coefficients == rsh_preset("wb97xd").to_range_separated_hybrid_coefficients():
        xc_spec = "wb97xd"
    if xc_spec is not None:
        preset = validate_rsh_inputs(xc_spec, eri, eri_lr, dispersion,
            jk_backend="direct" if direct_basis is not None else "full", direct_basis=direct_basis)
        if preset is None or preset.to_range_separated_hybrid_coefficients() != coefficients:
            raise ValueError("RSH reference and complete response functional must have matching coefficients.")
    elif dispersion is not None and jnp.ndim(dispersion) != 0:
        raise ValueError("dispersion_energy must be a scalar in Hartree.")
    return coefficients


def reject_rsh_response(source):
    """Keep single-exchange response consumers outside the new SCF boundary."""
    if (rsh_preset(getattr(source, "xc", "")) is not None
            or getattr(source, "range_separated_coefficients", None) is not None
            or getattr(source, "eri_lr", None) is not None):
        raise NotImplementedError(
            "General range-separated reference response is not supported by this consumer; "
            "use a dedicated molecular TD/GW adapter or explicit integral inputs "
            "with the orbital SCF derivative solver.")
