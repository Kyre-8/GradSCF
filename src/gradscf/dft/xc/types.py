"""Immutable XC recipes; densities and kinetic densities use atomic units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
from jaxtyping import Array

@dataclass(frozen=True)
class RestrictedFeatureBundle:
    """Spin semilocal fields (historical name); tau_sigma = 1/2 sum |grad psi|²."""

    rho_a: Array
    rho_b: Array
    sigma_aa: Array
    sigma_ab: Array
    sigma_bb: Array
    tau_a: Array
    tau_b: Array

    @property
    def rho(self) -> Array:
        return self.rho_a + self.rho_b

    @property
    def sigma(self) -> Array:
        return self.sigma_aa + 2.0 * self.sigma_ab + self.sigma_bb


try:
    jax.tree_util.register_dataclass(RestrictedFeatureBundle)
except TypeError:

    def _restricted_feature_bundle_unflatten(aux_data, children):
        del aux_data
        return RestrictedFeatureBundle(*children)

    jax.tree_util.register_pytree_node(
        RestrictedFeatureBundle,
        lambda x: (
            [x.rho_a, x.rho_b, x.sigma_aa, x.sigma_ab, x.sigma_bb, x.tau_a, x.tau_b],
            None,
        ),
        _restricted_feature_bundle_unflatten,
    )


@dataclass(frozen=True)
class XCTerm:
    name: str
    coefficient: float
    family: str
    kind: str = "semilocal"


@dataclass(frozen=True)
class LocalXCTermSpec:
    name: str
    coefficient: float = 1.0
    omega_mode: Literal["none", "fixed", "runtime"] = "none"
    fixed_omega: float | None = None


@dataclass(frozen=True)
class RSHFunctionalPreset:
    """Strict literature/libxc metadata for a named RSH XC functional."""

    name: str
    canonical_xc_name: str
    libxc_id: str
    exchange_form: str
    correlation_form: str
    default_sr_hf_fraction: float
    default_lr_hf_fraction: float
    default_omega: float
    omega_bounds: tuple[float, float]
    sr_hf_bounds: tuple[float, float]
    lr_hf_bounds: tuple[float, float] = (1.0, 1.0)
    optxc_default_omega: float | None = None
    optxc_omega_bounds: tuple[float, float] | None = None
    has_dispersion: bool = False
    dispersion_form: str | None = None
    jax_local_xc_spec: str | None = None
    local_term_specs: tuple[LocalXCTermSpec, ...] = ()
    strict_jax_supported: bool = False
    monotonic_lr_hf: bool = True
    notes: str = ""

    def to_range_separated_coefficients(self) -> tuple[float, float, float]:
        return (
            self.default_omega,
            self.default_lr_hf_fraction,
            self.default_sr_hf_fraction - self.default_lr_hf_fraction,
        )

    def to_range_separated_hybrid_coefficients(self) -> tuple[float, float, float]:
        return (
            self.default_omega,
            self.default_lr_hf_fraction,
            self.default_sr_hf_fraction,
        )


@dataclass(frozen=True)
class ExactExchange:
    """Coefficients of full Coulomb, erfc(omega*r)/r and erf(omega*r)/r HF."""

    full_range: float = 0.0
    short_range: float = 0.0
    long_range: float = 0.0
    omega: float | None = None


@dataclass(frozen=True)
class AdditionalTerm:
    """A physical contribution that a local quadrature cannot supply."""

    name: str
    coefficient: float = 1.0
    references: tuple[str, ...] = ()
    parameters: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class FunctionalRecipe:
    """Complete definition, independent of installed kernels or method support.

    ``semilocal`` is explicitly only the local contribution. A callable local
    channel does not establish an available SCF/response implementation or a
    validated derivative order for this complete recipe.
    """

    name: str
    semilocal: tuple[LocalXCTermSpec, ...] = ()
    exact_exchange: ExactExchange = ExactExchange()
    kind: str = "HF"
    dispersion: AdditionalTerm | None = None
    nlc: AdditionalTerm | None = None
    pt2: AdditionalTerm | None = None
    references: tuple[str, ...] = ()
    needs_laplacian: bool = False
    validation_status: str = "unverified"
    max_validated_derivative_order: int | None = None

    @property
    def default_omega(self) -> float | None:
        return self.exact_exchange.omega

    @property
    def local_term_specs(self) -> tuple[LocalXCTermSpec, ...]:
        return self.semilocal

    def require_complete_support(self, *, provided=()) -> None:
        """Reject missing extra terms before a consumer assembles total energy.

        ``provided`` names capabilities actually implemented by that consumer;
        this check does not validate its HF/RSH assembly or its derivatives.
        """
        supplied = {str(value).lower() for value in provided}
        missing = []
        for label, term in (("dispersion", self.dispersion), ("NLC", self.nlc),
                            ("PT2", self.pt2)):
            if term is not None and label.lower() not in supplied:
                missing.append(f"{label} ({term.name})")
        if missing:
            raise NotImplementedError(
                f"Complete XC recipe {self.name!r} requires " + ", ".join(missing) +
                "; a semilocal channel alone is not this functional."
            )

    def to_range_separated_hybrid_coefficients(self) -> tuple[float, float, float] | None:
        ex = self.exact_exchange
        if ex.omega is None:
            return None
        return ex.omega, ex.full_range + ex.long_range, ex.full_range + ex.short_range
