"""Canonical molecular MP configuration and JAX result containers."""
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import NamedTuple


@dataclass(frozen=True)
class MPConfig:
    order: int = 2
    with_t2: bool = True
    denominator_tol: float = 1e-10
    canonical_tol: float = 1e-8
    algorithm: str = "auto"
    max_determinants: int = 5000
    wavefunction_order: int | None = None
    with_coefficients: bool = False
    max_tensor_elements: int = 2_000_000
    orbital_basis: str = "canonical"

    def __post_init__(self):
        if isinstance(self.order, bool) or not isinstance(self.order, Integral) or self.order < 2:
            raise ValueError("MP order must be an integer at least 2")
        if self.algorithm not in {"auto", "direct", "series", "tensor"}:
            raise ValueError("MP algorithm must be auto, direct, series or tensor")
        if self.orbital_basis not in {"canonical", "noncanonical"}:
            raise ValueError("orbital_basis must be canonical or noncanonical")
        if self.algorithm == "direct" and self.order > 3:
            raise ValueError("Direct MP kernels support only orders 2 and 3")
        if self.algorithm == "direct" and self.order == 3 and self.orbital_basis == "noncanonical":
            raise ValueError("Noncanonical MP3 requires series or tensor rather than direct")
        if self.algorithm == "direct" and (self.with_coefficients or self.wavefunction_order is not None):
            raise ValueError("Wavefunction coefficients require the series or tensor algorithm")
        if (isinstance(self.max_determinants, bool) or not isinstance(self.max_determinants, Integral)
                or self.max_determinants < 1):
            raise ValueError("max_determinants must be a positive integer")
        if self.wavefunction_order is not None and (
            isinstance(self.wavefunction_order, bool) or not isinstance(self.wavefunction_order, Integral)
            or self.wavefunction_order < self.order // 2):
            raise ValueError("wavefunction_order must be an integer at least order//2")
        if (isinstance(self.max_tensor_elements, bool) or not isinstance(self.max_tensor_elements, Integral)
                or self.max_tensor_elements < 1):
            raise ValueError("max_tensor_elements must be a positive integer")
        if any(not isfinite(x) or x <= 0
               for x in (self.denominator_tol, self.canonical_tol)):
            raise ValueError("MP tolerances must be positive and finite")

    def _use_series(self, unrestricted=False):
        return (self.algorithm in {"series", "tensor"} or self.order > 3 or self.with_coefficients
                or self.wavefunction_order is not None
                or (self.order == 3 and self.orbital_basis == "noncanonical")
                or (unrestricted and self.order == 3 and self.algorithm == "auto"))


class MPResult(NamedTuple):
    """Hartree-valued MP corrections and input validity diagnostics.

    corrections contains E2 through the requested order; e2/e3 are shortcuts.
    correlation_energy is the sum of corrections. t2 contains
    T2^(1), even for MP3. same_spin_energy/opposite_spin_energy decompose E2
    only. Invalid inputs produce NaN total/correlation energy. Reference values,
    diagnostics and still-valid lower-order components may remain available.
    wavefunction_coefficients uses the selected space's orthonormal coordinates:
    determinant order for series, packed quasiparticle blocks for tensor.
    Both use intermediate normalization; see SERIES.md and TENSOR_BLOCKS.md.
    """
    total_energy: object
    correlation_energy: object
    reference_energy: object
    e2: object
    e3: object
    same_spin_energy: object
    opposite_spin_energy: object
    t2: object
    min_abs_denominator: object
    canonical_error: object
    valid: object
    corrections: object = None
    wavefunction_coefficients: object = None
    residual_norm: object = None
