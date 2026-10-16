"""Molecular static BSE configurations and first-order response contracts."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from ..scf._pytree import pytree_dataclass
from ..solvers import LinearSolverConfig


@dataclass(frozen=True)
class BSEConfig:
    nroots: int = 3
    singlet: bool | None = True
    tda: bool = True
    solver: str = "davidson"
    conv_tol: float = 1e-9
    max_cycle: int = 100
    max_space: int = 40
    gradient_mode: str = "implicit_eigenvector"
    adjoint_tol: float = 1e-10
    adjoint_max_cycle: int = 100
    gap_tol: float = 1e-8
    max_dense: int = 256
    max_aux: int = 1024
    max_factor_elements: int = 20_000_000
    block_size: int = 16
    seed: int = 0
    screening_config: LinearSolverConfig | None = None
    response_config: LinearSolverConfig | None = None

    def __post_init__(self):
        for name in ("screening_config", "response_config"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, LinearSolverConfig):
                raise TypeError(f"{name} must be a LinearSolverConfig")
        if (self.singlet is not None and type(self.singlet) is not bool) or type(self.tda) is not bool:
            raise ValueError("singlet must be boolean or None; tda must be boolean")
        if self.solver not in {"dense", "davidson"} or self.gradient_mode not in {
            "eigenvalue_only",
            "implicit_eigenvector",
        }:
            raise ValueError("Unsupported BSE solver or gradient mode")
        for name in (
            "nroots",
            "max_cycle",
            "max_space",
            "adjoint_max_cycle",
            "max_dense",
            "max_aux",
            "max_factor_elements",
            "block_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if any(
            not isfinite(x) or x <= 0
            for x in (self.conv_tol, self.adjoint_tol, self.gap_tol)
        ):
            raise ValueError("BSE tolerances must be finite and positive")


@pytree_dataclass(static_fields=("singlet", "amplitude_response"))
@dataclass(frozen=True)
class BSEResult:
    excitation_energies: object
    x_amplitudes: object
    y_amplitudes: object
    residual_norms: object
    converged: object
    stable: object
    response_valid: object
    screening_valid: object
    min_screening_gap: object
    singlet: bool | None
    amplitude_response: bool
    stability_margins: object = None
    stability_certified: object = False
    stability_residual_norms: object = None
