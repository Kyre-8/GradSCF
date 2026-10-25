"""Static closed-shell EOM-CCSD options and JAX-compatible results."""

from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple, TYPE_CHECKING
import numpy as np
import jax
from jaxtyping import Array
from ...solvers import EigenResponseConfig, LinearSolverConfig

if TYPE_CHECKING:
    from ...scf.diagnostics import RestrictedSCFDiagnostics
    from ..types import CCResult


@dataclass(frozen=True)
class EOMConfig:
    sector: str = "ee"
    nroots: int = 1
    solver: str = "dense"
    conv_tol: float = 1e-9
    gap_tol: float = 1e-8
    imaginary_tol: float = 1e-9
    max_condition: float = 1e8
    max_dense: int = 256
    block_size: int = 16
    max_intermediate_elements: int = 20_000_000
    max_cycle: int = 100
    max_space: int = 40
    guard_roots: int = 1
    seed: int = 0
    preconditioner_floor: float = 1e-8
    vector_response: bool = False
    response_config: LinearSolverConfig | None = None
    response: EigenResponseConfig | None = None

    def __post_init__(self):
        if self.sector not in {"ee", "ip", "ea"}:
            raise ValueError("EOM sector must be ee, ip or ea")
        if self.solver not in {"dense", "davidson"}:
            raise ValueError("EOM solver must be dense or davidson")
        if not isinstance(self.vector_response, bool):
            raise TypeError("vector_response must be a static bool")
        if self.response is not None:
            if not isinstance(self.response, EigenResponseConfig):
                raise TypeError("response must be an EigenResponseConfig")
            if self.vector_response or self.response_config is not None:
                raise ValueError("Use response controls or legacy AD controls, not both")
        if self.response_config is not None:
            if not isinstance(self.response_config, LinearSolverConfig):
                raise TypeError("response_config must be a LinearSolverConfig")
            if not self.vector_response:
                raise ValueError("response_config requires vector_response=True")
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        for name in (
            "nroots",
            "max_dense",
            "block_size",
            "max_intermediate_elements",
            "max_cycle",
            "max_space",
            "guard_roots",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if any(
            not isfinite(x) or x <= 0
            for x in (
                self.preconditioner_floor,
                self.conv_tol,
                self.gap_tol,
                self.imaginary_tol,
                self.max_condition,
            )
        ):
            raise ValueError("EOM tolerances must be finite and positive")


class EOMResult(NamedTuple):
    energies: Array
    right_vectors: Array
    left_vectors: Array
    residual_norms: Array
    left_residual_norms: Array
    converged: Array
    response_valid: Array
    condition_numbers: Array
    raw_eigenvalues: Array
    ground_valid: Array
    biorthogonality_error: Array
    spectrum_complete: Array
    iterations: Array
    subspace_dimension: Array
    spectral_gaps: Array
    guard_residual_norms: Array
    restarts: Array
    vector_response: bool = False
    sector: str | None = None


# Preserve the established NamedTuple/_replace interface, with the response
# contract in static pytree metadata just like BSE's amplitude_response.
jax.tree_util.register_pytree_node(
    EOMResult, lambda result: (result[:-2], (result.vector_response, result.sector)),
    lambda metadata, values: EOMResult(*values, vector_response=metadata[0], sector=metadata[1]),
)


class EOMSubspaceResult(NamedTuple):
    """Cluster invariants with a fully stopped conventional EOM diagnostic result."""
    projection: Array | None
    eigenvalue_sum: Array
    response_valid: Array
    boundary_gap: Array
    ground_valid: Array
    raw_result: EOMResult

    @property
    def energies(self):
        return None

    @property
    def right_vectors(self):
        return None

    @property
    def left_vectors(self):
        return None


class DysonResult(NamedTuple):
    """Active-MO columns; weights are one-spin signed left/right products."""

    left: Array
    right: Array
    weights: Array
    valid: Array
    response_valid: Array


@dataclass(frozen=True)
class EOMPrecisionDiagnostics:
    """Existing layer results plus residual targets; not an energy-error bound."""

    scf: "RestrictedSCFDiagnostics | None"
    cc: "CCResult"
    eom: EOMResult
    cc_tolerance: float
    eom_tolerance: float

    @property
    def cc_ok(self):
        return bool(
            self.cc.converged
            and np.isfinite(self.cc.residual_norm)
            and self.cc.residual_norm <= self.cc_tolerance
        )

    @property
    def eom_ok(self):
        return bool(
            np.all(self.eom.converged)
            and all(
                np.all(np.isfinite(r)) and np.all(r <= self.eom_tolerance)
                for r in (
                    self.eom.residual_norms,
                    self.eom.left_residual_norms,
                    self.eom.guard_residual_norms,
                )
            )
        )

    @property
    def all_passed(self):
        if self.scf is None:
            return None
        return bool(self.scf.stationary and self.cc_ok and self.eom_ok)
