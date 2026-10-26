"""Static solver configuration and JIT-compatible results."""

from dataclasses import dataclass, field
from math import isfinite
from typing import NamedTuple

from jaxtyping import Array


@dataclass(frozen=True)
class LinearSolverConfig:
    method: str = "gmres"
    rtol: float = 1e-9
    atol: float = 0.0
    maxiter: int = 100
    restart: int = 20
    max_dense: int = 2048

    def __post_init__(self):
        if self.method not in {"gmres", "direct"}:
            raise ValueError("linear method must be 'gmres' or 'direct'")
        if any(not isfinite(x) or x < 0 for x in (self.rtol, self.atol)):
            raise ValueError("linear tolerances must be finite and nonnegative")
        if self.rtol == self.atol == 0:
            raise ValueError("at least one tolerance must be positive")
        if min(self.maxiter, self.restart, self.max_dense) < 1:
            raise ValueError("solver limits must be positive")


@dataclass(frozen=True)
class EigenSolverConfig:
    method: str = "davidson"
    nroots: int = 1
    atol: float = 1e-9
    maxiter: int = 100
    max_subspace: int | None = None
    gradient_mode: str = "eigenvalue_only"
    adjoint_tol: float = 1e-10
    adjoint_maxiter: int = 100
    max_dense: int = 2048
    collapse_subspace: int | None = None
    initial_guess_count: int | None = None
    max_trial_vectors: int | None = None
    seed: int = 0
    value_min: float | None = None

    def __post_init__(self):
        if self.method not in {"davidson", "dense"}:
            raise ValueError("eigen method must be 'davidson' or 'dense'")
        if not isinstance(self.nroots, int) or self.nroots < 1:
            raise ValueError("nroots must be positive")
        if self.gradient_mode not in {"eigenvalue_only", "implicit_eigenvector"}:
            raise ValueError("Unknown eigen gradient_mode")
        if any(not isfinite(x) or x <= 0 for x in (self.atol, self.adjoint_tol)):
            raise ValueError("eigen tolerances must be finite and positive")
        if min(self.maxiter, self.adjoint_maxiter, self.max_dense) < 1:
            raise ValueError("solver limits must be positive")
        if self.max_subspace is not None and self.max_subspace < self.nroots:
            raise ValueError("max_subspace must be at least nroots")

        for name in ("collapse_subspace", "initial_guess_count", "max_trial_vectors"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if self.value_min is not None and not isfinite(self.value_min):
            raise ValueError("value_min must be finite")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


@dataclass(frozen=True)
class EigenResponseConfig:
    """Static differentiated observable; no automatic switching at degeneracy."""

    target: str = "eigenvalues"
    gap_atol: float = 1e-8
    gap_rtol: float = 1e-8
    linear_config: LinearSolverConfig = field(
        default_factory=lambda: LinearSolverConfig(rtol=1e-10, restart=40)
    )

    def __post_init__(self):
        if self.target not in {"eigenvalues", "eigenpairs", "subspace"}:
            raise ValueError("Unknown eigen response target")
        if (
            any(not isfinite(x) or x < 0 for x in (self.gap_atol, self.gap_rtol))
            or self.gap_atol + self.gap_rtol == 0
        ):
            raise ValueError(
                "At least one finite spectral gap tolerance must be positive"
            )
        if not isinstance(self.linear_config, LinearSolverConfig):
            raise TypeError("linear_config must be a LinearSolverConfig")


class LinearResult(NamedTuple):
    solution: Array
    residual_norm: Array
    converged: Array
    status: Array


class TensorSumResult(NamedTuple):
    solution: Array
    residual_norm: Array
    converged: Array
    min_abs_denominator: Array


class EigenResult(NamedTuple):
    """Unified spectral result with separate primal and AD diagnostics.

    values/vectors are None in subspace mode. raw_values/raw_vectors include
    an available boundary guard and are stopped diagnostic arrays. raw_present
    marks valid interval slots; lower_boundary_gap/residual audit a value_min
    cutoff from either side (infinity/zero when no lower bound is requested). In state
    modes, derivative validity applies to the whole requested set. In subspace
    mode it depends only on the external boundary. Actual adjoint solves are
    checked when requested, not certified by this primal response_valid flag.
    """

    values: Array | None
    vectors: Array | None
    residual_norms: Array
    converged: Array
    status: Array
    response_valid: Array
    boundary_gap: Array
    raw_values: Array
    raw_vectors: Array
    raw_residual_norms: Array
    raw_present: Array
    lower_boundary_gap: Array
    lower_boundary_residual: Array
    projection: Array | None = None
    eigenvalue_sum: Array | None = None


class RPAResult(NamedTuple):
    """Stable real RPA roots, column amplitudes and nondifferentiable diagnostics.

    Dense stability_margins contains min eig(A-B), min eig(A+B); Davidson
    reports lowest Ritz estimates, with stability_residual_norms and no global
    certificate (stability_certified=False). Unstable or unresolved stability
    checks return NaN physical outputs. response_valid additionally tests the
    isolated-root gap, including the extra excluded root when available.
    """

    values: Array
    x: Array
    y: Array
    residual_norms: Array
    converged: Array
    stable: Array
    response_valid: Array
    stability_margins: Array
    stability_certified: Array | bool = False
    stability_residual_norms: Array | None = None


@dataclass(frozen=True)
class NonHermitianSolverConfig:
    nroots: int = 1
    atol: float = 1e-9
    gap_atol: float = 1e-8
    gap_rtol: float = 1e-8
    imaginary_tol: float = 1e-9
    max_condition: float = 1e8
    max_dense: int = 256
    block_size: int = 16
    method: str = "dense"
    maxiter: int = 100
    max_space: int = 40
    guard_roots: int = 1
    seed: int = 0
    preconditioner_floor: float = 1e-8

    def __post_init__(self):
        if self.method not in {"dense", "davidson"}:
            raise ValueError("Non-Hermitian method must be dense or davidson")
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
            "maxiter",
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
                self.atol,
                self.gap_atol,
                self.gap_rtol,
                self.imaginary_tol,
                self.max_condition,
            )
        ):
            raise ValueError(
                "Non-Hermitian tolerances and condition limit must be positive"
            )


class NonHermitianResult(NamedTuple):
    values: Array
    right_vectors: Array
    left_vectors: Array
    residual_norms: Array
    left_residual_norms: Array
    converged: Array
    response_valid: Array
    condition_numbers: Array
    biorthogonality_error: Array
    raw_eigenvalues: Array
    spectrum_complete: Array
    iterations: Array
    subspace_dimension: Array
    spectral_gaps: Array
    guard_residual_norms: Array
    restarts: Array


class NonHermitianSubspaceResult(NamedTuple):
    """Invariant AD outputs and a stopped legacy-shaped diagnostic result.

    Proper subspaces require a complete isolated real semisimple cluster.
    The full space instead has P=I and trace(A), even at a finite Jordan block;
    its raw_result has empty spectral arrays because no eigensolve is needed.
    """
    projection: Array | None
    eigenvalue_sum: Array
    response_valid: Array
    boundary_gap: Array
    raw_result: NonHermitianResult

    @property
    def values(self):
        return None

    @property
    def right_vectors(self):
        return None

    @property
    def left_vectors(self):
        return None
