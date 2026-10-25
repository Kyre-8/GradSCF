"""Configuration and JAX-compatible results for real molecular CI."""
from dataclasses import dataclass
from typing import NamedTuple

from jaxtyping import Array


@dataclass(frozen=True)
class CIConfig:
    """CI roots and response; None max_space selects a bounded Davidson basis.

    The default capacity is min(ndet, max(40, 4*(nroots+1))).
    """
    nroots: int = 1
    solver: str = "davidson"
    conv_tol: float = 1e-9
    max_cycle: int = 100
    max_space: int | None = None
    gradient_mode: str = "eigenvalue_only"
    adjoint_tol: float = 1e-10
    adjoint_max_cycle: int = 100

    def __post_init__(self):
        if not isinstance(self.nroots, int) or self.nroots < 1:
            raise ValueError("nroots must be a positive integer")
        if self.solver not in {"dense", "davidson"}:
            raise ValueError("solver must be 'dense' or 'davidson'")
        if self.gradient_mode not in {"eigenvalue_only", "implicit_eigenvector"}:
            raise ValueError("Unknown CI gradient_mode")
        if self.conv_tol <= 0 or self.adjoint_tol <= 0:
            raise ValueError("CI and adjoint tolerances must be positive")
        if self.max_cycle < 1 or self.adjoint_max_cycle < 1:
            raise ValueError("Iteration limits must be positive")
        if self.max_space is not None and self.max_space < self.nroots:
            raise ValueError("max_space must be at least nroots")


class CIResult(NamedTuple):
    """All energies in Hartree; coefficients have shape (ndet, nroots).

    Coefficient derivatives require gradient_mode='implicit_eigenvector'.
    The determinant space fixes (Nalpha,Nbeta), and does not select total spin S.
    """
    total_energies: Array
    correlation_energies: Array
    reference_energy: Array
    coefficients: Array
    residual_norms: Array
    converged: Array


class CISResult(NamedTuple):
    """Spin-adapted amplitudes (root, occupied, virtual), each of unit norm."""
    excitation_energies: Array
    amplitudes: Array
    residual_norms: Array
    converged: Array
    amplitude_response: bool = False
    singlet: bool = True


class UCISResult(NamedTuple):
    """Spin-conserving singles; amplitudes=(alpha,beta), with root axis first."""
    excitation_energies: Array
    amplitudes: object
    residual_norms: Array
    converged: Array
    amplitude_response: bool = False


class CISDCorrectionResult(NamedTuple):
    """CIS(D) excitation energies; invalid denominators produce NaN values."""
    excitation_energies: Array
    corrections: Array
    cis_energies: Array
    min_abs_denominators: Array
    valid: Array


@dataclass(frozen=True)
class CIReference:
    """Explicit real spatial MO Hamiltonian in chemists' ERI notation.

    Occupied orbitals must precede virtual orbitals. Energies are Hartree.
    mo_energy is needed only for canonical-HF perturbative corrections.
    The caller is responsible for supplying an orthonormal orbital basis.
    """
    h1: Array
    eri: Array
    nocc: int
    nuclear_repulsion: float | Array = 0.0
    mo_energy: Array | None = None
