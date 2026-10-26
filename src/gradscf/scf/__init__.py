"""Pure-JAX SCF solvers."""

from .reference import as_reference
from .autodiff import (
    SCFDifferentiationConfig,
)
from .differentiable import (
    DifferentiableSCF,
    DifferentiableSCFConfig,
    DifferentiableSCFInfo,
)

from .rhf import (
    RHFConfig,
    RHFResult,
    nuclear_repulsion_energy,
    run_rhf,
    run_rhf_from_integrals,
)
from .rks import (
    RKSConfig,
    RKSResult,
    run_rks_from_integrals,
)
from .uks import (
    UKSConfig,
    UKSResult,
    run_uks_from_integrals,
)
from .facade import RHF, RKS, UKS
from .multistart import (
    RestrictedSCFAttempt, RestrictedMultistartResult, run_restricted_multistart,
)
from .uhf import UHF, UHFConfig, UHFResult, run_uhf, run_uhf_from_integrals
from .stability import (
    OrbitalStabilityResult, restricted_stability, unrestricted_stability,
    StabilityAttempt, SCFStabilizationResult, stabilize_scf,
    UnrestrictedStabilityResult, UnrestrictedStabilizationResult,
    UHFStabilityResult, UHFStabilizationResult, uhf_stability, stabilize_uhf_from_integrals,
    uks_stability, stabilize_uks_from_integrals,
)
from .rohf import ROHF, ROHFConfig, ROHFResult, run_rohf, run_rohf_from_integrals
from .roks import ROKS, ROKSConfig, ROKSResult, run_roks_from_integrals
from .ghf import GHF, GHFConfig, GHFResult, run_ghf, run_ghf_from_integrals
from .gks import GKS, GKSConfig, GKSResult, run_gks_from_integrals
from .orbital_optimization import (
    OrbitalOptimizationResult,
    minimize_uks_from_integrals,
    minimize_roks_from_integrals,
    minimize_gks_from_integrals,
)
from .molecules import QuadratureGrid, RestrictedMolecule, UnrestrictedMolecule

from .diagnostics import RestrictedSCFDiagnostics, restricted_scf_diagnostics

__all__ = [
    'as_reference',
    'StabilityAttempt',
    'SCFStabilizationResult',
    'stabilize_scf',
    'unrestricted_stability',
    'OrbitalStabilityResult',
    'restricted_stability',
    'RestrictedSCFDiagnostics',
    'restricted_scf_diagnostics',
    'RestrictedSCFAttempt',
    'RestrictedMultistartResult',
    'run_restricted_multistart',
    'OrbitalOptimizationResult',
    'minimize_uks_from_integrals',
    'minimize_roks_from_integrals',
    'minimize_gks_from_integrals',
    'SCFDifferentiationConfig',
    'DifferentiableSCF',
    'DifferentiableSCFConfig',
    'DifferentiableSCFInfo',
    'RHF',
    'RHFConfig',
    'RHFResult',
    'nuclear_repulsion_energy',
    'run_rhf',
    'run_rhf_from_integrals',
    'UHF',
    'UHFConfig',
    'UHFResult',
    'UnrestrictedStabilityResult',
    'UnrestrictedStabilizationResult',
    'uks_stability',
    'stabilize_uks_from_integrals',
    'UHFStabilityResult',
    'UHFStabilizationResult',
    'uhf_stability',
    'stabilize_uhf_from_integrals',
    'run_uhf',
    'run_uhf_from_integrals',
    'ROHF',
    'ROHFConfig',
    'ROHFResult',
    'run_rohf',
    'run_rohf_from_integrals',
    'ROKS',
    'ROKSConfig',
    'ROKSResult',
    'run_roks_from_integrals',
    'GHF',
    'GHFConfig',
    'GHFResult',
    'run_ghf',
    'run_ghf_from_integrals',
    'GKS',
    'GKSConfig',
    'GKSResult',
    'run_gks_from_integrals',
    'RKSConfig',
    'RKSResult',
    'RKS',
    'QuadratureGrid',
    'RestrictedMolecule',
    'UnrestrictedMolecule',
    'run_rks_from_integrals',
    'UKSConfig',
    'UKSResult',
    'UKS',
    'run_uks_from_integrals',
]
