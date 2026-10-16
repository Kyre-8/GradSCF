"""Real closed-shell EOM-CCSD: singlet EE and doublet IP/EA sectors."""

from .types import EOMConfig, EOMResult, EOMPrecisionDiagnostics, DysonResult
from .amplitudes import EOMAmplitudeSpace
from .operators import build_eom_operator, run_eom
from .api import EOMEE, EOMIP, EOMEA
from .properties import dyson_orbitals

__all__ = [
    "EOMPrecisionDiagnostics",
    "DysonResult",
    "dyson_orbitals",
    "EOMConfig",
    "EOMResult",
    "EOMAmplitudeSpace",
    "build_eom_operator",
    "run_eom",
    "EOMEE",
    "EOMIP",
    "EOMEA",
]
