"""PySCF-style linear-response namespace for GradSCF."""

from .api import TDA, TDDFT
from ..tddft import (
    TDDFTResult,
    TDAResult,
    UnrestrictedTDDFTResult,
    UnrestrictedTDAResult,
)

__all__ = [
    'TDA',
    'TDDFT',
    'TDDFTResult',
    'TDAResult',
    'UnrestrictedTDDFTResult',
    'UnrestrictedTDAResult',
]
