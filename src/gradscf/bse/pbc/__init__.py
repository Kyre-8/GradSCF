"""Bounded complex periodic singlet BSE-TDA and KS optical vertices.

This initial array interface is eager and forward-only. The screened direct
q=0 head/wing integration is not implemented. Reduced-D uses the original
finite-part Coulomb gauge and retains the bare finite head. These are finite-model
validation tools rather than converged material absorption predictions.
"""
from .kernel import build_tda_matrix
from .response import solve_tda, TDAResult
from .properties import (ks_dipoles, independent_particles, ks_transition_dipoles,
                         transition_dipoles, oscillator_strengths, polarizability)

__all__ = ['build_tda_matrix', 'solve_tda', 'TDAResult',
           'ks_dipoles', 'independent_particles', 'ks_transition_dipoles',
           'transition_dipoles', 'oscillator_strengths', 'polarizability']
