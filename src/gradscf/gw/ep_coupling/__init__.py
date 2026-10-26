"""Fixed harmonic phonons and molecular Fan/Debye--Waller kernels.

Energies, linear vertices g, and quadratic vertices Lambda use Hartree.
X_l = b_l + b_l^dagger, H_ep = sum_l g_l X_l and
H_ep^(2) = (1/2) sum_lm Lambda_lm X_l X_m. D = -<T X X>.
Vertices are supplied in one fixed orthonormal orbital frame; this module
never interprets them as AO derivatives or generates phonon normal modes.
The model is external and fixed during electronic self-consistency, while
its array fields remain differentiable inputs. No phonon feedback, tadpole,
acoustic sum rule, or complete electron--phonon total energy is supplied.
"""

from .types import PhononModel, PeriodicPhononModel, validate_model, validate_periodic_model
from .phonon import bose_occupation, phonon_propagator_iw, phonon_propagator_tau
from .self_energy import debye_waller, fan_self_energy, periodic_fan_self_energy_tau, periodic_fan_self_energy
from .spectral import fan_retarded, electron_linewidth, fan_linewidth, spectral_function


__all__ = ["PhononModel", "PeriodicPhononModel", "validate_model", "validate_periodic_model", "bose_occupation", "phonon_propagator_iw",
           "phonon_propagator_tau", "debye_waller", "fan_self_energy", "periodic_fan_self_energy_tau", "periodic_fan_self_energy",
           "fan_retarded", "electron_linewidth", "fan_linewidth", "spectral_function"]
