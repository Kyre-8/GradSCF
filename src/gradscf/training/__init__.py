"""Differentiable training: use Sample and Trainer for ordinary workflows.

Detailed loss/configuration helpers remain available for advanced workflows.
"""

from .config import Sample, MolecularTrainingConfig, MolecularTrainingDatum
from .checkpoints import load_params_checkpoint, save_params_checkpoint
from .targets import (
    density_on_grid,
    density_on_grid_spin_resolved,
    dm21_scf_regularization_delta_energy,
    dm21_scf_regularization_penalty,
    xc_kernel_matching_penalty,
    density_matching_penalty,
    molecular_loss,
    predict_excitation_energies,
    predict_oscillator_strengths,
    predict_excitation_spectrum,
    predict_ground_state_total_energy,
)
from .predictors import (
    make_fixed_density_predictor,
    make_ground_state_predictor,
    make_self_consistent_predictor,
    predict_ground_state_density,
    predict_ground_state_molecule,
)
from .trainer import (
    Trainer,
)
from .excited_state_trainer import (
    ExcitedStateFineTuneConfig,
    ExcitedStateFineTuneResult,
    ExcitedStateFineTuner,
)
from .forces import (
    EnergyAndForces,
    energy_and_forces,
    force_matching_loss,
    make_force_loss_and_grad,
)

__all__ = [
    'Sample',
    'Trainer',
    'MolecularTrainingDatum',
    'MolecularTrainingConfig',
    'load_params_checkpoint',
    'save_params_checkpoint',
    'density_on_grid',
    'density_on_grid_spin_resolved',
    'dm21_scf_regularization_delta_energy',
    'dm21_scf_regularization_penalty',
    'xc_kernel_matching_penalty',
    'density_matching_penalty',
    'molecular_loss',
    'predict_excitation_energies',
    'predict_oscillator_strengths',
    'predict_excitation_spectrum',
    'predict_ground_state_density',
    'predict_ground_state_molecule',
    'predict_ground_state_total_energy',
    'make_fixed_density_predictor',
    'make_ground_state_predictor',
    'make_self_consistent_predictor',
    'ExcitedStateFineTuneConfig',
    'ExcitedStateFineTuneResult',
    'ExcitedStateFineTuner',
    'EnergyAndForces',
    'energy_and_forces',
    'force_matching_loss',
    'make_force_loss_and_grad',
]
