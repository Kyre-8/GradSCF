"""Legacy config-driven reporting workflows.

These helpers execute calculations; build_molecule runs SCF and optional TD
and returns MoleculeRun, not a bare gto.M molecule. Ordinary calculations use
mol/mf facades; full experiments use workflows.ExperimentPipeline. The helpers
remain here for explicit legacy callers with their original I/O/return types.
"""

from __future__ import annotations

from gradscf.model.neural_xc.workflows.types import (
    MoleculeSpecConfig,
    NeuralXCTrainingConfig,
    OutputConfig,
    SimulationConfig,
    SpectrumGridConfig,
)

MoleculeConfig = MoleculeSpecConfig


def build_molecule(
    molecule: MoleculeConfig,
    *,
    simulation: SimulationConfig,
):
    """Run SCF and optional TD calculations from specs; return a MoleculeRun."""

    from gradscf.model.neural_xc.workflows.core import run_molecule_from_spec

    return run_molecule_from_spec(molecule, simulation=simulation)


def run_pipeline(
    molecule: MoleculeConfig,
    *,
    training: NeuralXCTrainingConfig,
    simulation: SimulationConfig,
    spectrum: SpectrumGridConfig,
):
    """Run strict-JAX molecule -> training -> TDDFT spectrum core pipeline."""

    from gradscf.model.neural_xc.workflows.core import run_pipeline_core

    return run_pipeline_core(
        molecule_spec=molecule,
        training_config=training,
        simulation_config=simulation,
        spectrum_config=spectrum,
    )


def run_spectrum_pipeline(
    *,
    system_label: str,
    molecule: MoleculeConfig,
    training: NeuralXCTrainingConfig,
    simulation: SimulationConfig,
    spectrum: SpectrumGridConfig,
    output: OutputConfig,
):
    """Run the strict-JAX molecule spectrum pipeline and write outputs."""

    from gradscf.model.neural_xc.workflows.pipeline import run_neural_xc_spectrum_pipeline

    return run_neural_xc_spectrum_pipeline(
        system_label=system_label,
        molecule_spec=molecule,
        training_config=training,
        simulation_config=simulation,
        spectrum_config=spectrum,
        output_config=output,
    )


__all__ = [
    "MoleculeConfig",
    "build_molecule",
    "run_pipeline",
    "run_spectrum_pipeline",
]
