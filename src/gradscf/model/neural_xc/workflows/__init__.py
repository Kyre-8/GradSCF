"""Reusable workflow utilities for training and spectrum benchmarking."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_PUBLIC_EXPORTS = {
    "ExperimentConfig": "config",
    "SystemConfig": "config",
    "ExperimentPipeline": "pipeline",
    "ExperimentRun": "pipeline",
    "MoleculeRun": "types",
    "MoleculeSpecConfig": "types",
    "NeuralExcitedStateRun": "types",
    "NeuralXCTrainingConfig": "types",
    "OutputConfig": "types",
    "OutputPaths": "types",
    "PipelineRun": "types",
    "SimulationConfig": "types",
    "SpectrumGridConfig": "types",
    "SpectrumRun": "types",
    "TrainingRun": "types",
}

_SUBMODULES = ("core", "pipeline", "presets", "reporting", "types", "config")

__all__ = list(_PUBLIC_EXPORTS) + list(_SUBMODULES)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name not in _PUBLIC_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{_PUBLIC_EXPORTS[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
