"""Model-owned execution must work independently of repository CLI scripts."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import gradscf


def test_training_and_workflows_are_owned_by_models():
    assert 'training' not in gradscf.__all__
    assert 'workflows' not in gradscf.__all__
    assert importlib.util.find_spec('gradscf.training') is None
    assert importlib.util.find_spec('gradscf.workflows') is None
    from gradscf.model.neural_xc.training import Sample, Trainer
    from gradscf.model.neural_xc.workflows import ExperimentPipeline
    from gradscf.model.nnao.training import BasisRHF, NativeJointEngine
    assert callable(Sample) and callable(Trainer) and callable(ExperimentPipeline)
    assert BasisRHF.__module__ == 'gradscf.model.nnao.training.objective'
    assert NativeJointEngine.__module__ == 'gradscf.model.nnao.training.native'


def test_nnao_does_not_load_repository_tools_as_a_library():
    root = Path(__file__).resolve().parents[1]/'src/gradscf/model/nnao'
    for directory in ('training', 'workflows', 'integrals'):
        for path in (root/directory).glob('*.py'):
            assert 'spec_from_file_location' not in path.read_text(), path
    assert not (root/'setup.py').exists()
    assert not (root/'mace_jax').exists()
    assert (root/'vendor/mace-jax/mace_jax/__init__.py').is_file()


def test_force_helpers_are_independent_of_model_training():
    from gradscf.scf.forces import energy_and_forces
    assert energy_and_forces.__module__ == 'gradscf.scf.forces'


def test_nnao_workflows_can_configure_devices_before_importing_jax():
    code = ('import sys; '
            'from gradscf.model.nnao.workflows import joint_ad, train, validate; '
            'assert "jax" not in sys.modules; assert "mace_jax" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', code],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, PYTHONPATH='src'), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
