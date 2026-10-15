"""End-to-end force supervision through native integrals and neural SCF."""
from pathlib import Path
import runpy

import pytest


@pytest.mark.parametrize('mode', ['implicit','explicit'])
def test_neural_scf_force_loss_gradient_and_training_step(mode):
    example = runpy.run_path(str(Path('examples/training/train_neural_scf_forces.py')))
    result = example['validate_and_train'](mode)
    assert result['converged']
    assert result['max_force_fd_error'] < 2e-8
    assert result['max_parameter_fd_error'] < 2e-12
    assert result['parameter_gradient_norm'] > 1e-8
    assert result['loss_after'] < result['loss_before']
