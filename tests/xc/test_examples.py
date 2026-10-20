"""Exercise public custom-XC examples without substituting a reference SCF."""
from pathlib import Path
import runpy

import pytest


@pytest.mark.parametrize('relative', [
    'xc/standard_recipes.py',
    'xc/custom_energy.py',
    'training/density_matrix_mlp.py',
    'training/external_functional.py',
])
def test_public_xc_example(relative):
    root = Path(__file__).resolve().parents[2]
    runpy.run_path(str(root/'examples'/relative), run_name='__main__')
