"""The grid supplier uses the same longstanding PyTree API as native bases."""
import subprocess
import sys


def test_ao_grid_registration_does_not_require_new_jax_dataclass_api():
    code = """
import importlib.util
from pathlib import Path
import sys
import jax
import numpy as np
import gradscf.integrals.grids.ao as ao

# Load under a fresh module name even if package initialization already loaded
# blocks. Older supported JAX versions do not have register_dataclass.
if hasattr(jax.tree_util, 'register_dataclass'):
    del jax.tree_util.register_dataclass
name = 'gradscf.integrals.grids._compat_probe'
spec = importlib.util.spec_from_file_location(name, Path(ao.__file__).with_name('blocks.py'))
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
grid = module.AOGrid(np.zeros((2, 3)), np.ones(2), np.zeros((1, 3)),
                     np.ones((1, 1)), np.ones((1, 1)), (1,), ((0, 0, 0),), 2)
leaves, structure = jax.tree_util.tree_flatten(grid)
assert len(leaves) == 5
restored = jax.tree_util.tree_unflatten(structure, leaves)
assert (restored.nprims, restored.angulars, restored.block_size) == ((1,), ((0, 0, 0),), 2)
assert restored.coords is grid.coords and restored.coefficients is grid.coefficients
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
