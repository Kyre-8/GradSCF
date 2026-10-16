"""An independent warm-start may change FFT resolution, not the model."""
import copy
import importlib.util
from pathlib import Path

import pytest


def adapter():
    path = Path(__file__).resolve().parents[2]/"examples/pbc/compare_lih_complex_q.py"
    spec = importlib.util.spec_from_file_location("complex_q_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "assert_compatible_seed"), "FFT-only independent seed adapter is not implemented"
    return module


def model():
    return dict(settings=dict(mesh=[31]*3, kmesh=[2, 2, 3], xc="pbe", basis="raw"),
        lattice_bohr=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        coordinates_bohr=[[0., 0., 0.], [.5, .5, .5]],
        fractional_kpoints=[[0., 0., 0.]], basis_data={"fixture": 1},
        pseudopotential_data=[[1.], [1.]], nao=3, explicit_nelectron=4)


def test_only_fft_resolution_can_change():
    module = adapter()
    old = model()
    new = copy.deepcopy(old)
    new["settings"]["mesh"] = [41]*3
    module.assert_compatible_seed(old, new)
    new["settings"]["kmesh"] = [2, 2, 6]
    with pytest.raises(ValueError, match="settings"):
        module.assert_compatible_seed(old, new)


def test_changed_geometry_cannot_reuse_seed():
    module = adapter()
    old = model()
    new = copy.deepcopy(old)
    new["coordinates_bohr"][1][0] += .01
    with pytest.raises(ValueError, match="coordinates"):
        module.assert_compatible_seed(old, new)
