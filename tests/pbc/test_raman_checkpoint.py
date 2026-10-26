"""The Raman example preserves numeric SCF state without NumPy object arrays."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from gradscf.pbc.scf import PeriodicSCFResult


def example():
    folder = Path(__file__).resolve().parents[2]/'examples/pbc'
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location('raman_checks', folder/'check_silicon_raman.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def state():
    matrix = np.eye(2)[None, None].repeat(2, axis=0)
    return PeriodicSCFResult(-1., matrix, matrix, np.zeros((2, 1, 2)),
                            np.ones((2, 1, 2)), matrix, True, 3, -1.)


def test_raman_reference_numeric_roundtrip(tmp_path):
    module = example()
    path = tmp_path/'reference.npz'
    original = state()
    overlap = np.eye(2)[None]
    module.save_reference(path, original, overlap)
    restored, observed_overlap = module.load_reference(path)
    assert restored.chemical_potential is None
    for key in original._fields:
        if getattr(original, key) is not None:
            np.testing.assert_array_equal(getattr(restored, key), getattr(original, key))
    np.testing.assert_array_equal(observed_overlap, overlap)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype != object for key in archive.files)


def test_raman_legacy_zero_temperature_checkpoint(tmp_path):
    module = example()
    path = tmp_path/'legacy.npz'
    original = state()
    np.savez(path, **{key: np.asarray(getattr(original, key)) for key in original._fields},
             overlap=np.eye(2)[None])
    restored, _ = module.load_reference(path, legacy_zero_temperature=True)
    assert restored.chemical_potential is None
    np.testing.assert_array_equal(restored.density_spin, original.density_spin)
    with pytest.raises(ValueError, match='legacy'):
        module.load_reference(path)
