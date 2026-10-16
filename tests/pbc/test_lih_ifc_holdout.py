"""A sampled point or changed electronic model cannot validate interpolation."""
import copy
import importlib.util
from pathlib import Path

import pytest


def validator():
    path = Path(__file__).resolve().parents[2]/"examples/pbc/check_lih_ifc_holdout.py"
    assert path.exists(), "Held-out interpolation compatibility gate is not implemented"
    spec = importlib.util.spec_from_file_location("lih_ifc_holdout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_holdout


def reference():
    return dict(settings=dict(kmesh=[2, 2, 6], mesh=[41]*3, xc="pbe"),
        lattice_bohr=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        coordinates_bohr=[[0, 0, 0], [.5, .5, .5]],
        fractional_kpoints=[[0, 0, 0]], basis_data={"test": 1},
        pseudopotential_data=[[1], [1]], masses_amu=[7, 1],
        source_python_sha256="frozen-source")


def test_changed_electronic_kmesh_is_not_interpolation_error():
    check = validator()
    model = reference()
    changed = copy.deepcopy(model)
    changed["settings"]["kmesh"] = [2, 2, 3]
    with pytest.raises(ValueError, match="settings"):
        check(model, changed, [[0, 0, 0], [0, 0, 1/3]], [0, 0, 1/6])


def test_reciprocal_copy_of_sample_is_not_a_holdout():
    check = validator()
    model = reference()
    samples = [[0, 0, 0], [0, 0, 1/3], [0, 0, 2/3]]
    with pytest.raises(ValueError, match="sampled"):
        check(model, model, samples, [1, -1, 4/3])
    check(model, model, samples, [0, 0, 1/6])
