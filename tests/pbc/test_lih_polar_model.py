"""Polar postprocessing must use the same validated finite electronic model."""
import copy
import importlib.util
from pathlib import Path

import pytest


def validator():
    path = Path(__file__).resolve().parents[2]/"examples/pbc/lih_polar_bands.py"
    assert path.exists(), "Matched-material polar input gate is not implemented"
    spec = importlib.util.spec_from_file_location("lih_polar_bands", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_models


def models():
    data = dict(settings=dict(mesh=[41]*3, kmesh=[2, 2, 3], xc="pbe"),
        lattice_bohr=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        coordinates_bohr=[[0, 0, 0], [.5, .5, .5]], fractional_kpoints=[[0, 0, 0]],
        energy_hartree=-1., basis_data={"raw": 1}, pseudopotential_data=[[1], [1]])
    reference = copy.deepcopy(data)
    reference["settings"]["displacement_bohr"] = [[0]*3]*2
    born = dict(copy.deepcopy(data), complete=True, full_tensor=True, validation_passed=True)
    dielectric = dict(copy.deepcopy(data), complete=True, full_tensor=True)
    return reference, born, dielectric


def test_electric_fft_mismatch_rejected():
    check = validator()
    reference, born, dielectric = models()
    check(reference, born, dielectric)
    dielectric["settings"]["mesh"] = [31]*3
    with pytest.raises(ValueError, match="settings"):
        check(reference, born, dielectric)


def test_electric_geometry_mismatch_rejected():
    check = validator()
    reference, born, dielectric = models()
    born["coordinates_bohr"][1][0] += .01
    with pytest.raises(ValueError, match="coordinates"):
        check(reference, born, dielectric)


def test_incomplete_born_fd_cannot_enter_material_plot():
    check = validator()
    reference, born, dielectric = models()
    born["validation_passed"] = False
    with pytest.raises(ValueError, match="validated"):
        check(reference, born, dielectric)
