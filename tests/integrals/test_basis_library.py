"""Basis family, contraction and packaged-resource coverage after consolidation."""
import json
from importlib.resources import files

import pytest

from gradscf.integrals.basis.data import load_basis_from_snapshot


@pytest.mark.parametrize("name,symbol", [
    ("sto-3g", "O"), ("6-31+g(d,p)", "C"), ("cc-pvtz", "N"),
    ("aug-cc-pvdz", "O"), ("def2-tzvp", "C"), ("ano-rcc", "O"),
    ("pcseg-2", "C"), ("dyall-v2z", "C"), ("iglo3", "H"), ("crenbl", "Na"),
])
def test_basis_families_preserve_raw_upstream_data(name, symbol):
    pyscf = pytest.importorskip("pyscf")
    assert load_basis_from_snapshot(name, symbol) == pyscf.gto.basis.load(name, symbol, optimize=False)


def test_basis_loader_does_not_share_mutable_coefficients():
    first = load_basis_from_snapshot("3-21g", "H")
    first[0][1][1] = 99.
    assert load_basis_from_snapshot("3-21g", "H")[0][1][1] == .156285


def test_all_supplemental_basis_families_are_retained():
    bundle = json.loads(files("gradscf.integrals.basis.data").joinpath("_pyscf_basis_bundle.json").read_text())
    assert set(bundle) == {"sto-3g", "6-31g", "6-31g*", "def2-svp", "cc-pvdz"}
    assert all(bundle[name] for name in bundle)
