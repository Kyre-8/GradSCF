"""The AO-to-MO implementation has one owning domain package."""

import importlib
from pathlib import Path


def test_ao2mo_is_a_public_domain_package():
    import gradscf

    ao2mo = importlib.import_module("gradscf.ao2mo")
    assert "ao2mo" in gradscf.__all__
    assert "ao2mo" in dir(gradscf)
    assert gradscf.ao2mo is ao2mo
    assert ao2mo.__spec__.submodule_search_locations is not None


def test_ao2mo_public_exports_have_focused_owners():
    ao2mo = importlib.import_module("gradscf.ao2mo")
    exports = {
        "blocks": ("transform_eri_block",),
        "reference": (
            "validate_integrals", "transform_integrals",
            "validate_unrestricted_integrals", "transform_unrestricted_integrals",
        ),
        "spin": (
            "frozen_indices", "unrestricted_frozen_indices", "spin_orbital_integrals",
        ),
        "response": (
            "df_factors_to_mo_eri_slices", "eri_pair_matrix_to_mo_eri_slices",
        ),
    }
    for owner, names in exports.items():
        module = importlib.import_module(f"gradscf.ao2mo.{owner}")
        for name in names:
            assert name in ao2mo.__all__
            assert getattr(ao2mo, name) is getattr(module, name)


def test_retired_ao2mo_locations_are_removed():
    package_root = Path(__file__).resolve().parents[2] / "src" / "gradscf"
    assert not (package_root / "integrals" / "molecular" / "ao2mo.py").exists()
    assert not (package_root / "integrals" / "backends" / "native" / "ao2mo.py").exists()
    native = importlib.import_module("gradscf.ao2mo.native")
    assert native.transform_eri_block.__module__ == "gradscf.ao2mo.native"
