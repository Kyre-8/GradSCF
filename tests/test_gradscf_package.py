import ast
import importlib
import importlib.util
import json
from pathlib import Path
import tomllib

import numpy as np


def test_distribution_uses_gradscf_name_and_package_data():
    metadata = tomllib.loads(Path("pyproject.toml").read_text())
    assert metadata["project"]["name"] == "gradscf"
    assert "gradscf.integrals.basis.data" in metadata["tool"]["setuptools"]["package-data"]
    assert not Path("src/td_graddft").exists()
    assert not Path("src/td_graddft_tools").exists()


def test_gradscf_exposes_existing_scientific_namespaces():
    assert importlib.util.find_spec("gradscf") is not None
    package = importlib.import_module("gradscf")
    assert Path(package.__file__).resolve() == Path("src/gradscf/__init__.py").resolve()
    for name in ("gto", "scf", "dft", "tdscf", "model", "training"):
        assert getattr(package, name) is importlib.import_module(f"gradscf.{name}")
    for name in ("UHF", "ROHF", "GHF", "RKS", "UKS", "ROKS", "GKS"):
        cls = getattr(package.scf, name)
        assert cls.__module__.startswith("gradscf.")
    assert package.dft.GKS is package.scf.GKS
    assert package.scf.run_rhf_from_integrals.__module__.startswith("gradscf.")


def test_gradscf_loads_bundled_basis_resources():
    assert importlib.util.find_spec("gradscf") is not None
    from importlib.resources import files

    basis = files("gradscf.integrals.basis.data").joinpath("pyscf_basis_snapshot/sto-3g.dat")
    assert basis.is_file()
    assert "BASIS" in basis.read_text()


def test_active_python_imports_do_not_use_retired_package_names():
    old_names = {"td_graddft", "td_graddft_tools"}
    violations = []
    for root in ("src", "tests", "tools", "examples"):
        for path in Path(root).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = ([node.module] if isinstance(node, ast.ImportFrom) and node.level == 0
                         else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                if any(name and name.split(".")[0] in old_names for name in names):
                    violations.append(str(path))
    assert violations == []


def test_gradscf_tools_reads_existing_target_bundle(tmp_path):
    assert importlib.util.find_spec("gradscf_tools") is not None
    from gradscf_tools import load_molecular_target_bundle

    metadata = {"input_info": {"system_label": "legacy_h2", "charge": 0, "spin": 0},
                "scalar_fields": {"weight": 2.0}, "present_array_fields": ["target_e0_total_h"]}
    path = tmp_path / "legacy_targets.npz"
    np.savez(path, target_e0_total_h=np.array(-1.1),
             __td_graddft_target_bundle_metadata_json__=np.asarray(json.dumps(metadata)))
    bundle = load_molecular_target_bundle(path)
    assert bundle.input_info.system_label == "legacy_h2"
    assert bundle.weight == 2.0
    np.testing.assert_allclose(bundle.target_e0_total_h, -1.1)
