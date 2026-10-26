"""Native implementation boundaries after integral package reorganization."""
import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2] / "src/gradscf/integrals"


@pytest.mark.parametrize("module", [
    "packing", "ffi", "eri", "density_fitting", "jk", "autodiff/common",
    "autodiff/geometry", "autodiff/coefficients", "autodiff/exponents", "autodiff/evaluation",
])
def test_native_roles_have_dedicated_modules(module):
    assert (ROOT / "backends/native" / f"{module}.py").is_file()


def test_coefficient_products_do_not_own_combined_evaluation():
    source = ROOT / "backends/native/autodiff/coefficients.py"
    functions = {node.name for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef)}
    assert functions == {"coefficient_products"}


@pytest.mark.parametrize("source", ["packed_eri.cc", "density_fitting.cc", "direct_jk.cc", "packed_jk.cc"])
def test_native_build_compiles_separate_coulomb_roles(source):
    assert (ROOT / "_native/csrc" / source).is_file()
    assert f"csrc/{source}" in (ROOT / "_native/CMakeLists.txt").read_text()
