import tomllib
from pathlib import Path


def test_pyscf_basis_snapshot_is_included_as_package_data():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    data_patterns = package_data["gradscf.integrals.basis.data"]

    assert "pyscf_basis_snapshot/**/*" in data_patterns


def test_integral_cuda_sources_are_not_included_as_package_data():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]

    assert "gradscf.integrals.backends.jax_reference" not in package_data
