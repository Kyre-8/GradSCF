"""The declared optional XC dependency must include the verified runtime."""
from importlib import metadata
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
import pytest


def test_installed_reference_backend_satisfies_declared_xc_extras():
    pytest.importorskip('jax_xc')
    project = tomllib.loads((Path(__file__).resolve().parents[2]/'pyproject.toml').read_text())
    installed = metadata.version('jax-xc')
    for extra in ('upstreams', 'reproducibility'):
        requirement = next(Requirement(spec) for spec in project['project']['optional-dependencies'][extra]
                           if Requirement(spec).name == 'jax-xc')
        assert installed in requirement.specifier, (extra, installed, str(requirement))
