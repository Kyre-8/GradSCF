from pathlib import Path
import ast
from importlib.util import resolve_name
import re


def _removed_adapter_imports(text, package):
    removed = {"gradscf.reference", "gradscf.reference_legacy", "gradscf.pyscf_adapter"}
    def forbidden(name):
        return any(name == old or name.startswith(old+".") for old in removed)
    names = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = resolve_name("."*node.level+base, package)
            names.append(base)
            names.extend(base+"."+alias.name for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Also catch a literal dynamic import of a removed module.
            names.append(node.value)
    return any(forbidden(name) for name in names)


def test_production_code_does_not_depend_on_removed_reference_adapters():
    root = Path("src/gradscf")
    offenders = []
    for path in root.rglob("*.py"):
        package = ".".join(("gradscf", *path.relative_to(root).parts[:-1]))
        if _removed_adapter_imports(path.read_text(), package):
            offenders.append(str(path))
    assert offenders == []


def test_adapter_boundary_resolves_relative_imports():
    assert not _removed_adapter_imports("from .reference import FCIReference", "gradscf.fci")
    assert not _removed_adapter_imports("from ..reference import Something", "gradscf.fci.internal")
    assert _removed_adapter_imports("from ..reference import Something", "gradscf.fci")
    assert _removed_adapter_imports("from gradscf import reference", "gradscf.fci")
    assert _removed_adapter_imports("import gradscf.reference_legacy", "gradscf.fci")
    assert _removed_adapter_imports("importlib.import_module('gradscf.pyscf_adapter')", "gradscf.fci")


def test_reference_legacy_and_pyscf_adapter_are_not_runtime_modules():
    import importlib
    import gradscf

    assert "pyscf_adapter" not in gradscf.__all__
    assert "reference_legacy" not in gradscf.__all__
    for module_name in (
        "gradscf.pyscf_adapter",
        "gradscf.reference_legacy",
        "gradscf.reference",
    ):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{module_name} should not exist in the runtime package")


def test_runtime_public_api_does_not_expose_pyscf_bridge_symbols():
    import gradscf
    from gradscf import dft
    from gradscf.tools import upstreams

    forbidden = {
        "PySCFRSHSpec",
        "make_pyscf_rsh_spec",
        "ground_state_from_pyscf_mean_field",
    }
    for module in (gradscf, dft, upstreams):
        names = set(getattr(module, "__all__", ())) | set(vars(module))
        assert forbidden.isdisjoint(names)


def test_core_public_api_does_not_expose_removed_geomopt_namespace():
    import importlib
    import gradscf

    forbidden = {
        "BOHR_TO_ANGSTROM",
        "CoordinateUnit",
        "EnergyFunction",
        "GeometryOptimizationConfig",
        "GeometryOptimizationResult",
        "compute_forces",
        "make_rks_ground_state_energy_fn",
        "run_geometry_optimization",
        "run_rks_ground_state_geometry_optimization",
    }

    assert "geomopt" not in gradscf.__all__
    assert forbidden.isdisjoint(set(gradscf.__all__))
    for name in forbidden:
        assert not hasattr(gradscf, name)

    removed_module = ".".join(("gradscf", "geomopt"))
    try:
        importlib.import_module(removed_module)
    except ModuleNotFoundError:
        return
    raise AssertionError(f"{removed_module} should not exist in the runtime package")


def test_pyscf_runtime_imports_are_limited_to_integral_modules():
    root = Path("src/gradscf")
    allowed = {
        Path("src/gradscf/data/reference.py"),
        Path("src/gradscf/df/jk.py"),
        Path("src/gradscf/scf/init_guess.py"),
    }
    allowed_prefixes = (
        Path("src/gradscf/integrals"),
        Path("src/gradscf/integrals/basis/data/pyscf_basis_snapshot"),
    )
    pattern = re.compile(r"^\s*(from\s+pyscf\b|import\s+pyscf\b)", re.MULTILINE)

    offenders = []
    for path in root.rglob("*.py"):
        if path in allowed or any(path.is_relative_to(prefix) for prefix in allowed_prefixes):
            continue
        if pattern.search(path.read_text()):
            offenders.append(str(path))

    assert offenders == []


def test_legacy_mean_field_tddft_calls_are_not_in_runtime_code():
    root = Path("src/gradscf")
    forbidden = ("mf.TDDFT", "mf.TDA", "grad_dft.interface.pyscf")

    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        if any(pattern in text for pattern in forbidden):
            offenders.append(str(path))

    assert offenders == []


def test_physical_hf_pt2_helpers_are_owned_by_dft_not_a_neural_model():
    import reference_scf_features as scf_features
    from gradscf.model.neural_xc import inputs
    from gradscf.dft import hfx, pt2
    for name, owner in (
        ('_local_hfx_features_from_basis_dm', hfx),
        ('_local_pt2_feature_from_restricted_orbitals', pt2),
    ):
        assert not hasattr(scf_features, name)
        assert not hasattr(inputs, name)
        assert hasattr(owner, name)


def test_restricted_response_hvp_uses_factorized_transition_features():
    text = Path("src/gradscf/tddft/response.py").read_text()

    assert "_transition_densities_on_grid" not in text
    assert "def _restricted_grid_xc_response" in text
    assert "def _restricted_response_factors" in text
    assert "def _project_restricted_transition_to_grid" in text
    assert "def _project_grid_response_to_restricted_transition" in text
    hvp_body = text.split("def _restricted_grid_xc_response_hvp", maxsplit=1)[1].split(
        "def _as_nonlocal_response_diagonal",
        maxsplit=1,
    )[0]
    assert "_restricted_response_features(" not in hvp_body
    assert "_restricted_response_factors(" in hvp_body


def test_restricted_response_feature_kind_helpers_are_owned_by_features_module():
    import jax.numpy as jnp
    from gradscf.tools import features as features_module

    text = Path("src/gradscf/tddft/response.py").read_text()

    assert "def _normalize_response_feature_kind" not in text
    assert "def _infer_response_feature_kind" not in text
    assert features_module.normalize_response_feature_kind("mgga_lapl") == "MGGA_LAPL"
    assert features_module.normalize_response_feature_kind(None, default="GGA") == "GGA"
    assert features_module.infer_response_feature_kind(jnp.ones((5, 5, 2))) == "MGGA"


def test_public_api_prefers_owned_namespaces_and_object_conversion():
    import gradscf
    from gradscf import scf
    from gradscf.model.neural_xc import workflows
    from gradscf.scf import builders
    assert {'QuadratureGrid','RestrictedMolecule','UnrestrictedMolecule','as_reference'}.issubset(scf.__all__)
    assert {'MoleculeRun','MoleculeSpecConfig','ExperimentPipeline'}.issubset(workflows.__all__)
    for name in ('_build_restricted_reference','_build_unrestricted_reference'):
        assert name not in scf.__all__
        assert callable(getattr(builders,name))
    for name in ('build_molecule','run_molecule_from_spec','put_molecule_on_device'):
        assert name not in gradscf.__all__
    assert callable(scf.RKS.to_reference)
