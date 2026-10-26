import ast
from pathlib import Path


def test_workflow_implementation_entrypoint_is_not_flattened_at_root():
    import gradscf
    from gradscf.model.neural_xc.workflows.core import run_pipeline_core

    assert "run_pipeline_core" not in gradscf.__all__
    assert "run_pipeline_core" not in gradscf.model.neural_xc.workflows.__all__
    assert gradscf.model.neural_xc.workflows.core.run_pipeline_core is run_pipeline_core


def test_workflow_reporting_defers_pyplot_import_until_plotting():
    source = Path("src/gradscf/model/neural_xc/workflows/reporting.py").read_text()
    tree = ast.parse(source)
    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    assert all(
        not any(alias.name == "matplotlib.pyplot" for alias in getattr(node, "names", ()))
        and getattr(node, "module", None) != "matplotlib.pyplot"
        for node in top_level_imports
    )


def test_workflow_core_uses_public_facades_for_xc_and_tdscf():
    text = Path("src/gradscf/model/neural_xc/workflows/core.py").read_text()

    assert "make_neural_xc_functional" not in text
    assert "RestrictedCasidaTDDFT" not in text
    assert "UnrestrictedCasidaTDDFT" not in text
    assert "SemilocalResponseFunctional" not in text
    assert "neural_xc.Functional(" in text
    assert "tdscf.TDDFT(" in text
    assert "tdscf.TDA(" in text
