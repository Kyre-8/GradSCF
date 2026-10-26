import importlib


def test_gradscf_dir_lists_recommended_namespaces():
    import gradscf

    for name in ("gto", "scf", "dft", "tdscf", "model", "training"):
        assert name in gradscf.__all__
        assert name in dir(gradscf)
        assert getattr(gradscf, name) is importlib.import_module(f"gradscf.{name}")

    # neural model namespaces moved to gradscf.model (hard switch)
    from gradscf import model

    for name in ("neural_xc", "neural_d", "nnao"):
        assert getattr(model, name) is importlib.import_module(f"gradscf.model.{name}")


def test_gradscf_exposes_pyscf_style_namespaces():
    for name in (
        "gradscf.gto",
        "gradscf.dft",
        "gradscf.scf",
        "gradscf.tdscf",
    ):
        module = importlib.import_module(name)
        assert module is not None


def test_gradscf_reference_module_is_removed():
    try:
        importlib.import_module("gradscf.reference")
    except ModuleNotFoundError:
        return
    raise AssertionError("gradscf.reference should not import successfully")


def test_dft_namespace_exposes_ks_facades():
    from gradscf import dft, scf

    assert dft.RKS is scf.RKS
    assert dft.UKS is scf.UKS


def test_generic_and_model_functionals_have_distinct_owners():
    import gradscf

    assert gradscf.dft.Functional is not gradscf.model.neural_xc.Functional
    assert gradscf.dft.Functional.__module__ == 'gradscf.dft.functional'
    assert not hasattr(gradscf, 'Functional')
    assert not hasattr(gradscf.model, 'training')


def test_top_level_removes_legacy_neural_xc_exports():
    import gradscf

    removed = (
        "Density" "NeuralXCFunctional",
        "Neural" "XCFunctional",
        "Pointwise" "MLP",
        "make_neural" "_lda_functional",
        "make_dm21" "_like_functional",
    )

    for name in removed:
        assert not hasattr(gradscf, name), f"{name} should not be exported at top level"


def test_gradscf_pyscf_style_submodules_import():
    for name in (
        "gradscf.gto.basis",
        "gradscf.gto.grid",
        "gradscf.scf.rks",
        "gradscf.scf.uks",
        "gradscf.dft.xc",
    ):
        module = importlib.import_module(name)
        assert module is not None
