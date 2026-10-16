"""Checkpoint identity and completion gates for the P03 q-mesh workflow."""
import importlib.util
from pathlib import Path

import pytest


def workflow():
    path = Path(__file__).resolve().parents[2]/"examples/pbc/lih_qmesh_phonons.py"
    assert path.exists(), "The resumable P03 q-mesh workflow is not implemented"
    spec = importlib.util.spec_from_file_location("p03_qmesh_workflow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_changed_source_or_q_cannot_resume(tmp_path):
    module = workflow()
    path = tmp_path/"q.json"
    identity = {"source": "frozen", "q": [0., 0., 1/3]}
    record = module.open_checkpoint(path, identity)
    assert record["complete"] is False
    module.save_checkpoint(path, record)
    for changed in ({"source": "other", "q": identity["q"]},
                    {"source": "frozen", "q": [0., 0., -1/3]}):
        with pytest.raises(ValueError, match="identity"):
            module.open_checkpoint(path, changed)


def test_partial_column_survives_without_claiming_completion(tmp_path):
    module = workflow()
    path = tmp_path/"q.json"
    identity = {"source": "frozen", "q": [0., 0., 1/3]}
    record = module.open_checkpoint(path, identity)
    record["columns"] = [{"index": 0, "real": [1., 2.], "imag": [.3, -.2],
                           "residual": 1e-12, "converged": True}]
    module.save_checkpoint(path, record)
    resumed = module.open_checkpoint(path, identity)
    assert not resumed["complete"] and not resumed["passed"]
    assert resumed["columns"] == record["columns"]
    assert not path.with_suffix(".tmp").exists()
