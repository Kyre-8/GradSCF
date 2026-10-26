"""Legacy reporting helpers retain their distinct behavior only in their owner."""
import gradscf
from gradscf.tools import api


def test_legacy_reporting_helpers_are_not_root_electronic_structure_apis():
    for name in ('MoleculeConfig','build_molecule','run_pipeline','run_spectrum_pipeline'):
        assert not hasattr(gradscf,name)
        assert hasattr(api,name)
