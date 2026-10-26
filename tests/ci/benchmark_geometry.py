"""Explicit large-basis check; run this file with pytest when resources permit."""
from pathlib import Path

from test_geometry import check_nitrogen_cisd_geometry


def test_nitrogen_def2_tzvp_cisd_geometry():
    check_nitrogen_cisd_geometry(basis="def2-tzvp", conv_tol=1e-10,
        max_cycle=150, max_space=40,
        report_path=Path("artifacts/ci_mp_validation/ci_large_geometry/result.json"))
