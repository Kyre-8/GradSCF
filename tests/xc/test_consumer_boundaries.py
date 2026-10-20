"""New recipe names must not fall through old full-range-only consumers."""
from types import SimpleNamespace

import pytest


def test_noncollinear_solver_does_not_drop_screened_exchange():
    from gradscf.scf.gks import GKSConfig, _validate_mode

    with pytest.raises(NotImplementedError, match='range|screened'):
        _validate_mode(GKSConfig(xc_spec='hse06'))


@pytest.mark.parametrize('assembly', ['displacement', 'curvature'])
def test_finite_q_geometry_does_not_use_full_range_exchange_for_rsh(assembly):
    source = SimpleNamespace(xc='hse06', inputs=None, exxdiv=None)
    with pytest.raises(NotImplementedError, match='range|screened'):
        if assembly == 'displacement':
            from gradscf.dfpt.displacement import _displacement_source
            _displacement_source(source, None, None)
        else:
            from gradscf.dfpt.force_constants import geometry_curvature
            geometry_curvature(source, None, None, None)
