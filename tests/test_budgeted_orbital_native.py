"""Reuse the existing R/U native state, density and geometry acceptance gates."""
import pytest

from gradscf import scf
import test_scf_orbital_direct as old


@pytest.fixture(scope='module', params=['rhf', 'uhf'])
def budgeted_problem(request):
    name = 'minimize_uks_from_integrals' if request.param == 'uhf' else 'minimize_roks_from_integrals'
    original = getattr(scf, name)
    def minimize(**kwargs):
        if kwargs.get('direct_basis') is not None:
            kwargs.update(max_memory=128., reserved_bytes=4096)
        return original(**kwargs)
    # The existing fixture captures this callable once. Its full-ERI oracle
    # remains unbudgeted; molecule/basis/iterations/thresholds are unchanged.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scf, name, minimize)
        return old.native_problem.__wrapped__(request)


def test_existing_budgeted_native_state(budgeted_problem, monkeypatch):
    old.test_direct_stationary_hf_matches_full_state(budgeted_problem, monkeypatch)


def test_existing_budgeted_native_density_response(budgeted_problem, monkeypatch):
    old.test_direct_density_observable_retains_implicit_response(budgeted_problem, monkeypatch)


def test_existing_budgeted_native_geometry_gradient_hvp_fd(budgeted_problem, monkeypatch):
    old.test_direct_geometry_gradient_and_hvp_match_full(budgeted_problem, monkeypatch)
