"""Native RHF/UHF acceptance above the 32-angle preconditioner threshold.

Water/6-31G has 40 RHF angles; NH/6-31G (spin=2) has 54 UHF angles.
Reuse the full-ERI oracle, direct no-ERI guard, float64, and tolerances from
test_scf_orbital_direct; all response coordinates and steps are in bohr.
"""
import pytest

from gradscf.scf import orbital_optimization
import test_scf_orbital_direct as old


@pytest.fixture(scope="module", params=["water_631g", "nh_631g"], ids=["rhf", "uhf"])
def large_native_problem(request):
    problem = old.native_problem.__wrapped__(request, max_memory=128.)
    nao, dimension = (13, 40) if request.param == "water_631g" else (11, 54)
    assert problem[2].topology.nao == nao
    return problem, dimension


@pytest.fixture
def preconditioned_problem(large_native_problem, monkeypatch):
    problem, dimension = large_native_problem
    original = orbital_optimization.solve_orbitals
    seen = set()

    def solve(args, coeff, **kwargs):
        # Observe the real public minimizer at Python trace time; do not call
        # the callback or add any Hessian-vector products to the calculation.
        assert kwargs["dimension"] == dimension > 32
        assert callable(kwargs.get("hessian_diagonal"))
        assert kwargs["max_memory"] == 128.
        seen.add(args["direct_basis"] is not None)
        return original(args, coeff, **kwargs)

    monkeypatch.setattr(orbital_optimization, "solve_orbitals", solve)
    yield problem
    assert seen == {False, True}, "Both full-ERI and direct solves must be observed"


def test_native_preconditioned_state(preconditioned_problem, monkeypatch):
    old.test_direct_stationary_hf_matches_full_state(preconditioned_problem, monkeypatch)


def test_native_preconditioned_density_response(preconditioned_problem, monkeypatch):
    old.test_direct_density_observable_retains_implicit_response(preconditioned_problem, monkeypatch)


def test_native_preconditioned_geometry_gradient_hvp_fd(preconditioned_problem, monkeypatch):
    old.test_direct_geometry_gradient_and_hvp_match_full(preconditioned_problem, monkeypatch)
