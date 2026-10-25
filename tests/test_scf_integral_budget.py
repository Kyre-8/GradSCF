"""SCF planning charges direct/DF integral work before allocating integrals."""
from types import SimpleNamespace

import numpy as np
import pytest

from gradscf import gto
from gradscf.scf.inputs import assembly, grid
from gradscf.scf.rks import RKSConfig
from gradscf.scf.uks import UKSConfig
from gradscf.integrals.backends.native import jk


def _forbidden(*args, **kwargs):
    raise AssertionError("Integral evaluation started before the memory check.")


def _spec():
    return gto.M(atom="H 0 0 0; H 0 0 .8", basis="sto-3g", cart=True).to_spec()


def test_native_workspace_estimate_is_shared_with_response():
    from gradscf.tddft import response_options

    estimate = getattr(jk, "native_jk_workspace_bytes", None)
    assert callable(estimate), "The integral provider must own its native workspace estimate."
    assert response_options._native_direct_workspace_bytes is estimate
    topology = SimpleNamespace(angular_momenta=(0, 1), primitive_counts=(3, 2),
        contraction_counts=(1, 1), nuclear_charges=(1, 1), cart=True)
    provider = SimpleNamespace(plan=SimpleNamespace(topology=topology))
    assert estimate(provider, 4, 0.) > 0
    assert estimate(provider, 4, 1e-12) > estimate(provider, 4, 0.)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_direct_native_workspace_rejected_before_any_integrals(monkeypatch, unrestricted):
    monkeypatch.setattr(grid, "_GRID_AO_INPUT_CACHE", {})
    monkeypatch.setattr(jk, "native_jk_workspace_bytes", lambda *args: 25_000_000,
                        raising=False)
    monkeypatch.setattr(assembly, "make_plan", lambda top, **kwargs:
        SimpleNamespace(topology=top, backend="native", evaluate=_forbidden))
    config_type = UKSConfig if unrestricted else RKSConfig
    builder = assembly.build_uks_integral_inputs if unrestricted else assembly.build_rks_integral_inputs
    cfg = config_type(xc_spec="pbe", jk_backend="direct", max_memory=10.,
                      grid_block_size=1, direct_scf_workers=1)
    with pytest.raises(ValueError, match="budget|max_memory|workspace"):
        builder(atom=_spec(), basis="sto-3g", config=cfg, integral_backend="native")


def test_direct_workspace_is_reserved_in_grid_plan(monkeypatch):
    monkeypatch.setattr(jk, "native_jk_workspace_bytes", lambda *args: 25_000_000,
                        raising=False)
    monkeypatch.setattr(assembly, "make_plan", lambda top, **kwargs:
        SimpleNamespace(topology=top, backend="native", evaluate=_forbidden))

    class ReachedPlanner(Exception):
        pass

    def inspect_plan(**kwargs):
        assert kwargs.get("fixed_workspace_bytes") == 75_000_000
        raise ReachedPlanner

    monkeypatch.setattr(assembly, "_prepare_basis_grid_context", inspect_plan)
    with pytest.raises(ReachedPlanner):
        assembly.build_rks_integral_inputs(atom=_spec(), basis="sto-3g",
            config=RKSConfig(jk_backend="direct", max_memory=1000.), integral_backend="native")


def test_auxiliary_df_rejected_before_metric_or_three_center_allocation(monkeypatch):
    from gradscf.integrals.molecular import density_fitting

    original_prepare = assembly.prepare_basis
    prepared_auxiliary = []

    def prepare(spec, basis, **kwargs):
        if basis == "oversized-test-aux":
            prepared_auxiliary.append(basis)
            return SimpleNamespace(nao=4000), object()
        return original_prepare(spec, basis, **kwargs)

    monkeypatch.setattr(grid, "_GRID_AO_INPUT_CACHE", {})
    monkeypatch.setattr(assembly, "prepare_basis", prepare)
    monkeypatch.setattr(assembly, "make_plan", lambda top, **kwargs:
        SimpleNamespace(topology=top, backend="native", evaluate=_forbidden))
    monkeypatch.setattr(density_fitting, "make_auxiliary_plan", _forbidden)
    cfg = RKSConfig(xc_spec="pbe", jk_backend="df", auxbasis="oversized-test-aux",
                    max_memory=10., grid_block_size=1)
    with pytest.raises(ValueError, match="budget|max_memory|workspace|DF|auxiliary"):
        assembly.build_rks_integral_inputs(atom=_spec(), basis="sto-3g",
            config=cfg, integral_backend="native")
    assert prepared_auxiliary == ["oversized-test-aux"]


def test_small_auxiliary_metadata_is_prepared_once_and_tolerance_preserved(monkeypatch):
    from gradscf.integrals.molecular import density_fitting

    original_prepare = assembly.prepare_basis
    auxiliary_calls, tolerances = [], []

    def prepare(spec, basis, **kwargs):
        if basis == "small-test-aux":
            auxiliary_calls.append(basis)
            basis = "sto-3g"
        return original_prepare(spec, basis, **kwargs)

    def factors(parameters, auxiliary_parameters, *, lindep):
        tolerances.append(lindep)
        return np.ones((2, 3))

    monkeypatch.setattr(grid, "_GRID_AO_INPUT_CACHE", {})
    monkeypatch.setattr(assembly, "prepare_basis", prepare)
    monkeypatch.setattr(assembly, "make_plan", lambda top, **kwargs:
        SimpleNamespace(topology=top, backend="native", evaluate=lambda *args, **kw: np.eye(2)))
    monkeypatch.setattr(density_fitting, "make_auxiliary_plan", lambda *args:
        SimpleNamespace(factors=factors))
    cfg = RKSConfig(xc_spec="pbe", jk_backend="df", auxbasis="small-test-aux",
                    df_tol=3e-11, max_memory=64., grid_block_size=1)
    inputs = assembly.build_rks_integral_inputs(atom=_spec(), basis="sto-3g",
        config=cfg, integral_backend="native", include_dipole_integrals=False)
    assert inputs.df_factors.shape == (2, 2, 2)
    assert auxiliary_calls == ["small-test-aux"]
    assert tolerances == [3e-11]


def test_pair_factorization_accounts_for_full_rank_output(monkeypatch):
    # A 100-AO packed pair matrix fits 500 MB, but it and the default
    # full-rank (5050,100,100) factor output do not. No large arrays are made.
    basis = SimpleNamespace(nao=100)
    context = SimpleNamespace(basis=basis, coords=np.zeros((1, 3)),
        grid_weights=np.ones(1), grid_ao=None, grid_ao_bundle=None,
        geometry_is_traced=False)
    monkeypatch.setattr(assembly, "_prepare_basis_grid_context", lambda **kwargs: context)
    monkeypatch.setattr(grid, "_grid_resident_bytes", lambda *args, **kwargs: 0)
    monkeypatch.setattr(grid, "_grid_device", lambda *args: None)
    monkeypatch.setattr(assembly, "prepare_basis", lambda *args, **kwargs: (basis, object()))
    monkeypatch.setattr(assembly, "make_plan", lambda top, **kwargs:
        SimpleNamespace(topology=top, backend="native", evaluate=_forbidden))
    cfg = RKSConfig(xc_spec="pbe", jk_backend="df", max_memory=500.)
    with pytest.raises(ValueError, match="budget|max_memory|workspace|DF|factors"):
        assembly.build_rks_integral_inputs(atom=_spec(), basis="sto-3g",
            config=cfg, integral_backend="native")
