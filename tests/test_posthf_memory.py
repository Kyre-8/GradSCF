"""Post-HF callers propagate decimal-MB budgets to existing AO->MO blocks.

These checks concern eager budget inheritance and individual transforms. They
do not establish a bound on solver histories, DF-MP Qia storage, spin expansion,
other resident SCF states, or the complete differentiated process.
"""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc, ci, gto, mp, scf
from gradscf._memory import DEFAULT_MAX_MEMORY_MB
from gradscf import ao2mo
from gradscf.ao2mo import reference as ao2mo_reference
from gradscf.ao2mo import blocks as ao2mo_blocks
from gradscf.scf.reference import RestrictedReference, UnrestrictedReference


def _hf_source(unrestricted=False, *, max_memory=128.0):
    molecule = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", cart=True)
    return (scf.UHF if unrestricted else scf.RHF)(
        molecule, integral_backend="native", execution_device="cpu",
        max_memory=max_memory, conv_tol=1e-12, conv_tol_grad=1e-10,
        conv_tol_density=1e-11,
    )


def _method(name, source, **kwargs):
    if name == "mp3":
        return mp.MP3(source, **kwargs)
    if name == "series":
        return mp.MP(source, order=2, algorithm="series", **kwargs)
    return {"mp2": mp.MP2, "ccsd": cc.CCSD,
            "cisd": ci.CISD, "cis": ci.CIS}[name](source, **kwargs)


@pytest.fixture(scope="module", params=(False, True), ids=("rhf", "uhf"))
def solved_hf(request):
    source = _hf_source(request.param).run()
    assert source.converged
    return source


@pytest.mark.parametrize("name", ["mp2", "ccsd", "cisd", "cis"])
@pytest.mark.parametrize("unrestricted", [False, True], ids=["rhf", "uhf"])
@pytest.mark.parametrize("kwargs", [{}, {"max_memory": None}, {"max_memory": 31.25}],
                         ids=["inherit", "none-inherits", "override"])
def test_facade_inherits_scf_budget_or_accepts_override(name, unrestricted, kwargs):
    source = _hf_source(unrestricted, max_memory=173.5)
    calculation = _method(name, source, **kwargs)
    expected = 31.25 if kwargs.get("max_memory") is not None else 173.5
    assert calculation.max_memory == expected
    assert source.max_memory == 173.5


@pytest.mark.parametrize("name", ["mp2", "ccsd", "cisd", "cis"])
@pytest.mark.parametrize("unrestricted", [False, True], ids=["restricted", "unrestricted"])
def test_explicit_mo_reference_uses_shared_default_budget(name, unrestricted):
    h = np.diag([-1., .5])
    g = np.zeros((2,) * 4)
    reference = (UnrestrictedReference((h, h), (g, g, g), (1, 1))
                 if unrestricted else RestrictedReference(h, g, 1))
    calculation = _method(name, reference)
    assert calculation.max_memory == DEFAULT_MAX_MEMORY_MB


@pytest.mark.parametrize("name", ["mp2", "ccsd", "cisd", "cis"])
@pytest.mark.parametrize("budget", [0., -1., np.inf, np.nan, True, "1024"])
def test_facade_rejects_invalid_budget_without_preparing_integrals(name, budget, monkeypatch):
    source = _hf_source()

    def forbidden(*args, **kwargs):
        pytest.fail("An invalid facade budget reached AO->MO allocation")

    monkeypatch.setattr(ao2mo_reference, "transform_eri_block", forbidden)
    monkeypatch.setattr(importlib.import_module("gradscf.mp.integrals"),
                        "transform_eri_blocks", forbidden)
    with pytest.raises(ValueError, match="max_memory"):
        _method(name, source, max_memory=budget)


@pytest.mark.parametrize("unrestricted", [False, True], ids=["restricted", "unrestricted"])
@pytest.mark.parametrize("memory_kwargs", [{}, {"max_memory": 24.5}],
                         ids=["shared-default", "override"])
def test_full_transform_wrappers_forward_budget_to_each_block(
        unrestricted, memory_kwargs, monkeypatch):
    h, c = jnp.eye(3), jnp.eye(3)[:, :2]
    seen = []

    def block(coefficients, *, max_memory=DEFAULT_MAX_MEMORY_MB, **kwargs):
        seen.append(max_memory)
        return jnp.zeros(tuple(x.shape[1] for x in coefficients), dtype=c.dtype)

    monkeypatch.setattr(ao2mo_reference, "transform_eri_block", block)
    monkeypatch.setattr(ao2mo_blocks, "transform_eri_block", block)
    if unrestricted:
        ao2mo.transform_unrestricted_integrals(
            h, (c, c), eri=jnp.zeros((3,) * 4), **memory_kwargs)
    else:
        ao2mo.transform_integrals(h, c, eri=jnp.zeros((3,) * 4), **memory_kwargs)
    expected = memory_kwargs.get("max_memory", DEFAULT_MAX_MEMORY_MB)
    assert seen == [expected] * (3 if unrestricted else 1)


@pytest.mark.parametrize("representation", ["dense", "packed", "df"])
@pytest.mark.parametrize("unrestricted", [False, True], ids=["restricted", "unrestricted"])
def test_budgeted_full_transform_matches_small_dense_oracle(representation, unrestricted):
    rng = np.random.default_rng(921)
    factors = rng.normal(scale=.2, size=(17, 3, 3))
    factors += factors.transpose(0, 2, 1)
    eri = np.einsum("Lpq,Lrs->pqrs", factors, factors)
    coefficients = tuple(rng.normal(scale=.3, size=(3, 2)) for _ in range(2))
    h = np.diag([-1., -.4, .6])
    rows, columns = np.tril_indices(3)
    inputs = {"dense": {"eri": jnp.asarray(eri)},
              "packed": {"eri_pair_matrix": jnp.asarray(eri[rows, columns][:, rows, columns])},
              "df": {"df_factors": jnp.asarray(factors)}}[representation]
    cs = tuple(jnp.asarray(c, dtype=jnp.float64) for c in coefficients)
    # The 17-row DF axis must be tiled by the shared block planner.
    if unrestricted:
        budget = .005
        if representation == "packed":
            # Joint inputs/output buffers/scratch need 4060 working bytes:
            # 0.005 MB leaves 4000; 0.0051 MB leaves 4080 after the reserve.
            with pytest.raises(ValueError, match="memory|budget|resident"):
                ao2mo.transform_unrestricted_integrals(h, cs, max_memory=budget, **inputs)
            budget = .0051
        hs, gs = ao2mo.transform_unrestricted_integrals(h, cs, max_memory=budget, **inputs)
        pairs = ((0, 0), (0, 1), (1, 1))
    else:
        transformed_h, transformed_g = ao2mo.transform_integrals(
            h, cs[0], max_memory=.005, **inputs)
        hs, gs, pairs = (transformed_h,), (transformed_g,), ((0, 0),)
    for actual, c in zip(hs, coefficients):
        np.testing.assert_allclose(actual, c.T @ h @ c, atol=2e-12, rtol=2e-12)
    for actual, (a, b) in zip(gs, pairs):
        expected = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri,
                             coefficients[a], coefficients[a], coefficients[b], coefficients[b])
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


class _BudgetReached(Exception):
    """Stop at a real caller boundary before running its downstream solver."""


@pytest.mark.parametrize("override", [None, 23.5], ids=["inherit", "override"])
def test_scf_mo_adapter_forwards_source_or_override_budget(solved_hf, override, monkeypatch):
    reference = importlib.import_module("gradscf.scf.reference")
    unrestricted = isinstance(solved_hf, scf.UHF)
    transform = "transform_unrestricted_integrals" if unrestricted else "transform_integrals"
    adapter = (reference.unrestricted_reference_from_source if unrestricted
               else reference.reference_from_source)
    seen = []

    def capture(*args, **kwargs):
        seen.append(kwargs.get("max_memory"))
        raise _BudgetReached

    monkeypatch.setattr(reference, transform, capture)
    with pytest.raises(_BudgetReached):
        adapter(solved_hf, max_memory=override)
    assert seen == [solved_hf.max_memory if override is None else override]
    assert solved_hf.max_memory == 128.0


def test_ci_mo_adapter_forwards_override_to_shared_reference(monkeypatch):
    integrals = importlib.import_module("gradscf.ci.integrals")
    source = _hf_source()
    seen = []

    def capture(received_source, **kwargs):
        assert received_source is source
        seen.append(kwargs.get("max_memory"))
        raise _BudgetReached

    monkeypatch.setattr(integrals, "prepare_reference", capture)
    with pytest.raises(_BudgetReached):
        integrals.reference_from_source(source, max_memory=23.5)
    assert seen == [23.5]


@pytest.mark.parametrize("override", [None, 23.5], ids=["inherit", "override"])
def test_mp2_adapter_forwards_budget_to_selected_spin_blocks(solved_hf, override, monkeypatch):
    integrals = importlib.import_module("gradscf.mp.integrals")
    original = integrals.transform_eri_blocks
    seen = []

    def capture(coefficients, requests, **kwargs):
        seen.extend([kwargs.get("max_memory")] * len(requests))
        return original(coefficients, requests, **kwargs)

    monkeypatch.setattr(integrals, "transform_eri_blocks", capture)
    result = integrals.from_scf(solved_hf, max_memory=override)
    count = 3 if isinstance(solved_hf, scf.UHF) else 1
    assert seen == [solved_hf.max_memory if override is None else override] * count
    assert bool(result.finite)
    assert solved_hf.max_memory == 128.0


@pytest.mark.parametrize("name", ["mp2", "mp3", "series", "ccsd", "cisd", "cis"])
def test_facade_kernel_forwards_current_method_budget(name, solved_hf, monkeypatch):
    unrestricted = isinstance(solved_hf, scf.UHF)
    family = "mp" if name in {"mp2", "mp3", "series"} else "cc" if name == "ccsd" else "ci"
    api = importlib.import_module(f"gradscf.{family}.api")
    boundary = ("unrestricted_reference_from_source" if unrestricted else "reference_from_source")
    if name == "mp2" or (name == "mp3" and not unrestricted):
        boundary = "prepare_scf_source"
    elif name == "ccsd" and not unrestricted:
        boundary = "from_orbitals"
    elif name == "cis" and not unrestricted:
        boundary = "solve_cis_response"
    calculation = _method(name, solved_hf)
    calculation.max_memory = 29.75
    seen = []

    def capture(*args, **kwargs):
        seen.append(kwargs.get("max_memory"))
        raise _BudgetReached

    monkeypatch.setattr(api, boundary, capture)
    with pytest.raises(_BudgetReached):
        calculation.kernel()
    assert seen == [29.75]
    assert solved_hf.max_memory == 128.0


@pytest.mark.parametrize("adapter_name", ["scf", "mp"])
def test_adapter_revalidates_mutated_source_budget_before_transform(
        adapter_name, solved_hf, monkeypatch):
    unrestricted = isinstance(solved_hf, scf.UHF)
    if adapter_name == "scf":
        owner = importlib.import_module("gradscf.scf.reference")
        adapter = (owner.unrestricted_reference_from_source if unrestricted
                   else owner.reference_from_source)
        boundary = "transform_unrestricted_integrals" if unrestricted else "transform_integrals"
    else:
        owner = importlib.import_module("gradscf.mp.integrals")
        adapter, boundary = owner.from_scf, "transform_eri_blocks"

    def forbidden(*args, **kwargs):
        pytest.fail("A mutated invalid source budget reached AO->MO allocation")

    monkeypatch.setattr(owner, boundary, forbidden)
    monkeypatch.setattr(solved_hf, "max_memory", np.nan)
    with pytest.raises(ValueError, match="max_memory"):
        adapter(solved_hf)
