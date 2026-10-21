"""Exact native-direct MP2 selected blocks; CPU float64, energies in Hartree."""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module", params=("rhf", "closed-uhf", "open-uhf"))
def direct_reference(request):
    from gradscf import gto, scf

    unrestricted = request.param != "rhf"
    open_shell = request.param == "open-uhf"
    atom = ("H 0 0 0; H 0 0 .85; H 0 0 1.9" if open_shell
            else "Li 0 0 0; H 0 0 1.6")
    basis = "3-21g" if open_shell else "sto-3g"
    molecule = gto.M(atom=atom, basis=basis, spin=int(open_shell), cart=True)
    factory = scf.UHF if unrestricted else scf.RHF
    options = dict(integral_backend="native", execution_device="cpu",
                   max_memory=128., conv_tol=1e-13, conv_tol_grad=1e-10,
                   conv_tol_density=1e-11, max_cycle=200)
    full = factory(molecule, **options).run()
    direct = factory(molecule, jk_backend="direct", direct_scf_tol=0., **options).run()
    assert full.converged and direct.converged
    np.testing.assert_allclose(direct.e_tot, full.e_tot, atol=2e-10, rtol=0)
    inputs = direct.reference if unrestricted else direct._scf_inputs
    assert inputs.direct_basis is not None and inputs.df_factors is None
    ao_data = inputs.rep_tensor if unrestricted else inputs.eri
    assert ao_data is None or ao_data.size == 0
    if unrestricted:
        assert direct._scf_inputs is None
    else:
        assert inputs.eri_pair_matrix is None
    # Tiny full AO/MO arrays are independent test oracles only. Use the direct
    # orbitals so amplitude comparisons do not depend on eigenvector phases or
    # arbitrary rotations in the degenerate virtual space of LiH.
    provider = inputs.direct_basis
    ao_eri = provider.plan.evaluate("eri", provider.parameters)
    coefficients = tuple(direct.mo_coeff) if unrestricted else (direct.mo_coeff,)
    hcore = inputs.h1e if unrestricted else inputs.hcore
    hs = tuple(c.T @ hcore @ c for c in coefficients)
    pairs = ((0, 0), (0, 1), (1, 1)) if unrestricted else ((0, 0),)
    gs = tuple(jnp.einsum("pqrs,pi,qj,rk,sl->ijkl", ao_eri,
                         coefficients[a], coefficients[a], coefficients[b], coefficients[b],
                         precision="highest") for a, b in pairs)
    nocc = (inputs.nocc_alpha, inputs.nocc_beta) if unrestricted else inputs.nelectron // 2
    return dict(source=direct, full=full, h=hs if unrestricted else hs[0],
                g=gs if unrestricted else gs[0], nocc=nocc, atom=atom,
                basis=basis, spin=int(open_shell), unrestricted=unrestricted,
                frozen=([0], []) if open_shell else 1, provider=provider,
                nuclear_repulsion=inputs.nuclear_repulsion)


def _forbid_reference_materialization(source, provider, monkeypatch):
    from gradscf.ao2mo import reference
    from gradscf.scf.inputs.types import RKSIntegralInputs

    def forbidden(*args, **kwargs):
        pytest.fail("Direct MP2 must reuse solved Focks and transform only selected native blocks")

    monkeypatch.setattr(source, "to_reference", forbidden)
    monkeypatch.setattr(source, "_ensure_reference", forbidden)
    monkeypatch.setattr(source, "_build_reference", forbidden)
    monkeypatch.setattr(RKSIntegralInputs, "response_eri_pair_matrix", forbidden)
    monkeypatch.setattr(type(provider), "response_eri_pair_matrix", forbidden)
    monkeypatch.setattr(type(provider), "get_jk", forbidden)
    monkeypatch.setattr(type(provider.plan), "evaluate", forbidden)
    monkeypatch.setattr(reference, "transform_integrals", forbidden)
    monkeypatch.setattr(reference, "transform_unrestricted_integrals", forbidden)
    monkeypatch.setattr(reference, "_transform_eri", forbidden)


@pytest.mark.parametrize("freeze", [False, True], ids=["all-active", "frozen"])
@pytest.mark.parametrize("with_t2", [False, True], ids=["energy-only", "amplitudes"])
def test_direct_mp2_matches_full_scf_and_selected_amplitudes(
        direct_reference, freeze, with_t2, monkeypatch):
    from gradscf import mp

    data = direct_reference
    source = data["source"]
    frozen = data["frozen"] if freeze else None
    expected = mp.run_mp(data["h"], data["g"], nocc=data["nocc"], frozen=frozen,
                         nuclear_repulsion=data["nuclear_repulsion"],
                         config=mp.MPConfig(with_t2=with_t2))
    full = mp.MP2(data["full"], frozen=frozen, with_t2=with_t2).run()
    assert expected.valid and full.converged
    _forbid_reference_materialization(source, data["provider"], monkeypatch)
    integrals = importlib.import_module("gradscf.mp.integrals")
    transform = integrals.transform_eri_block
    seen = []

    def selected(coefficients, **kwargs):
        assert kwargs.get("direct_basis") is data["provider"]
        assert all(kwargs.get(key) is None for key in ("eri", "eri_pair_matrix", "df_factors"))
        seen.append((tuple(c.shape[1] for c in coefficients), kwargs.get("max_memory")))
        return transform(coefficients, **kwargs)

    monkeypatch.setattr(integrals, "transform_eri_block", selected)
    budget = 23.5 if with_t2 else None
    actual = mp.MP2(source, frozen=frozen, with_t2=with_t2, max_memory=budget).run()
    assert actual.converged
    ns = data["nocc"] if data["unrestricted"] else (data["nocc"],)
    frozen_sets = (([0], []) if data["spin"] else ([0], [0])) if freeze else ([], [])
    ovs = tuple((n - len(f), source.mo_coeff.shape[-1] - n)
                for n, f in zip(ns, frozen_sets))
    pairs = ((0, 0), (0, 1), (1, 1)) if data["unrestricted"] else ((0, 0),)
    assert seen == [(ovs[a] + ovs[b], source.max_memory if budget is None else budget)
                    for a, b in pairs]
    np.testing.assert_allclose(actual.e_corr, expected.e2, atol=2e-11, rtol=0)
    np.testing.assert_allclose(actual.e_tot, expected.total_energy, atol=2e-10, rtol=0)
    np.testing.assert_allclose(actual.e_corr, full.e_corr, atol=2e-10, rtol=0)
    np.testing.assert_allclose(actual.e_corr_ss, expected.same_spin_energy, atol=2e-11, rtol=0)
    np.testing.assert_allclose(actual.e_corr_os, expected.opposite_spin_energy, atol=2e-11, rtol=0)
    if with_t2:
        actual_t2 = actual.t2 if data["unrestricted"] else (actual.t2,)
        expected_t2 = expected.t2 if data["unrestricted"] else (expected.t2,)
        for a, e in zip(actual_t2, expected_t2):
            np.testing.assert_allclose(a, e, atol=2e-9, rtol=0)
    else:
        assert actual.t2 is None
    # Direct inputs retain the same source-freshness gate as full-integral SCF.
    monkeypatch.setattr(source, "mo_coeff", source.mo_coeff.at[..., 0].multiply(-1))
    with pytest.raises(RuntimeError, match="Orbital state changed"):
        actual.kernel()
    assert not actual.converged and actual.result is None and actual.t2 is None


def test_direct_mp2_matches_independent_pyscf_energy(direct_reference):
    pytest.importorskip("pyscf")
    from pyscf import gto, mp as pyscf_mp, scf
    from gradscf import mp

    data = direct_reference
    molecule = gto.M(atom=data["atom"], basis=data["basis"], spin=data["spin"],
                     cart=True, verbose=0)
    factory = scf.UHF if data["unrestricted"] else scf.RHF
    mf = factory(molecule).run(conv_tol=1e-13, conv_tol_grad=1e-10, max_cycle=200)
    assert mf.converged
    reference = pyscf_mp.MP2(mf).run()
    actual = mp.MP2(data["source"], with_t2=False).run()
    np.testing.assert_allclose(actual.e_corr, reference.e_corr, atol=2e-9, rtol=0)
    np.testing.assert_allclose(actual.e_tot, reference.e_tot, atol=2e-9, rtol=0)


@pytest.fixture(scope="module")
def legacy_jax_direct_rhf():
    from gradscf import gto, scf

    molecule = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", cart=True)
    options = dict(integral_backend="jax", execution_device="cpu",
                   conv_tol=1e-12, conv_tol_grad=1e-10, conv_tol_density=1e-11,
                   max_memory=128.)
    full = scf.RHF(molecule, **options).run()
    direct = scf.RHF(molecule, jk_backend="direct", direct_scf_tol=0., **options).run()
    assert full.converged and direct.converged
    return direct, full


def test_legacy_jax_direct_rhf_mp2_matches_full(legacy_jax_direct_rhf):
    from gradscf import mp
    from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis

    direct, full = legacy_jax_direct_rhf
    assert not isinstance(direct._scf_inputs.direct_basis,
                          (NativeDirectBasis, ProjectedNativeDirectBasis))
    expected, actual = mp.MP2(full).run(), mp.MP2(direct).run()
    np.testing.assert_allclose(actual.e_corr, expected.e_corr, atol=2e-11, rtol=0)
    np.testing.assert_allclose(actual.e_tot, expected.e_tot, atol=2e-10, rtol=0)
    np.testing.assert_allclose(actual.t2, expected.t2, atol=2e-9, rtol=0)


def test_legacy_jax_direct_rhf_budget_precedes_packed_materialization(
        legacy_jax_direct_rhf, monkeypatch):
    from gradscf import mp
    from gradscf.scf.inputs.types import RKSIntegralInputs

    direct, _ = legacy_jax_direct_rhf

    def forbidden(*args, **kwargs):
        pytest.fail("An insufficient MP budget reached legacy packed AO ERI materialization")

    monkeypatch.setattr(RKSIntegralInputs, "response_eri_pair_matrix", forbidden)
    with pytest.raises(ValueError, match="(?i)(budget|memory|resident|minimum)"):
        mp.MP2(direct, max_memory=.0001).run()
