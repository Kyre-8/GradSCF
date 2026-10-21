"""Shared HF source preparation retains the existing MP adapter boundary."""
from importlib import import_module
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


def _source(unrestricted=False, representation="eri"):
    from gradscf.scf.facade import RKS, UKS
    from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis

    source = object.__new__(UKS if unrestricted else RKS)
    c = jnp.asarray(np.eye(4), dtype=jnp.float32)
    fock = jnp.diag(jnp.array([-1.3, -.7, .3, .8], dtype=jnp.float64))
    fock = fock.at[0, 1].set(.02).at[1, 0].set(.02)
    hcore = jnp.eye(4, dtype=jnp.float32)
    data = SimpleNamespace(hcore=hcore, h1e=hcore, nelectron=4,
        nocc_alpha=2, nocc_beta=1, fock_matrix=(fock, fock+.1*jnp.eye(4)),
        eri=None, rep_tensor=None, eri_pair_matrix=None, df_factors=None,
        direct_basis=None, overlap=None, ao=None, ao_deriv1=None,
        ao_laplacian=None, coords=None, grid_weights=None)
    if representation in ("eri", "s4-in-eri"):
        data.eri = jnp.zeros((4,)*4 if representation == "eri" else (10, 10))
        data.rep_tensor = data.eri
        key, value = ("eri" if representation == "eri" else "eri_pair_matrix"), data.eri
    elif representation == "pair":
        data.eri_pair_matrix = jnp.zeros((10, 10))
        key, value = "eri_pair_matrix", data.eri_pair_matrix
    elif representation == "df":
        data.df_factors = jnp.zeros((3, 4, 4))
        # Existing DF precedence is retained even if a direct provider exists.
        data.direct_basis = object.__new__(NativeDirectBasis)
        key, value = "df_factors", data.df_factors
    else:
        cls = NativeDirectBasis if representation == "native" else ProjectedNativeDirectBasis
        data.direct_basis = object.__new__(cls)
        key, value = "direct_basis", data.direct_basis
    source.xc, source.max_memory, source.e_tot = "hf", 128., jnp.asarray(-7.)
    source.mo_coeff = jnp.stack((c, c)) if unrestricted else c
    source.mo_occ = jnp.array([[1, 1, 0, 0], [1, 0, 0, 0]]) if unrestricted else jnp.array([2, 2, 0, 0])
    source.reference = data if unrestricted else None
    source._scf_inputs = None if unrestricted else data
    data.mf_energy = source.e_tot
    source.scf_result = SimpleNamespace(fock_matrix=fock, total_energy=source.e_tot)
    checks = []
    source._check_reference_source = lambda: checks.append("current")

    def forbidden(*args, **kwargs):
        pytest.fail("Source preparation must not rebuild a reference, J/K or MO ERIs")

    source.to_reference = source._ensure_reference = source._build_reference = forbidden
    data.response_eri_pair_matrix = forbidden
    return source, data, checks, key, value


@pytest.mark.parametrize("unrestricted", [False, True], ids=["rhf", "uhf"])
@pytest.mark.parametrize("representation", ["eri", "s4-in-eri", "df", "native", "projected"])
def test_shared_hf_source_preserves_current_orbitals_and_unique_ao_data(unrestricted, representation):
    from gradscf.ao2mo.scf import prepare_scf_source
    from gradscf.ao2mo.reference import transform_fock

    source, data, checks, key, value = _source(unrestricted, representation)
    payload = prepare_scf_source(source)
    assert checks == ["current"]
    assert payload.nocc == ((2, 1) if unrestricted else 2)
    assert payload.reference_energy is source.e_tot and payload.hcore is data.hcore
    assert list(payload.ao_kwargs) == [key] and payload.ao_kwargs[key] is value
    cs = payload.coefficients if unrestricted else (payload.coefficients,)
    fs = data.fock_matrix if unrestricted else (source.scf_result.fock_matrix,)
    assert len(payload.focks) == len(cs) == (2 if unrestricted else 1)
    for c, f, actual in zip(cs, fs, payload.focks):
        assert c.dtype == jnp.float32 and actual.dtype == jnp.float64
        np.testing.assert_array_equal(actual, transform_fock(f, c))
        assert abs(float(actual[0, 1])) > .01


def test_restricted_pair_storage_is_preserved():
    from gradscf.ao2mo.scf import prepare_scf_source

    source, _, _, key, value = _source(representation="pair")
    payload = prepare_scf_source(source)
    assert list(payload.ao_kwargs) == [key] and payload.ao_kwargs[key] is value


@pytest.mark.parametrize("unrestricted", [False, True])
def test_mp_from_scf_uses_shared_source_and_existing_orbital_preparation(unrestricted, monkeypatch):
    integrals = import_module("gradscf.mp.integrals")
    source, data, _, key, value = _source(unrestricted)
    payload = SimpleNamespace(coefficients=source.mo_coeff, focks=(jnp.eye(4),),
        nocc=(2, 1) if unrestricted else 2, reference_energy=source.e_tot,
        hcore=data.hcore, ao_kwargs={key: value})
    calls = []

    def prepare(current, *, max_memory):
        calls.append((current, max_memory))
        return payload

    def from_orbitals(*args, **kwargs):
        assert args[0] is payload.coefficients and args[1] is payload.focks
        assert args[2] == payload.nocc and args[3] is payload.reference_energy
        assert kwargs.pop(key) is value
        assert kwargs == dict(frozen=[0], frozen_gap_tol=2e-9, max_memory=17.5)
        return "prepared"

    monkeypatch.setattr(integrals, "prepare_scf_source", prepare, raising=False)
    monkeypatch.setattr(integrals, "from_orbitals", from_orbitals)
    assert integrals.from_scf(source, frozen=[0], frozen_gap_tol=2e-9, max_memory=17.5) == "prepared"
    assert calls == [(source, 17.5)]


@pytest.mark.parametrize("failure", ["freshness", "occupations", "odd", "dft", "complex", "missing-fock"])
def test_shared_source_retains_validation_gates(failure):
    from gradscf.ao2mo.scf import prepare_scf_source

    source, data, checks, _, _ = _source(failure == "missing-fock")
    error, match = ValueError, "occupations"
    if failure == "freshness":
        def stale():
            raise RuntimeError("Orbital state changed")
        source._check_reference_source = stale
        error, match = RuntimeError, "Orbital state changed"
    elif failure == "occupations":
        source.mo_occ = jnp.array([2., 1.5, .5, 0.])
    elif failure == "odd":
        data.nelectron = 3
        match = "even electron"
    elif failure == "dft":
        source.xc = "pbe"
        error, match = NotImplementedError, "RHF/UHF"
    elif failure == "complex":
        source.mo_coeff = source.mo_coeff.astype(jnp.complex128)
        error, match = NotImplementedError, "real orbitals"
    else:
        data.fock_matrix = None
        error, match = RuntimeError, "Fock data is missing"
    with pytest.raises(error, match=match):
        prepare_scf_source(source)
    assert checks == ([] if failure in ("freshness", "dft") else ["current"])


def test_legacy_jax_packed_preflight_still_precedes_materialization():
    from gradscf.ao2mo.scf import prepare_scf_source

    source, data, _, _, _ = _source()
    data.eri, data.direct_basis = None, object()
    calls, pair = [], jnp.zeros((10, 10))

    def packed():
        calls.append("packed")
        return pair

    data.response_eri_pair_matrix = packed
    with pytest.raises(ValueError, match="(?i)(budget|memory|resident|minimum)"):
        prepare_scf_source(source, max_memory=.0001)
    assert calls == []
    payload = prepare_scf_source(source, max_memory=1.)
    assert calls == ["packed"] and payload.ao_kwargs == {"eri_pair_matrix": pair}


@pytest.mark.parametrize("unrestricted", [False, True], ids=["rhf", "uhf"])
def test_solved_energy_is_independent_of_mutable_public_alias(unrestricted):
    from gradscf.ao2mo.scf import prepare_scf_source

    source, _, _, _, _ = _source(unrestricted)
    solved = source.reference.mf_energy if unrestricted else source.scf_result.total_energy
    source.e_tot = source.e_tot + .5
    payload = prepare_scf_source(source)
    assert payload.reference_energy is solved


@pytest.mark.parametrize("unrestricted", [False, True], ids=["rhf", "uhf"])
@pytest.mark.parametrize("missing", [False, True], ids=["none", "absent"])
def test_missing_solved_energy_is_rejected_without_public_alias_fallback(unrestricted, missing):
    from gradscf.ao2mo.scf import prepare_scf_source

    source, _, _, _, _ = _source(unrestricted)
    record = source.reference if unrestricted else source.scf_result
    field = "mf_energy" if unrestricted else "total_energy"
    if missing:
        delattr(record, field)
    else:
        setattr(record, field, None)
    with pytest.raises(RuntimeError, match="(?i)energy.*missing"):
        prepare_scf_source(source)
