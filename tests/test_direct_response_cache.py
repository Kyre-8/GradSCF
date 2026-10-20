"""Direct response caches preserve the bound AO action without storing ERIs."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.data.hdf5_cache import (
    read_restricted_molecule, write_restricted_molecule,
    read_unrestricted_molecule, write_unrestricted_molecule,
)
from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis
from gradscf.scf.molecules import QuadratureGrid, RestrictedMolecule, UnrestrictedMolecule


@pytest.fixture(scope="module")
def native_provider():
    topology, parameters = integrals.prepare_basis(
        "H 0 0 0; H 0 0 .8", "6-31g", cart=False)
    return NativeDirectBasis(integrals.make_plan(topology), parameters)


def _provider(native, kind):
    if kind == "native":
        return native
    transform = np.arange(native.nao * 3, dtype=np.float64).reshape(native.nao, 3) / 17
    projected = ProjectedNativeDirectBasis(native, transform)
    if kind == "projected":
        return projected
    return ProjectedNativeDirectBasis(projected, np.asarray([[.8, .1], [.2, .9], [.4, -.3]]))


def _reference(provider):
    n = provider.nao
    return RestrictedMolecule(
        ao=np.zeros((1, n)), grid=QuadratureGrid(np.ones(1)),
        dipole_integrals=np.zeros((3, n, n)), rep_tensor=np.zeros((0, 0, 0, 0)),
        mo_coeff=np.eye(n), mo_occ=np.r_[2., np.zeros(n - 1)],
        mo_energy=np.linspace(-.5, .5, n), rdm1=np.zeros((2, n, n)),
        h1e=np.eye(n), nuclear_repulsion=0., nocc=1,
        exact_exchange_fraction=1., direct_basis=provider)


def _assert_provider(actual, expected, array_backend):
    assert type(actual) is type(expected)
    array_type = np.ndarray if array_backend == "host" else jax.Array
    if isinstance(expected, ProjectedNativeDirectBasis):
        assert isinstance(actual.transform, array_type)
        np.testing.assert_array_equal(actual.transform, expected.transform)
        _assert_provider(actual.primitive_basis, expected.primitive_basis, array_backend)
    else:
        assert actual.plan.topology == expected.plan.topology
        assert actual.plan.backend == expected.plan.backend
        for value, reference in zip(jax.tree.leaves(actual.parameters),
                                    jax.tree.leaves(expected.parameters), strict=True):
            assert isinstance(value, array_type)
            np.testing.assert_array_equal(value, reference)


@pytest.mark.parametrize("array_backend", ["host", "jax"])
@pytest.mark.parametrize("kind", ["native", "projected"])
def test_unrestricted_direct_cache_preserves_provider_and_discards_stale_integrals(
        tmp_path, native_provider, monkeypatch, array_backend, kind):
    from dataclasses import fields
    h5py = pytest.importorskip("h5py")
    provider = _provider(native_provider, kind)
    restricted = _reference(provider)
    values = {field.name: getattr(restricted, field.name)
              for field in fields(UnrestrictedMolecule) if hasattr(restricted, field.name)}
    n = provider.nao
    values.update(mo_coeff=np.stack([np.eye(n)] * 2),
                  mo_occ=np.stack([np.r_[1., np.zeros(n-1)], np.zeros(n)]),
                  mo_energy=np.stack([np.linspace(-.5, .5, n)] * 2),
                  nocc_alpha=1, nocc_beta=0, direct_basis=provider)
    molecule = UnrestrictedMolecule(**values)
    density = jnp.eye(n)
    expected = provider.get_jk(density)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unrestricted direct cache must not evaluate ERIs")

    monkeypatch.setattr(type(native_provider.plan), "evaluate", forbidden)
    path = tmp_path / "unrestricted_direct.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("reference")
        for field in ("eri_pair_matrix", "df_factors", "eri_ovov"):
            group.create_dataset(field, data=np.ones((2, 2)))
        write_unrestricted_molecule(group, molecule)
        assert "direct_basis" in group
        assert all(field not in group for field in ("eri_pair_matrix", "df_factors", "eri_ovov"))
        restored = read_unrestricted_molecule(group, array_backend=array_backend)
    _assert_provider(restored.direct_basis, provider, array_backend)
    assert restored.rep_tensor.size == 0 and restored.eri_pair_matrix is None
    assert (restored.nocc_alpha, restored.nocc_beta) == (1, 0)
    for actual, reference in zip(restored.direct_basis.get_jk(density), expected):
        np.testing.assert_allclose(actual, reference, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("array_backend", ["host", "jax"])
@pytest.mark.parametrize("kind", ["native", "projected"])
def test_direct_provider_roundtrip_retains_action_without_eri_materialization(
        tmp_path, native_provider, monkeypatch, array_backend, kind):
    h5py = pytest.importorskip("h5py")
    provider = _provider(native_provider, kind)
    molecule = _reference(provider)
    density = jnp.arange(provider.nao**2, dtype=jnp.float64).reshape(provider.nao, provider.nao) / 23
    expected = provider.get_jk(density)

    def forbidden(*args, **kwargs):
        raise AssertionError("Caching a direct provider must not materialize ERIs")

    monkeypatch.setattr(NativeDirectBasis, "response_eri_pair_matrix", forbidden)
    monkeypatch.setattr(type(native_provider.plan), "evaluate", forbidden)
    path = tmp_path / "direct.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("reference")
        write_restricted_molecule(group, molecule)
        datasets = []
        group["direct_basis"].visititems(
            lambda name, value: datasets.append(value) if isinstance(value, h5py.Dataset) else None)
        assert datasets and all(value.ndim <= 2 for value in datasets)
        # Only basis data and contraction matrices belong in this subgroup.
        assert sum(value.size * value.dtype.itemsize for value in datasets) < 4096
    with h5py.File(path, "r") as handle:
        loaded = read_restricted_molecule(handle["reference"], array_backend=array_backend)
    _assert_provider(loaded.direct_basis, provider, array_backend)
    assert loaded.rep_tensor.size == 0
    assert loaded.eri_pair_matrix is None and loaded.df_factors is None
    actual = jax.jit(lambda dm: loaded.direct_basis.get_jk(dm))(density)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, atol=2e-11, rtol=0)


@pytest.mark.parametrize("array_backend", ["host", "jax"])
def test_nested_projected_provider_structure_roundtrips(tmp_path, native_provider, array_backend):
    h5py = pytest.importorskip("h5py")
    provider = _provider(native_provider, "nested_projected")
    with h5py.File(tmp_path / "nested.h5", "w") as handle:
        group = handle.create_group("reference")
        write_restricted_molecule(group, _reference(provider))
        loaded = read_restricted_molecule(group, array_backend=array_backend)
    # Nested provider serialization does not add support for a nested action:
    # the existing projected operator expects a primitive native action.
    _assert_provider(loaded.direct_basis, provider, array_backend)


def test_rewriting_group_replaces_and_removes_previous_direct_provider(tmp_path, native_provider):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "rewrite.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("reference")
        for kind in ("nested_projected", "native", "projected"):
            provider = _provider(native_provider, kind)
            write_restricted_molecule(group, _reference(provider))
            loaded = read_restricted_molecule(group, array_backend="host")
            _assert_provider(loaded.direct_basis, provider, "host")
        plain = replace(_reference(native_provider), direct_basis=None,
                        rep_tensor=np.ones((native_provider.nao,) * 4))
        write_restricted_molecule(group, plain)
        assert "direct_basis" not in group
        loaded = read_restricted_molecule(group, array_backend="host")
        assert loaded.direct_basis is None
        np.testing.assert_array_equal(loaded.rep_tensor, plain.rep_tensor)


def test_rewriting_array_cache_as_direct_removes_stale_two_electron_datasets(tmp_path, native_provider):
    h5py = pytest.importorskip("h5py")
    direct = _reference(native_provider)
    n = native_provider.nao
    npair, nv = n * (n + 1) // 2, n - 1
    stale_arrays = {
        "eri_pair_matrix": np.ones((npair, npair)),
        "df_factors": np.ones((2, n, n)),
        "response_df_factors_j": np.ones((3, n, n)),
        "response_df_factors_k": np.ones((1, n, n)),
        "eri_ovov": np.ones((1, nv, 1, nv)),
        "eri_ovvo": np.ones((1, nv, nv, 1)),
        "eri_oovv": np.ones((1, 1, nv, nv)),
    }
    old = replace(direct, direct_basis=None, **stale_arrays)
    with h5py.File(tmp_path / "array-to-direct.h5", "w") as handle:
        group = handle.create_group("reference")
        write_restricted_molecule(group, old)
        write_restricted_molecule(group, direct)
        loaded = read_restricted_molecule(group, array_backend="host")
        for field in stale_arrays:
            assert getattr(loaded, field) is None, field
            assert field not in group
    assert loaded.rep_tensor.size == 0
    _assert_provider(loaded.direct_basis, native_provider, "host")
    density = jnp.arange(n**2, dtype=jnp.float64).reshape(n, n) / 23
    expected = native_provider.get_jk(density)
    actual = jax.jit(lambda dm: loaded.direct_basis.get_jk(dm))(density)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, atol=2e-11, rtol=0)


@pytest.mark.parametrize("nested", [False, True])
def test_unknown_direct_provider_is_rejected_before_cache_write(tmp_path, native_provider, nested):
    h5py = pytest.importorskip("h5py")

    class UnknownProvider:
        nao = native_provider.nao

    unknown = UnknownProvider()
    provider = (ProjectedNativeDirectBasis(unknown, np.eye(unknown.nao))
                if nested else unknown)
    with h5py.File(tmp_path / "unknown.h5", "w") as handle:
        group = handle.create_group("reference")
        with pytest.raises(TypeError, match="direct.*provider|Direct.*provider"):
            write_restricted_molecule(group, _reference(provider))
        assert not list(group) and not list(group.attrs)


def test_unknown_cached_provider_kind_is_rejected(tmp_path, native_provider):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "unknown-kind.h5", "w") as handle:
        group = handle.create_group("reference")
        write_restricted_molecule(group, _reference(native_provider))
        group["direct_basis"].attrs["kind"] = "unknown"
        with pytest.raises(ValueError, match="direct.*provider|Direct.*provider"):
            read_restricted_molecule(group)


@pytest.mark.parametrize("array_backend", ["host", "jax"])
def test_existing_array_cache_remains_readable(tmp_path, native_provider, array_backend):
    h5py = pytest.importorskip("h5py")
    molecule = replace(_reference(native_provider), direct_basis=None,
                       rep_tensor=np.ones((native_provider.nao,) * 4))
    with h5py.File(tmp_path / "legacy.h5", "w") as handle:
        group = handle.create_group("reference")
        write_restricted_molecule(group, molecule)
        assert "direct_basis" not in group
        loaded = read_restricted_molecule(group, array_backend=array_backend)
    assert loaded.direct_basis is None
    np.testing.assert_array_equal(loaded.rep_tensor, molecule.rep_tensor)
