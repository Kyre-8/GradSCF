"""HDF5 reference caches preserve streamed quadrature without storing AO."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.data.hdf5_cache import (
    read_restricted_molecule, read_unrestricted_molecule,
    write_restricted_molecule, write_unrestricted_molecule,
)
from gradscf.integrals.basis import CartesianAO, CartesianBasis
from gradscf.integrals.grids.ao import evaluate_cartesian_ao
from gradscf.integrals.grids.blocks import AOGrid
from gradscf.scf.molecules import QuadratureGrid, RestrictedMolecule, UnrestrictedMolecule


def _reference(*, unrestricted, streamed=True):
    basis = CartesianBasis(aos=(
        CartesianAO(center=jnp.asarray([-.2, .1, 0.]), angular=(0, 0, 0),
                    exponents=jnp.asarray([.7, 1.8]), coefficients=jnp.asarray([.6, .4])),
        CartesianAO(center=jnp.asarray([.4, -.2, .1]), angular=(1, 0, 0),
                    exponents=jnp.asarray([.9]), coefficients=jnp.asarray([.8])),
    ), precompute_eri_groups=False)
    coords = jnp.asarray([[0., 0., 0.], [.2, -.1, .3], [-.4, .5, -.2],
                          [.8, -.7, .1], [.1, .3, -.4], [-.2, -.1, .6], [.5, .2, .1]])
    weights = jnp.asarray([.2, .4, .1, .3, .15, .25, .35])
    provider = AOGrid.from_basis(basis, coords, weights, block_size=3)
    if streamed:
        ao, dao, lap = jnp.zeros((0, 2)), jnp.zeros((4, 0, 2)), None
    else:
        values = evaluate_cartesian_ao(basis, coords, deriv=2)
        ao, dao, lap = values[0], values[:4], values[4]
    molecule_type = UnrestrictedMolecule if unrestricted else RestrictedMolecule
    return molecule_type(
        ao=ao, ao_deriv1=dao, ao_laplacian=lap, grid_ao=provider if streamed else None,
        grid=QuadratureGrid(weights, coords), dipole_integrals=jnp.zeros((3, 2, 2)),
        rep_tensor=jnp.zeros((0, 0, 0, 0)), mo_coeff=jnp.stack((jnp.eye(2), jnp.eye(2))),
        mo_occ=jnp.asarray([[1., 0.], [1., 0.]]), mo_energy=jnp.asarray([[-1., .4], [-.9, .5]]),
        rdm1=jnp.asarray([[[.7, .08], [.08, .4]], [[.5, .03], [.03, .2]]]),
        h1e=jnp.eye(2), nuclear_repulsion=0.,
    )


def _cache_functions(unrestricted):
    if unrestricted:
        return write_unrestricted_molecule, read_unrestricted_molecule
    return write_restricted_molecule, read_restricted_molecule


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("array_backend", ["host", "jax"])
def test_streamed_grid_cache_roundtrip_preserves_ao_action(tmp_path, unrestricted, array_backend):
    h5py = pytest.importorskip("h5py")
    molecule = _reference(unrestricted=unrestricted)
    write, read = _cache_functions(unrestricted)
    path = tmp_path / "streamed.h5"
    with h5py.File(path, "w") as handle:
        write(handle.create_group("reference"), molecule)
    with h5py.File(path, "r") as handle:
        group = handle["reference"]
        assert "grid_ao" in group
        # The existing grid already owns coordinates and weights. The provider
        # subgroup contains only compact basis parameters, never full AO.
        assert set(group["grid_ao"]) == {"centers", "exponents", "coefficients"}
        loaded = read(group, array_backend=array_backend)
    assert type(loaded.grid_ao) is AOGrid
    assert loaded.ao.shape == (0, 2) and loaded.ao_deriv1.shape == (4, 0, 2)
    assert loaded.grid_ao.coords is loaded.grid.coords
    assert loaded.grid_ao.weights is loaded.grid.weights
    assert loaded.grid_ao.block_size == 3
    assert loaded.grid_ao.nprims == molecule.grid_ao.nprims
    assert loaded.grid_ao.angulars == molecule.grid_ao.angulars
    array_type = np.ndarray if array_backend == "host" else jax.Array
    for actual, expected in zip(jax.tree.leaves(loaded.grid_ao),
                                jax.tree.leaves(molecule.grid_ao), strict=True):
        assert isinstance(actual, array_type)
        np.testing.assert_array_equal(actual, expected)
    # Evaluate after closing the HDF5 file: the supplier must be self-contained.
    for actual, expected in zip(loaded.grid_ao.block(6, 3, deriv=2),
                                molecule.grid_ao.block(6, 3, deriv=2), strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(loaded.density(), molecule.density(), atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_rewriting_streamed_grid_cache_removes_stale_provider(tmp_path, unrestricted):
    h5py = pytest.importorskip("h5py")
    write, read = _cache_functions(unrestricted)
    with h5py.File(tmp_path / "rewrite.h5", "w") as handle:
        group = handle.create_group("reference")
        write(group, _reference(unrestricted=unrestricted))
        dense = _reference(unrestricted=unrestricted, streamed=False)
        write(group, dense)
        assert "grid_ao" not in group
        loaded = read(group)
    assert loaded.grid_ao is None
    np.testing.assert_array_equal(loaded.ao, dense.ao)
    np.testing.assert_allclose(loaded.density(), dense.density(), atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_unknown_grid_provider_is_rejected_before_cache_mutation(tmp_path, unrestricted):
    h5py = pytest.importorskip("h5py")
    write, _ = _cache_functions(unrestricted)
    molecule = replace(_reference(unrestricted=unrestricted), grid_ao=object())
    with h5py.File(tmp_path / "unknown.h5", "w") as handle:
        group = handle.create_group("reference")
        with pytest.raises(TypeError, match="grid.*provider|AO.*provider"):
            write(group, molecule)
        assert not list(group) and not list(group.attrs)


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("field", ["coords", "weights"])
def test_mismatched_provider_grid_is_rejected_before_cache_mutation(tmp_path, unrestricted, field):
    h5py = pytest.importorskip("h5py")
    write, _ = _cache_functions(unrestricted)
    molecule = _reference(unrestricted=unrestricted)
    molecule = replace(molecule, grid=replace(
        molecule.grid, **{field: getattr(molecule.grid, field) + .01}))
    with h5py.File(tmp_path / "mismatch.h5", "w") as handle:
        group = handle.create_group("reference")
        with pytest.raises(ValueError, match=f"provider {field}.*match"):
            write(group, molecule)
        assert not list(group) and not list(group.attrs)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_unknown_cached_grid_provider_kind_is_rejected(tmp_path, unrestricted):
    h5py = pytest.importorskip("h5py")
    write, read = _cache_functions(unrestricted)
    with h5py.File(tmp_path / "unknown-kind.h5", "w") as handle:
        group = handle.create_group("reference")
        write(group, _reference(unrestricted=unrestricted))
        group["grid_ao"].attrs["kind"] = "unknown"
        with pytest.raises(ValueError, match="AO grid provider kind"):
            read(group)
