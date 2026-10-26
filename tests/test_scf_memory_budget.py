"""A molecular memory budget survives SCF configuration and saved references."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from gradscf import dft, gto
from gradscf.data.hdf5_cache import (
    read_restricted_molecule, read_unrestricted_molecule,
    write_restricted_molecule, write_unrestricted_molecule,
)
from gradscf.scf.molecules import QuadratureGrid, RestrictedMolecule, UnrestrictedMolecule
from gradscf.scf.rks import RKSConfig
from gradscf.scf.uks import UKSConfig
from gradscf.integrals.grids.blocks import AOGrid


@pytest.mark.parametrize("facade, config", [(dft.RKS, RKSConfig), (dft.UKS, UKSConfig)])
def test_scf_memory_defaults_and_explicit_grid_cap(facade, config):
    mol = gto.M(atom="H 0 0 0; H 0 0 .8", basis="sto-3g")
    mf = facade(mol)
    assert getattr(mf, "max_memory", None) == 60000.
    assert mf.grid_max_memory is None
    assert mf._config().max_memory == config().max_memory == 60000.
    assert mf._config().grid_max_memory is None

    mf = facade(mol, max_memory=128., grid_max_memory=16.)
    assert mf._config().max_memory == 128.
    assert mf._config().grid_max_memory == 16.


@pytest.mark.parametrize("config", [RKSConfig, UKSConfig])
@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), True])
def test_scf_configs_reject_invalid_total_memory(config, value):
    with pytest.raises(ValueError, match="max_memory"):
        config(max_memory=value)


@pytest.mark.parametrize("config", [RKSConfig, UKSConfig])
@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), True])
def test_scf_configs_reject_invalid_explicit_grid_memory(config, value):
    with pytest.raises(ValueError, match="grid_max_memory"):
        config(grid_max_memory=value)


@pytest.mark.parametrize("facade", [dft.RKS, dft.UKS])
def test_memory_change_invalidates_scf_source_signature(facade):
    mol = gto.M(atom="H 0 0 0; H 0 0 .8", basis="sto-3g")
    mf = facade(mol)
    original = mf._scf_signature()
    mf.max_memory = 128.
    assert mf._scf_signature() != original


def _reference(unrestricted):
    molecule_type = UnrestrictedMolecule if unrestricted else RestrictedMolecule
    return molecule_type(
        ao=np.ones((3, 2)), grid=QuadratureGrid(np.ones(3), np.zeros((3, 3))),
        dipole_integrals=np.zeros((3, 2, 2)), rep_tensor=np.zeros((0, 0, 0, 0)),
        mo_coeff=np.stack([np.eye(2)] * 2), mo_occ=np.array([[1., 0.]] * 2),
        mo_energy=np.array([[-.5, .2]] * 2), rdm1=np.stack([np.eye(2)] * 2),
        h1e=np.eye(2), nuclear_repulsion=0.,
    )


@pytest.mark.parametrize("unrestricted", [False, True])
def test_reference_memory_is_static_pytree_metadata(unrestricted):
    reference = replace(_reference(unrestricted), max_memory=321.)
    children, metadata = reference.tree_flatten()
    assert all(child is not reference.max_memory for child in children)
    restored = type(reference).tree_unflatten(metadata, children)
    assert restored.max_memory == 321.
    assert jax.tree.structure(reference) != jax.tree.structure(replace(reference, max_memory=322.))


@pytest.mark.parametrize("unrestricted", [False, True])
def test_reference_memory_cache_roundtrip_and_legacy_default(tmp_path, unrestricted):
    h5py = pytest.importorskip("h5py")
    write, read = ((write_unrestricted_molecule, read_unrestricted_molecule) if unrestricted else
                   (write_restricted_molecule, read_restricted_molecule))
    reference = replace(_reference(unrestricted), max_memory=321.)
    with h5py.File(tmp_path / "memory.h5", "w") as archive:
        group = archive.create_group("reference")
        write(group, reference)
        assert group.attrs["max_memory"] == 321.
        restored = read(group, array_backend="host")
        assert restored.max_memory == 321.
        np.testing.assert_array_equal(restored.mo_coeff, reference.mo_coeff)
        del group.attrs["max_memory"]
        assert read(group, array_backend="host").max_memory == 60000.


def test_reference_cache_preserves_adaptive_grid_and_legacy_fixed_default(tmp_path):
    h5py = pytest.importorskip("h5py")
    reference = _reference(False)
    provider = AOGrid(reference.grid.coords, reference.grid.weights,
                      np.zeros((2, 3)), np.ones((2, 1)), np.ones((2, 1)),
                      (1, 1), ((0, 0, 0), (0, 0, 0)), 2, adaptive=True)
    reference = replace(reference, grid_ao=provider)
    with h5py.File(tmp_path / "adaptive.h5", "w") as archive:
        group = archive.create_group("reference")
        write_restricted_molecule(group, reference)
        assert bool(group["grid_ao"].attrs["adaptive"])
        assert read_restricted_molecule(group, array_backend="host").grid_ao.adaptive
        del group["grid_ao"].attrs["adaptive"]
        assert not read_restricted_molecule(group, array_backend="host").grid_ao.adaptive


@pytest.mark.parametrize("facade", [dft.RKS, dft.UKS])
def test_converged_public_reference_retains_total_memory(facade):
    mol = gto.M(atom="H 0 0 0; H 0 0 .8", basis="sto-3g", cart=True)
    mf = facade(mol, xc="hf", integral_backend="native", jk_backend="direct",
                execution_device="cpu", max_memory=128., grid_block_size=3,
                direct_scf_workers=1, max_cycle=40).run()
    assert mf.converged
    reference = mf.to_reference()
    assert reference.max_memory == 128.
    assert reference.direct_basis is not None
    assert reference.grid_ao is not None
