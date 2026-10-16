import jax.numpy as jnp
import numpy as np
from gradscf.pbc import gto
from gradscf.integrals.periodic.coulomb import ewald_energy


def test_cell_units_kpoints_and_neutrality():
    cell=gto.Cell(atom='He 0 0 0',a=np.eye(3)*4,unit='Bohr',mesh=(15,15,15)).build()
    assert cell.nelectron==2
    np.testing.assert_allclose(cell.reciprocal_vectors(),np.eye(3)*np.pi/2)
    points=cell.make_kpts((2,1,1))
    assert points.shape==(2,3)
    assert cell.volume==64.


def test_ewald_madelung_matches_cubic_constant():
    a=jnp.eye(3)*5
    energy=ewald_energy(a,jnp.zeros((1,3)),jnp.ones(1),precision=1e-10)
    np.testing.assert_allclose(-2*energy,2.837297479480619/5,atol=1e-10,rtol=0)


def test_unsupported_cell_and_coincident_images_are_rejected():
    import pytest
    with pytest.raises(ValueError,match='coincide'):
        gto.M(atom='H 0 0 0; H 5 0 0',a=np.eye(3)*5,unit='Bohr')
    with pytest.raises(NotImplementedError,match='dimension'):
        gto.M(atom='He 0 0 0',a=np.eye(3)*5,dimension=0)
    with pytest.raises(ValueError,match='orthogonal'):
        gto.M(atom='He 0 0 0',a=np.array([[5.,0.,0.],[0.,5.,0.],[1.,0.,12.]]),dimension=2)
