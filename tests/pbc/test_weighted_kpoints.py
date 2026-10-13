import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_weights_enter_energy_and_fock_consistently():
    from gradscf.pbc import gto
    from gradscf.integrals.periodic.fft import build_inputs
    from gradscf.pbc._kpoint import energy_and_fock
    cell = gto.M(atom='H 0 0 0; H 1.4 0 0', a=np.eye(3)*6., unit='Bohr', mesh=(9,)*3)
    data = build_inputs(cell,kpts=cell.make_kpts((2,1,1)),kweights=[.3,.7])
    d = jnp.broadcast_to(jnp.eye(2)*.2,(2,2,2,2)).astype(complex)
    b = jnp.broadcast_to(jnp.array([[.2,.1j],[-.1j,-.1]]),(2,2,2,2))
    e,f = energy_and_fock(d,data,mesh=cell.mesh,exxdiv=None)
    derivative = jax.grad(lambda t:energy_and_fock(d+t*b,data,mesh=cell.mesh,exxdiv=None)[0])(0.)
    expected = jnp.einsum('k,skij,skji->',data.kweights,b,f).real
    np.testing.assert_allclose(derivative,expected,atol=1e-10)
    with pytest.raises(ValueError,match='weights'):
        build_inputs(cell,kpts=cell.make_kpts((2,1,1)),kweights=[-.1,1.1])


def test_shifted_mesh_momentum_and_time_reversal():
    from gradscf.pbc.kpoints import uniform_kpoints, symmetry_reduce
    from gradscf.pbc.momentum import momentum_transfer_table
    lattice = jnp.eye(3)*2*jnp.pi
    k = uniform_kpoints(lattice,(4,1,1),shift=(.5,0,0))
    q = uniform_kpoints(lattice,(4,1,1))
    table = momentum_transfer_table(k,qpts=q)
    for i in range(4):
        for j in range(4):
            delta = np.asarray(k[i]-q[j]-k[table[i,j]])
            np.testing.assert_allclose(delta-np.rint(delta),0.,atol=1e-12)
    reduced = symmetry_reduce(k)
    assert len(reduced.representatives) == 2
    np.testing.assert_allclose(reduced.weights,[.5,.5])


def test_supplied_reciprocal_rotations_reduce_scalar_sums():
    from gradscf.pbc.kpoints import uniform_kpoints,symmetry_reduce
    k=uniform_kpoints(jnp.eye(3)*2*jnp.pi,(2,2,2))
    rotation=np.array([[0,1,0],[0,0,1],[1,0,0]])
    reduced=symmetry_reduce(k,rotations=np.stack([np.eye(3),rotation,rotation@rotation]),time_reversal=False)
    assert len(reduced.representatives)==4
    values=np.cos(2*np.pi*np.asarray(k)).sum(axis=1)
    np.testing.assert_allclose(values.mean(),reduced.weights@values[reduced.representatives],atol=1e-12)
