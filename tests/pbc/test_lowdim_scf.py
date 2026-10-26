"""Neutral reduced-dimensional cells use one consistent electrostatic boundary."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.pbc import gto, dft
from gradscf.scf import SCFDifferentiationConfig


def test_slab_scf_bands_and_geometry_response_match_pyscf():
    from pyscf.pbc import gto as pgto, dft as pdft
    from pyscf.pbc.dft.gen_grid import UniformGrids
    lattice = np.diag([5., 6., 18.])
    coords = np.array([[.4, .5, 8.7], [1.8, .7, 9.2]])
    settings = dict(atom=list(zip(['H', 'H'], coords)), a=lattice, unit='Bohr',
        dimension=2, basis='gth-szv', pseudo='gth-pade', cart=True, mesh=(25, 31, 91), precision=1e-10)
    cell = gto.M(**settings)
    kpts = cell.make_kpts((2, 1, 1))
    mf = dft.KRKS(cell, kpts=kpts, xc='svwn', conv_tol=1e-12, conv_tol_density=1e-11,
        conv_tol_grad=1e-10, differentiation=SCFDifferentiationConfig(mode='implicit')).run()
    assert mf.converged
    def reference(positions):
        options = dict(settings, atom=list(zip(['H', 'H'], positions)), verbose=0)
        pc = pgto.M(**options)
        pmf = pdft.KRKS(pc, kpts=np.asarray(kpts))
        pmf.xc='svwn';pmf.grids=UniformGrids(pc);pmf.grids.mesh=pc.mesh
        pmf.conv_tol=1e-12;pmf.conv_tol_grad=1e-10;pmf.kernel()
        assert pmf.converged
        return pmf
    pmf = reference(coords)
    np.testing.assert_allclose(mf.e_tot, pmf.e_tot, atol=3e-7, rtol=0.)
    query = np.array([[.13, -.04, 0.], [.31, .07, 0.]])
    np.testing.assert_allclose(mf.get_bands(query)[0], pmf.get_bands(query)[0], atol=1e-6, rtol=0.)
    forces = -np.asarray(mf.nuc_grad_method().kernel())
    direction = np.array([[.2, -.1, .15], [-.1, .1, -.2]])
    h=1e-4
    fd=(reference(coords+h*direction).e_tot-reference(coords-h*direction).e_tot)/(2*h)
    np.testing.assert_allclose(-np.sum(forces*direction), fd, atol=2e-6, rtol=0.)


@pytest.mark.parametrize('dimension', [1, 2])
def test_lowdim_sampling_and_consumer_boundaries(dimension):
    cell = gto.M(atom='H .4 .5 8.7; H 1.8 .7 9.2', a=np.diag([5., 12., 18.]),
        unit='Bohr', dimension=dimension, basis='gth-szv', pseudo='gth-pade', mesh=(15, 31, 41))
    with pytest.raises(ValueError, match='nonperiodic'):
        cell.make_kpts((2, 2, 2))
    with pytest.raises(ValueError, match='nonperiodic'):
        cell.make_kpts((2, 1, 1), shift=(0., 0., .2))


def test_wire_scf_preserves_neutrality_and_vacuum_limit():
    # Same reciprocal cutoff; the rejected 12-bohr box retains visible tails.
    energies = []
    for length, transverse_mesh in [(16., 81), (20., 101)]:
        cell = gto.M(atom=f'H .4 {length/2} {length/2}; H 1.8 {length/2} {length/2}',
            a=np.diag([5., length, length]), unit='Bohr', dimension=1,
            basis='gth-szv', pseudo='gth-pade', mesh=(25, transverse_mesh, transverse_mesh), precision=1e-10)
        mf = dft.KRKS(cell, kpts=cell.make_kpts((2, 1, 1)), xc='svwn', conv_tol=1e-11,
            conv_tol_density=1e-10, conv_tol_grad=1e-9).run()
        assert mf.converged
        count=jnp.einsum('k,kpq,kqp->',mf.inputs.kweights,mf.make_rdm1(),mf.inputs.overlap).real
        np.testing.assert_allclose(count, 2., atol=2e-10)
        ao=mf.inputs.ao[:,0]
        rho=jnp.einsum('k,kgp,kpq,kgq->g',mf.inputs.kweights,ao,mf.make_rdm1(),ao.conj()).real
        grid=np.asarray(rho).reshape(cell.mesh)
        faces=max(np.max(grid[:,0,:]),np.max(grid[:,-1,:]),np.max(grid[:,:,0]),np.max(grid[:,:,-1]))
        assert faces<1e-9  # electrons / bohr^3 at the transverse cell faces
        energies.append(float(mf.e_tot))
        query=np.array([[.13, 0., 0.]])
        assert np.isfinite(mf.get_bands(query)[0]).all()
        with pytest.raises(NotImplementedError, match='dimension|3D|three-dimensional'):
            from gradscf.gw.pbc import KRGW
            KRGW(mf).kernel()
    np.testing.assert_allclose(*energies, atol=3e-6, rtol=0.)
