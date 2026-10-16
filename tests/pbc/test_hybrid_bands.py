"""Smooth hybrid bands with a declared finite-cell exchange quadrature."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.pbc import gto, dft, scf
from gradscf.integrals.periodic.coulomb import exchange_kernel


def test_spherical_exchange_kernel_matches_pyscf_and_strain_fd():
    from pyscf.pbc import gto as pgto, tools
    lattice=np.eye(3)*6
    pc=pgto.M(atom='He 0 0 0',a=lattice,unit='Bohr',basis='gth-szv',
              pseudo='gth-pade',mesh=(9,)*3,verbose=0)
    kpts=pc.make_kpts((2,1,1))
    from types import SimpleNamespace
    vectors=pc.get_Gv()
    for shift in (np.zeros(3),np.array([.13,-.07,.04])):
        expected=tools.get_coulG(pc,k=shift,exx='vcut_sph',mf=SimpleNamespace(kpts=kpts),
                                Gv=vectors,wrap_around=False)
        actual=exchange_kernel(jnp.asarray(vectors+shift),lattice=jnp.asarray(lattice),nk=2,exxdiv='vcut_sph')
        np.testing.assert_allclose(actual,expected,atol=2e-11,rtol=2e-12)
    def loss(scale):
        a=jnp.asarray(lattice)*scale
        values=exchange_kernel(jnp.asarray(vectors)/scale,lattice=a,nk=2,exxdiv='vcut_sph')
        return jnp.sum(values**2)*1e-4
    h=1e-4
    np.testing.assert_allclose(jax.grad(jax.grad(loss))(1.),
        (jax.grad(loss)(1+h)-jax.grad(loss)(1-h))/(2*h),rtol=2e-7)


def test_pbe0_spherical_scf_bands_match_independent_pyscf():
    from pyscf.pbc import gto as pgto,dft as pdft
    from pyscf.pbc.dft.gen_grid import UniformGrids
    options=dict(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6,unit='Bohr',
        basis='gth-szv',pseudo='gth-pade',mesh=(31,)*3,cart=True,precision=1e-10)
    cell=gto.M(**options);kpts=cell.make_kpts((2,1,1))
    mf=dft.KRKS(cell,kpts=kpts,xc='pbe0',exxdiv='vcut_sph',conv_tol=1e-12,
                conv_tol_density=1e-11,conv_tol_grad=1e-10).run()
    pc=pgto.M(**options,verbose=0)
    pmf=pdft.KRKS(pc,kpts=np.asarray(kpts),exxdiv='vcut_sph')
    pmf.xc='pbe0';pmf.grids=UniformGrids(pc);pmf.grids.mesh=pc.mesh
    pmf.conv_tol=1e-12;pmf.conv_tol_grad=1e-10;pmf.kernel()
    assert mf.converged and pmf.converged
    np.testing.assert_allclose(mf.e_tot,pmf.e_tot,atol=2e-6,rtol=0.)
    query=np.array([[.11,.03,-.07],[.21,-.09,.02],[0.,0.,0.],[1e-5,0.,0.],[-1e-5,0.,0.]])
    saved=mf.result
    actual,_=mf.get_bands(query,chunk_size=2)
    expected,_=pmf.get_bands(query)
    assert mf.result is saved
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=0.)
    np.testing.assert_allclose(actual[3:],np.broadcast_to(actual[2],(2,2)),atol=5e-7,rtol=0.)
    np.testing.assert_allclose(mf.get_bands(kpts)[0],mf.mo_energy,atol=2e-7,rtol=0.)


def test_gamma_spherical_hf_uses_exchange_cutoff_without_changing_hartree():
    from pyscf.pbc import gto as pgto,scf as pscf
    options=dict(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6,unit='Bohr',
        basis='gth-szv',pseudo='gth-pade',mesh=(31,)*3,cart=True)
    mf=scf.RHF(gto.M(**options),exxdiv='vcut_sph',conv_tol=1e-11).run()
    pmf=pscf.RHF(pgto.M(**options,verbose=0),exxdiv='vcut_sph').run(conv_tol=1e-11)
    assert mf.converged and pmf.converged
    np.testing.assert_allclose(mf.e_tot,pmf.e_tot,atol=2e-6,rtol=0.)
    np.testing.assert_allclose(mf.get_bands(np.zeros((1,3)))[0],mf.mo_energy,atol=2e-7,rtol=0.)
