"""Native primitive geometry curvature, checked by independent supercell AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.pbc import gto,dft


def _mixed_curvature(function,coordinates,u,v):
    def mixed(a,b):
        return jax.jvp(lambda r:jax.jvp(function,(r,),(a,))[1],(coordinates,),(b,))[1]
    return mixed(u.real,v.real)+mixed(u.imag,v.imag)+1j*(mixed(u.real,v.imag)-mixed(u.imag,v.real))


@pytest.mark.parametrize('repetitions,q',[(3,1/3),(2,.5),(1,0.)])
@pytest.mark.parametrize('outside',[False,True])
def test_ewald_q_curvature_matches_supercell(repetitions,q,outside):
    from gradscf.integrals.periodic.coulomb import ewald_energy,ewald_curvature
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6,unit='Bohr',mesh=(7,)*3)
    u=jnp.array([[.3,-.2,.1],[-.4,.1,.2]])
    v=jnp.array([[.2,.1,-.2],[.5,-.3,.4]])
    coordinates=cell.coords.at[1].add(cell.lattice[0]) if outside else cell.coords
    indices=jnp.arange(repetitions)
    positions=(coordinates[None]+indices[:,None,None]*cell.lattice[0]).reshape(-1,3)
    lattice=jnp.diag(jnp.array([repetitions,1,1]))@cell.lattice
    phase=jnp.exp(2j*jnp.pi*indices*q)
    pu=(phase[:,None,None]*u).reshape(-1,3);pv=(phase[:,None,None]*v).reshape(-1,3)
    energy=lambda r:ewald_energy(lattice,r,jnp.tile(cell.charges,repetitions),reference_lattice=lattice)
    expected=_mixed_curvature(energy,positions,pu,pv)/repetitions
    actual=ewald_curvature(cell.lattice,coordinates,cell.charges,jnp.array([q,0.,0.]),u,v)
    np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=0)


@pytest.mark.parametrize('xc',['hf','hf-noewald','pbe','r2scan'])
def test_bare_geometry_curvature_matches_supercell(xc):
    from gradscf.dfpt.force_constants import geometry_curvature
    from gradscf.pbc.phonon import _replica
    from gradscf.integrals.periodic.fft import build_inputs,get_kpoint_jk
    from gradscf.pbc._kpoint import grid_xc_potential
    from gradscf.dft.libxc_jax.jax_libxc import hybrid_coeff
    exxdiv=None if xc=='hf-noewald' else 'ewald'
    xc='hf' if xc=='hf-noewald' else xc
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6,unit='Bohr',mesh=(15,)*3)
    mf=dft.KRKS(cell,kpts=cell.make_kpts((3,1,1)),xc=xc,temperature=.1,exxdiv=exxdiv).run(
        conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10)
    assert mf.converged
    u=jnp.array([[.3,-.2,.1],[-.4,.1,.2]])
    v=jnp.array([[.2,.1,-.2],[.5,-.3,.4]])
    q=np.array([1/3,0,0]);sup,indices,translations=_replica(cell,np.array([3,1,1]))
    b=jnp.kron(jnp.exp(1j*translations@mf.kpts.T)/jnp.sqrt(3.),jnp.eye(cell.topology.nao))
    fold=lambda d:jnp.stack([b@jax.scipy.linalg.block_diag(*spin)@b.conj().T for spin in d])[:,None]
    density=fold(mf.result.density_spin)
    w=mf.result.density_spin@mf.result.fock_spin@jnp.linalg.inv(mf.inputs.overlap)
    w=fold(.5*(w+w.conj().swapaxes(-1,-2)))
    alpha=hybrid_coeff(xc)
    def components(r):
        data=build_inputs(sup,parameters=sup.parameters_at(r))
        j,k=get_kpoint_jk(data,density,mesh=sup.mesh,with_k=bool(alpha),exxdiv=exxdiv)
        trace=lambda a:jnp.einsum('skij,skji->',density,a).real
        return jnp.array([trace(data.hcore[None]),.5*trace(j[None]),
            -.5*alpha*trace(k),grid_xc_potential(density,data,xc=xc)[0],
            -jnp.einsum('skij,kji->',w,data.overlap).real,data.nuclear_repulsion])
    phase=jnp.exp(2j*jnp.pi*indices[:,0]/3)
    expected=_mixed_curvature(components,sup.coords,(phase[:,None,None]*u).reshape(-1,3),
                               (phase[:,None,None]*v).reshape(-1,3))/3
    actual=geometry_curvature(mf,q,u,v)
    print(xc,'native components',actual,'supercell',expected,flush=True)
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=0)


def test_nonlocal_geometry_curvature_matches_supercell():
    from gradscf.dfpt.displacement import _orbitals
    from gradscf.dfpt.force_constants import _nonlocal_curvature,_mixed
    from gradscf.integrals.periodic.fft import build_inputs
    from gradscf.integrals.periodic.ao import reciprocal_grid,fourier_ao
    from gradscf.integrals.periodic.pseudo import nonlocal_matrix
    from gradscf.pbc.momentum import momentum_transfer
    from gradscf.pbc.phonon import _replica
    cell=gto.M(atom='C .4 .5 .6; H 1.8 .8 .9',spin=1,a=np.eye(3)*6,unit='Bohr',mesh=(21,)*3)
    assert any(channel[1] for pp in cell.pseudopotentials for channel in pp[5:])
    k=cell.make_kpts((3,1,1));data=build_inputs(cell,kpts=k);q=np.array([1/3,0,0])
    fractional=np.asarray(k)@np.asarray(cell.lattice).T/(2*np.pi)
    p=momentum_transfer(fractional,q).partners;m=momentum_transfer(fractional,-q).partners
    u=jnp.array([[.3,-.2,.1],[-.4,.1,.2]]);v=jnp.array([[.2,.1,-.2],[.5,-.3,.4]])
    coeff=lambda r:_orbitals(cell,data,r)[1]
    n=cell.topology.nao
    density=jnp.broadcast_to(.2*jnp.eye(n)+.01*jnp.ones((n,n)),(2,3,n,n)).astype(complex)
    def native(u,v):
        ft,fu=jax.jvp(coeff,(cell.coords,),(u,));fv=jax.jvp(coeff,(cell.coords,),(v,))[1]
        fuv=_mixed(coeff,cell.coords,u,v)
        return _nonlocal_curvature(cell,data,density,ft,fu,fv,fuv,u,v,p,m)
    actual=jax.jit(native)(u,v)
    sup,indices,translations=_replica(cell,np.array([3,1,1]))
    b=jnp.kron(jnp.exp(1j*translations@k.T)/jnp.sqrt(3.),jnp.eye(n))
    ds=jnp.stack([b@jax.scipy.linalg.block_diag(*d)@b.conj().T for d in density])
    gv=reciprocal_grid(sup.lattice,sup.mesh)
    def energy(r):
        c=fourier_ao(sup.topology,sup.parameters_at(r),gv)
        matrix=nonlocal_matrix(gv,c,r,sup.pseudopotentials,sup.volume)
        return jnp.einsum('sij,ji->',ds,matrix).real
    phase=jnp.exp(2j*jnp.pi*indices[:,0]/3)
    expected=_mixed_curvature(energy,sup.coords,(phase[:,None,None]*u).reshape(-1,3),
                               (phase[:,None,None]*v).reshape(-1,3))/3
    print('nonlocal curvature error',float(jnp.abs(actual-expected)),flush=True)
    np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=0)


@pytest.mark.parametrize('spin',[0,1])
def test_primitive_phonons_never_construct_supercell(monkeypatch,spin):
    from gradscf.pbc import phonon
    atom='H .4 .5 .6; H 1.8 .8 .9'+('; H 2.9 1.2 1.1' if spin else '')
    cell=gto.M(atom=atom,spin=spin,a=np.eye(3)*6,unit='Bohr',mesh=(15,)*3)
    cls=dft.KUKS if spin else dft.KRKS
    mf=cls(cell,kpts=cell.make_kpts((3,1,1)),xc='hf',temperature=.1).run(
        conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10)
    driver=phonon.Phonons(mf,masses=[1.,2.,1.] if spin else [1.,2.])
    expected=driver.kernel([1/3,0,0],method='supercell')
    def forbidden(*args,**kwargs):raise AssertionError('Native phonons must not construct a supercell')
    monkeypatch.setattr(phonon,'_replica',forbidden)
    actual=driver.kernel([1/3,0,0],method='primitive')
    assert actual.converged and driver.super_reference is None
    np.testing.assert_allclose(actual.force_constants,expected.force_constants,atol=2e-6,rtol=0)
    if not spin:
        negative=driver.kernel([-1/3,0,0],method='primitive')
        np.testing.assert_allclose(negative.force_constants,actual.force_constants.conj(),atol=1e-7,rtol=0)


def test_primitive_phonons_shifted_even_mesh(monkeypatch):
    from gradscf.pbc import phonon
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6,unit='Bohr',mesh=(41,)*3)
    mf=dft.KRKS(cell,kpts=cell.make_kpts((2,1,1),shift=(.5,0,0)),xc='hf',temperature=.1).run(
        conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10)
    def forbidden(*args,**kwargs):raise AssertionError('Native phonons must not construct a supercell')
    monkeypatch.setattr(phonon,'_replica',forbidden)
    result=phonon.Phonons(mf,masses=[1.,1.],max_grid=100_000).kernel([-.5,0,0],method='primitive')
    assert result.converged and result.hermiticity_error<1e-6
    np.testing.assert_allclose(result.force_constants.imag,0.,atol=1e-6,rtol=0)
