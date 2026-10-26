"""Force-constant interpolation against analytic crystals, not fitted outputs."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _cell(positions):
    return SimpleNamespace(lattice=np.eye(3)*4.,coords=np.asarray(positions)*4.)


@pytest.mark.parametrize('size',[2,3,4])
def test_monatomic_chain_roundtrip_and_off_grid(size):
    from gradscf.pbc.ifc import ForceConstants
    q=np.zeros((size,3));q[:,0]=np.arange(size)/size
    matrices=2*(1-np.cos(2*np.pi*q[:,0]))[:,None,None]*np.eye(3)
    model=ForceConstants.from_qmesh(_cell([[0,0,0]]),q,matrices,masses=[1.])
    np.testing.assert_allclose(model.matrix(q),matrices,atol=1e-12)
    points=jnp.array([[.13,0,0],[.27,0,0],[.5,0,0]])
    expected=2*(1-jnp.cos(2*jnp.pi*points[:,0]))[:,None,None]*jnp.eye(3)
    np.testing.assert_allclose(model.matrix(points),expected,atol=1e-12)
    np.testing.assert_allclose(model.matrix(points+jnp.array([1,0,0])),expected,atol=1e-12)
    loss=lambda x:jnp.trace(model.matrix(jnp.array([x,0.,0.]))).real
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(.21),12*np.pi*np.sin(2*np.pi*.21),atol=1e-10)


def test_diatomic_shortest_images_and_cell_gauge():
    from gradscf.pbc.ifc import ForceConstants
    gamma=np.kron([[2.,-2.],[-2.,2.]],np.eye(3))[None]
    model=ForceConstants.from_qmesh(_cell([[0,0,0],[.5,0,0]]),np.zeros((1,3)),gamma,masses=[1.,2.])
    x=.27;q=np.array([x,0.,0.]);phase=np.exp(2j*np.pi*x)
    expected=np.kron([[2.,-(1+phase.conjugate())],[-(1+phase),2.]],np.eye(3))
    np.testing.assert_allclose(model.matrix(q),expected,atol=1e-12)
    shifted=ForceConstants.from_qmesh(_cell([[0,0,0],[1.5,0,0]]),np.zeros((1,3)),gamma,masses=[1.,2.])
    gauge=np.diag(np.repeat([1.,np.exp(-2j*np.pi*x)],3))
    np.testing.assert_allclose(shifted.matrix(q),gauge.conj().T@expected@gauge,atol=1e-12)
    a=model.bands(np.array([[.1,0,0],[.3,0,0]]))
    b=shifted.bands(np.array([[.1,0,0],[.3,0,0]]))
    np.testing.assert_allclose(a.frequencies,b.frequencies,atol=1e-10)


def test_grid_ordering_validation_and_explicit_asr():
    from gradscf.pbc.ifc import ForceConstants
    q=np.array([[0.,0,0],[1/3,0,0],[-1/3,0,0]])
    matrices=2*(1-np.cos(2*np.pi*q[:,0]))[:,None,None]*np.eye(3)
    matrices[0]+=.01*np.eye(3)
    cell=_cell([[0,0,0]])
    model=ForceConstants.from_qmesh(cell,q[[2,0,1]],matrices[[2,0,1]],masses=[1.],asr=True)
    assert model.raw_asr_residual>0
    np.testing.assert_allclose(model.matrix([0,0,0]),0.,atol=1e-12)
    np.testing.assert_allclose(model.matrix(q[1:]),matrices[1:],atol=1e-12)
    for invalid in (q[:2],q+.01):
        with pytest.raises(ValueError,match='grid|mesh'):
            ForceConstants.from_qmesh(cell,invalid,matrices[:len(invalid)],masses=[1.])
    bad=matrices.copy().astype(complex);bad[1,0,0]+=.2
    with pytest.raises(ValueError,match='conjug|real'):
        ForceConstants.from_qmesh(cell,q,bad,masses=[1.])


def test_modes_mass_scaling_and_dos_normalization():
    from gradscf.pbc.ifc import ForceConstants
    q=np.array([[0.,0,0],[.5,0,0]])
    matrices=np.array([np.eye(3),3*np.eye(3)])
    model=ForceConstants.from_qmesh(_cell([[0,0,0]]),q,matrices,masses=[1.])
    heavier=ForceConstants.from_qmesh(_cell([[0,0,0]]),q,matrices,masses=[4.])
    modes=model.bands(q)
    np.testing.assert_allclose(heavier.bands(q).frequencies,modes.frequencies/2,atol=1e-12)
    energies=np.linspace(-.02,.08,2001)
    dos=model.dos(q,energies,width=.001)
    np.testing.assert_allclose(np.trapezoid(dos,energies),3.,atol=1e-7)
    np.testing.assert_allclose(model.dos(np.tile(q,(50,1)),energies,width=.001),dos,atol=1e-10)


def test_asr_cannot_hide_invalid_raw_conjugacy():
    from gradscf.pbc.ifc import ForceConstants
    gamma=np.array([[1.,1j,0.],[-1j,1.,0.],[0.,0.,1.]])[None]
    with pytest.raises(ValueError,match='conjug|real'):
        ForceConstants.from_qmesh(_cell([[0,0,0]]),np.zeros((1,3)),gamma,masses=[1.],asr=True)


def test_integer_asr_and_mixed_query_precision():
    from gradscf.pbc.ifc import ForceConstants
    gamma=np.diag([1,0,0,0,0,0])[None]
    model=ForceConstants.from_qmesh(_cell([[0,0,0],[.5,0,0]]),np.zeros((1,3)),gamma,masses=[1.,1.],asr=True)
    expected=np.zeros((6,6));expected[np.ix_([0,3],[0,3])]=[[.25,-.25],[-.25,.25]]
    np.testing.assert_allclose(model.matrix([0.,0.,0.]),expected,atol=1e-12)
    assert model.matrix(jnp.array([.2,0,0],dtype=jnp.float32)).dtype==jnp.complex128


def test_multidimensional_grid_accepts_mixed_reciprocal_representatives():
    from gradscf.pbc.ifc import ForceConstants
    import itertools
    q=np.array(list(itertools.product(np.arange(3)/3,np.arange(2)/2,[0.])))
    matrices=(2*(1-np.cos(2*np.pi*q[:,0]))+2*(1-np.cos(2*np.pi*q[:,1])))[:,None,None]*np.eye(3)
    q[4,0]-=1.
    model=ForceConstants.from_qmesh(_cell([[0,0,0]]),q,matrices,masses=[1.])
    assert model.mesh==(3,2,1)
    np.testing.assert_allclose(model.matrix(q),matrices,atol=1e-12)


def test_oblique_cell_uses_cartesian_shortest_images():
    from gradscf.pbc.ifc import ForceConstants
    lattice=np.array([[4.,0.,0.],[2.,3.,0.],[0.,0.,5.]])
    cell=SimpleNamespace(lattice=lattice,coords=np.array([[0.,0.,0.],[.49,.49,0.]])@lattice)
    gamma=np.kron([[1.,-1.],[-1.,1.]],np.eye(3))[None]
    model=ForceConstants.from_qmesh(cell,np.zeros((1,3)),gamma,masses=[1.,1.])
    q=np.array([.13,.23,0.]);phase=np.exp(-2j*np.pi*q[1])
    expected=np.kron([[1.,-phase],[-phase.conjugate(),1.]],np.eye(3))
    np.testing.assert_allclose(model.matrix(q),expected,atol=1e-12)


def test_qmesh_driver_preserves_direct_results_and_checks_closure(monkeypatch):
    from gradscf.pbc import gto,scf
    from gradscf.pbc.phonon import Phonons
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6,unit='Bohr',mesh=(15,)*3)
    mf=scf.KRHF(cell,kpts=cell.make_kpts((3,1,1))).run(
        conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10)
    driver=Phonons(mf,masses=[1.,1.])
    model=driver.mesh((3,1,1),asr=True)
    assert model.mesh==(3,1,1) and driver.super_reference is None
    np.testing.assert_allclose(model.matrix([-1/3,0,0]),driver.result.force_constants,atol=1e-8)
    np.testing.assert_allclose(model.matrix([0,0,0])@np.tile(np.eye(3),(2,1)),0.,atol=1e-10)
    def forbidden(*args,**kwargs):raise AssertionError('An incompatible q mesh must fail before any phonon solve')
    monkeypatch.setattr(driver,'kernel',forbidden)
    with pytest.raises(ValueError,match='commensurate'):
        driver.mesh((2,1,1))
