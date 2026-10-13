"""Cartesian forces and tensile-positive symmetric strain derivatives."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.pbc import gto, scf
from gradscf.scf import SCFDifferentiationConfig


def test_shared_derivative_convention_and_parameter_ad():
    from gradscf.pbc.grad import energy_and_derivatives
    coords=jnp.array([[.4,.8,1.],[1.8,.6,1.3]])
    lattice=jnp.array([[5.,.3,0.],[0.,5.5,.2],[.1,0.,6.]])
    def energy(p,r,a):return p*(.3*jnp.sum(r*r)+.2*jnp.sum(a*a))+.07*jnp.linalg.det(a)
    out=jax.jit(lambda p:energy_and_derivatives(energy,p,coords,lattice))(.4)
    np.testing.assert_allclose(out.forces,-.6*.4*coords,atol=1e-13)
    expected=(.6*.4*coords.T@coords+.4*.4*lattice.T@lattice)/jnp.linalg.det(lattice)+.07*jnp.eye(3)
    np.testing.assert_allclose(out.stress,expected,atol=1e-13)
    shear=jnp.array([[.1,.3,0.],[.3,-.2,.1],[0.,.1,.2]])
    def strained(t):
        deformation=jnp.eye(3)+t*shear
        return energy(.4,coords@deformation.T,lattice@deformation.T)
    np.testing.assert_allclose(jnp.sum(out.stress*shear)*jnp.linalg.det(lattice),
                              (strained(1e-5)-strained(-1e-5))/2e-5,atol=1e-8)
    slope=jax.grad(lambda p:jnp.sum(energy_and_derivatives(energy,p,coords,lattice).forces))(.4)
    np.testing.assert_allclose(slope,-.6*jnp.sum(coords),atol=1e-12)


def test_cell_parameters_move_all_shells():
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .6 .8',a=np.eye(3)*6.,unit='Bohr',basis='gth-dzvp')
    coords=cell.coords+jnp.array([[.1,0.,0.],[0.,.2,0.]])
    moved=cell.parameters_at(coords)
    owners=np.argmin(np.linalg.norm(np.asarray(cell.parameters.centers)[:,None]-np.asarray(cell.coords)[None],axis=-1),axis=1)
    np.testing.assert_allclose(moved.centers,coords[owners],atol=0)
    assert len(moved.centers)>len(coords)
    np.testing.assert_array_equal(moved.nuclear_coords,coords)


@pytest.mark.parametrize('nk,temperature,mode,external',[(1,None,'explicit',False),(2,.1,'implicit',False),(2,.1,'implicit',True)])
def test_scf_forces_and_stress_match_resolved_energy(nk,temperature,mode,external):
    from gradscf import dft
    from gradscf.pbc import dft as pdft
    from gradscf.integrals.periodic.fft import build_inputs
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6.,unit='Bohr',mesh=(11,)*3)
    kpts=cell.make_kpts((nk,1,1),shift=(.5,0,0)) if nk>1 else cell.make_kpts((1,1,1))
    controls=dict(conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10,max_cycle=120,damping=.1,
                  differentiation=SCFDifferentiationConfig(mode=mode,tolerance=1e-11,max_iter=80))
    cls=scf.KRHF if nk>1 else scf.RHF
    if external:
        cls=pdft.KRKS
        model=dft.Functional(lambda s:(s.rho,s.weights),lambda p,x:p*jnp.dot(x[1],x[0]**2))
        controls.update(functional=model,params=.02)
    mf=cls(cell,kpts=kpts,temperature=temperature,**controls).run()
    driver=mf.nuc_grad_method()
    derivative=driver.kernel()
    assert driver.result.forces.shape==cell.coords.shape
    np.testing.assert_allclose(derivative,-driver.result.forces,atol=0)
    np.testing.assert_allclose(driver.result.energy,mf.e_free,atol=1e-10)
    def energy(r,a):
        data=build_inputs(cell,parameters=replace(cell.parameters,centers=r,nuclear_coords=r),lattice=a,kpts=kpts)
        result=scf.run_kpoint_scf(data,mesh=cell.mesh,nelec=cell.nelec,temperature=temperature,**controls)
        assert result.converged
        return result.free_energy
    step=1e-4
    fd=(energy(cell.coords.at[1,0].add(step),cell.lattice)-energy(cell.coords.at[1,0].add(-step),cell.lattice))/(2*step)
    np.testing.assert_allclose(derivative[1,0],fd,atol=2e-6,rtol=0)
    strain=jnp.array([[.1,.2,0.],[.2,-.3,.1],[0.,.1,.2]])
    plus=jnp.eye(3)+step*strain;minus=jnp.eye(3)-step*strain
    fd=(energy(cell.coords@plus.T,cell.lattice@plus.T)-energy(cell.coords@minus.T,cell.lattice@minus.T))/(2*step)
    np.testing.assert_allclose(jnp.sum(driver.result.stress*strain)*cell.volume,fd,atol=3e-6,rtol=0)


def test_unconverged_geometry_derivative_is_rejected():
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6.,unit='Bohr',mesh=(7,)*3)
    mf=scf.RHF(cell,max_cycle=1)
    driver=mf.nuc_grad_method()
    with pytest.raises(RuntimeError,match='convergence'):
        driver.kernel()
    assert driver.result is None
    with pytest.raises(RuntimeError,match='convergence'):
        driver.stress()


def test_gamma_force_and_pressure_against_pyscf():
    pytest.importorskip('pyscf')
    from pyscf.pbc import gto as pgto,scf as pscf
    kwargs=dict(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6.,unit='Bohr',
                basis='gth-szv',pseudo='gth-pade',mesh=(41,)*3,precision=1e-10)
    cell=gto.M(**kwargs)
    mf=scf.RHF(cell,conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10,
               differentiation=SCFDifferentiationConfig(tolerance=1e-11,max_iter=80))
    driver=mf.nuc_grad_method()
    gradient=driver.kernel()
    def reference(r,a):
        kw=dict(kwargs,atom=[('H',tuple(row)) for row in np.asarray(r)],a=np.asarray(a))
        calc=pscf.RHF(pgto.M(**kw,cart=True,verbose=0))
        calc.conv_tol=1e-12;calc.conv_tol_grad=1e-10
        value=calc.kernel()
        assert calc.converged
        return value
    step=1e-4
    fd=(reference(cell.coords.at[1,0].add(step),cell.lattice)
        -reference(cell.coords.at[1,0].add(-step),cell.lattice))/(2*step)
    np.testing.assert_allclose(gradient[1,0],fd,atol=3e-6,rtol=0)
    fd=(reference(cell.coords*(1+step),cell.lattice*(1+step))
        -reference(cell.coords*(1-step),cell.lattice*(1-step)))/(2*step)
    np.testing.assert_allclose(jnp.trace(driver.stress())*cell.volume,fd,atol=5e-6,rtol=0)
    np.testing.assert_allclose(jnp.sum(gradient,axis=0),0.,atol=1e-7)


def test_geometry_driver_does_not_replace_a_different_reference_state():
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6.,unit='Bohr',mesh=(7,)*3)
    mf=scf.RHF(cell).run()
    # A result inconsistent with a fresh solve must not silently acquire that
    # other solution's force. This also guards cold restarts in multistable SCF.
    mf.result=mf.result._replace(density_spin=.8*mf.result.density_spin)
    with pytest.raises(RuntimeError,match='reference'):
        mf.nuc_grad_method().kernel()
