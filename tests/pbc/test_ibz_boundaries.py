"""Joint spatial/finite-q integration and nonunitary/native contraction gates."""
import jax.numpy as jnp
import numpy as np
from gradscf import dft,dfpt
from gradscf.pbc import gto,scf,dft as pdft
from gradscf.pbc.symmetry import Operation
from gradscf.pbc.reduction import SpatialReduction
from gradscf.pbc._kpoint import energy_and_fock
from gradscf.pbc.occupations import fermi_density
from gradscf.integrals.periodic.fft import build_inputs


def test_combined_symmetry_does_not_require_a_separate_tr_closed_mesh():
    cell=gto.M(atom='He 0 0 0',a=np.eye(3)*6.,unit='Bohr',mesh=(7,)*3)
    points=cell.make_kpts((1,1,1),shift=(.2,.1,.3))
    mf=scf.KRHF(cell,kpts=points,symmetry=True,time_reversal=True,exxdiv=None).run()
    assert mf.converged
    assert mf.reduction.consistent(mf.result.fock_spin)


def test_nonunitary_cartesian_native_jk_and_energy():
    cell=gto.M(atom='He 0 0 0',a=[[5.,0,0],[2.5,np.sqrt(3)*2.5,0],[0,0,7.]],
        unit='Bohr',mesh=(7,)*3,basis={'He':[[0,[.8,1.]],[1,[.6,1.]],[2,[.5,1.]]]})
    points=cell.make_kpts((3,3,1));data=build_inputs(cell,kpts=points)
    op=Operation.from_cell(cell,[[0,1,0],[1,0,0],[0,0,1]])
    plan=SpatialReduction.from_cell(cell,points,operations=[op])
    assert len(plan.representatives)<len(points)
    f=jnp.broadcast_to(data.hcore,(2,)+data.hcore.shape)
    density=fermi_density(f,data.overlap,jnp.array(cell.nelec),jnp.array(.2),data.kweights)[0]
    assert plan.consistent_density(density)
    reference=energy_and_fock(density,data,mesh=cell.mesh,exxdiv=None)
    actual=energy_and_fock(density,data,mesh=cell.mesh,exxdiv=None,integration=plan)
    np.testing.assert_allclose(actual[0],reference[0],atol=2e-9)
    np.testing.assert_allclose(actual[1],reference[1],atol=2e-9)


def test_reduced_external_scf_feeds_full_finite_q_response():
    from gradscf.pbc.functional import density_inputs
    from gradscf.pbc.response import q_density
    from gradscf.pbc.kpoints import reciprocal_vectors
    cell=gto.M(atom='H -.7 0 0; H .7 0 0',a=np.eye(3)*6.,unit='Bohr',mesh=(9,)*3)
    points=cell.make_kpts((3,1,1))
    def energy(p,x):return p*jnp.dot(x.weights,x.rho**3)/3
    def q_energy(p,a,d,t,z):
        base=density_inputs(a,d).rho
        envelope=q_density(a,t,z).sum(axis=0)
        r=a.coords[None]+jnp.arange(3)[:,None,None]*a.lattice[0]
        phase=jnp.exp(1j*r@(jnp.asarray(t.q)@reciprocal_vectors(a.lattice)))
        rho=base+2*(phase*envelope).real
        return p*jnp.mean(jnp.sum(a.weights*rho**3,axis=-1))/3
    functional=dft.Functional(lambda state:state,energy,q_energy_fn=q_energy)
    options=dict(functional=functional,params=.15,temperature=.2,exxdiv=None)
    controls=dict(max_cycle=100,damping=.1,conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10)
    full=pdft.KRKS(cell,kpts=points,**options).run(**controls)
    reduced=pdft.KRKS(cell,kpts=points,symmetry=True,**options).run(**controls)
    assert full.converged and reduced.converged
    assert len(reduced.reduction.representatives)<len(points)
    potential=jnp.broadcast_to(jnp.array([[.1,.03j],[.02,-.2]]),full.inputs.hcore.shape)
    left=dfpt.DFPT(full,q=[1/3,0,0]).response(potential=potential)
    right=dfpt.DFPT(reduced,q=[1/3,0,0]).response(potential=potential)
    assert left.converged and right.converged
    np.testing.assert_allclose(left.density,right.density,atol=1e-9)


def test_symmetry_tied_basis_exponent_gradient():
    from dataclasses import replace
    import jax
    from gradscf.pbc._kpoint import run_kpoint_scf
    from gradscf.scf import SCFDifferentiationConfig
    cell=gto.M(atom='H -.7 0 0; H .7 0 0',a=np.eye(3)*6.,unit='Bohr',mesh=(9,)*3)
    points=cell.make_kpts((3,1,1))
    plan=SpatialReduction.from_cell(cell,points,operations=[Operation.from_cell(cell,-np.eye(3))])
    def energy(scale,reduction):
        parameters=replace(cell.parameters,exponents=tuple(a*scale for a in cell.parameters.exponents))
        data=build_inputs(cell,parameters=parameters,kpts=points)
        result=run_kpoint_scf(data,mesh=cell.mesh,nelec=cell.nelec,temperature=.2,exxdiv=None,
            reduction=reduction,max_cycle=100,damping=.1,conv_tol=1e-12,conv_tol_density=1e-11,
            conv_tol_grad=1e-10,differentiation=SCFDifferentiationConfig(mode='implicit',tolerance=1e-11,max_iter=80))
        return result.free_energy
    derivative=jax.jit(jax.grad(lambda a:energy(a,plan)))(1.)
    full=jax.jit(lambda a:energy(a,None))
    finite=(full(1.+1e-5)-full(1.-1e-5))/2e-5
    assert abs(float(derivative))>1e-5
    np.testing.assert_allclose(derivative,finite,atol=1e-6,rtol=1e-5)
