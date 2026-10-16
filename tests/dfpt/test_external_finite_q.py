"""External finite-q energies: independent real-space supercell Hessian oracle."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.dft import Functional
from gradscf.pbc.momentum import momentum_transfer


def _toy(nk):
    """Two orbitals/cell, nonlinear local and inter-site density energy."""
    n=2
    fractional=np.zeros((nk,3));fractional[:,0]=np.arange(nk)/nk
    transfer=momentum_transfer(fractional,[1/nk,0,0])
    transform=jnp.kron(jnp.exp(2j*jnp.pi*jnp.arange(nk)[:,None]*jnp.arange(nk)/nk)/jnp.sqrt(nk),jnp.eye(n))
    inputs=SimpleNamespace(kpoints=jnp.asarray(fractional),kweights=jnp.ones(nk)/nk,
        ao=jnp.zeros((nk,4,1,n)),overlap=jnp.broadcast_to(jnp.eye(n),(nk,n,n)))
    def assemble(d):
        return jax.vmap(lambda x:jax.scipy.linalg.block_diag(*x))(d)
    def energy(p,matrix):
        realspace=transform@matrix@transform.conj().T
        rho=jnp.diagonal(realspace,axis1=-2,axis2=-1).real.sum(axis=0)
        # Translation-invariant nonlinear local and nonlocal contributions.
        return p*(jnp.sum(rho**3)/3+.17*jnp.sum(jnp.abs(realspace)**4))/nk
    def q_energy(p,a,d,t,z):
        matrix=assemble(d)
        offset=jnp.zeros_like(matrix,dtype=complex)
        for k,kq in enumerate(t.partners):
            offset=offset.at[:,n*kq:n*(kq+1),n*k:n*(k+1)].set(z[:,k])
        return energy(p,matrix+offset+offset.conj().swapaxes(-1,-2))
    functional=Functional(lambda v:v.density_matrix,lambda p,d:energy(p,assemble(d)),q_energy_fn=q_energy)
    rng=np.random.default_rng(34)
    a=rng.normal(size=(2,nk,n,n))+1j*rng.normal(size=(2,nk,n,n))
    d=jnp.asarray(.1*(a+a.conj().swapaxes(-1,-2)))+.4*jnp.eye(n)
    z=jnp.asarray(rng.normal(size=d.shape)+1j*rng.normal(size=d.shape))*.04
    return inputs,transfer,functional,d,z,assemble,energy


@pytest.mark.parametrize('nk',[2,3])
def test_external_q_hessian_matches_independent_supercell_finite_difference(nk):
    a,t,f,d,z,assemble,energy=_toy(nk)
    inverse=np.argsort(t.partners)
    adjoint=lambda x:x[:,inverse].conj().swapaxes(-1,-2)
    if nk==2:z=.5*(z+adjoint(z))
    matrix=assemble(d);offset=jnp.zeros_like(matrix,dtype=complex)
    for k,kq in enumerate(t.partners):
        offset=offset.at[:,2*kq:2*kq+2,2*k:2*k+2].set(z[:,k])
    physical=offset+offset.conj().swapaxes(-1,-2)
    potential=lambda x:nk*jax.grad(lambda v:energy(.3,v))(x).conj()
    step=1e-5
    fd=(potential(matrix+step*physical)-potential(matrix-step*physical))/(2*step)
    expected=jnp.stack([fd[:,2*kq:2*kq+2,2*k:2*k+2] for k,kq in enumerate(t.partners)],axis=1)
    if nk==2:expected/=2
    actual=f.q_kernel_action(.3,a,d,t,z)
    np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=1e-8)
    np.testing.assert_allclose(f.q_kernel_action(.3,a,d,t,1j*z),1j*actual,atol=1e-12)
    negative=momentum_transfer(np.asarray(a.kpoints),-t.q)
    np.testing.assert_allclose(f.q_kernel_action(.3,a,d,negative,adjoint(z)),adjoint(actual),atol=1e-12)
    loss=lambda p:jnp.sum(jnp.abs(f.q_kernel_action(p,a,d*(1+p),t,z))**2)
    derivative=jax.grad(loss)(.3)
    fd_loss=(loss(.30001)-loss(.29999))/2e-5
    np.testing.assert_allclose(derivative,fd_loss,atol=1e-9,rtol=1e-7)


def test_external_q_requires_explicit_real_scalar_extension():
    f=Functional(lambda x:x.rho,lambda p,x:jnp.sum(x))
    with pytest.raises(NotImplementedError,match='q_energy_fn'):
        f.q_kernel_action(None,None,jnp.zeros((2,2,1,1)),None,jnp.zeros((2,2,1,1)))
    a,t,_,d,z,_,_=_toy(2)
    for invalid in (lambda *args:jnp.ones(2),lambda *args:jnp.array(1j)):
        f=Functional(lambda x:x.rho,lambda p,x:jnp.sum(x),q_energy_fn=invalid)
        with pytest.raises(ValueError,match='real scalar'):
            f.q_kernel_action(None,a,d,t,z)


def _grid_functional(nk):
    """Nonlinear density/gradient energy, extended on translated cell grids."""
    from gradscf.pbc.functional import density_inputs
    from gradscf.pbc.response import q_density
    from gradscf.pbc.kpoints import reciprocal_vectors,weights_for
    def local(rho,gradient):
        return rho**3/3+.02*rho*jnp.sum(gradient**2,axis=-1)
    def energy(p,x):
        return p*jnp.dot(x.weights,local(x.rho,x.grad_rho))
    def q_energy(p,a,d,t,z):
        base=density_inputs(a,d)
        rho=q_density(a,t,z).sum(axis=0)
        g=jnp.asarray(t.shifts)@reciprocal_vectors(a.lattice)
        plus=a.ao[t.partners]*jnp.exp(1j*g@a.coords.T)[:,None,:,None]
        # Independent physical gradient contractions, with covariant Bloch AO
        # derivatives: do not differentiate only the periodic envelope.
        gradient=jnp.einsum('k,kxgp,skpq,kgq->gx',weights_for(a),plus[:,1:4],z,a.ao[:,0].conj())
        gradient+=jnp.einsum('k,kgp,skpq,kxgq->gx',weights_for(a),plus[:,0],z,a.ao[:,1:4].conj())
        translations=jnp.arange(nk)[:,None]*a.lattice[0]
        phase=jnp.exp(1j*(a.coords[None]+translations[:,None])@(jnp.asarray(t.q)@reciprocal_vectors(a.lattice)))
        physical_rho=base.rho+2*(phase*rho).real
        physical_gradient=base.grad_rho+2*(phase[...,None]*gradient).real
        return p*jnp.mean(jnp.sum(a.weights*local(physical_rho,physical_gradient),axis=-1))
    return Functional(lambda x:x,energy,q_energy_fn=q_energy)


def _cell(nk):
    from gradscf.pbc import gto
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6.,unit='Bohr',mesh=(9,)*3)
    return cell,cell.make_kpts((nk,1,1))


OPTIONS=dict(conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10,max_cycle=120,damping=.1)


@pytest.mark.parametrize('nk',[2,3])
def test_external_q_screening_facade(nk):
    from gradscf import dfpt
    from gradscf.pbc import dft as pdft
    from gradscf.pbc.response import project_q_potential
    cell,k=_cell(nk);model=_grid_functional(nk)
    mf=pdft.KRKS(cell,kpts=k,functional=model,params=.15,temperature=.12).run(**OPTIONS)
    assert mf.converged
    fractional=np.asarray(k)@np.asarray(cell.lattice).T/(2*np.pi)
    q=np.array([1/nk,0,0]);transfer=momentum_transfer(fractional,q)
    adjoint=lambda z:z[:,np.argsort(transfer.partners)].conj().swapaxes(-1,-2)
    potential=(1+.3j)*project_q_potential(mf.inputs,transfer,.03*jnp.ones(len(mf.inputs.weights)))
    if nk==2:potential=.5*(potential+adjoint(potential))
    positive=dfpt.DFPT(mf,q=q).response(potential=potential)
    negative=dfpt.DFPT(mf,q=-q).response(potential=adjoint(potential))
    assert positive.converged and negative.converged
    np.testing.assert_allclose(negative.density,adjoint(positive.density),atol=1e-9)
    # An exactly commensurate 2x FFT grid is even, excluded by Cell's +/-G
    # guard. The independent full-matrix energy Hessian above covers nk=2;
    # the odd nk=3 grid additionally admits this self-consistent SCF oracle.
    if nk==3:_check_supercell_scf(cell,k,model,transfer,potential,positive)
    assert dfpt.DFPT(mf).response(potential=jnp.broadcast_to(jnp.eye(2),mf.inputs.hcore.shape)).converged
    with pytest.raises(ValueError,match='Set the periodic Functional'):
        dfpt.DFPT(mf,q=q,functional=model,params=.15)
    from dataclasses import replace
    bare=pdft.KRKS(cell,kpts=k,functional=replace(model,q_energy_fn=None),params=.15,temperature=.12).run(**OPTIONS)
    assert bare.converged
    with pytest.raises(NotImplementedError,match='q_energy_fn'):
        dfpt.DFPT(bare,q=q)


def _check_supercell_scf(cell,k,model,transfer,potential,positive):
    from gradscf.pbc import gto,scf
    from gradscf.integrals.periodic.fft import build_inputs
    nk=len(k)
    translations=np.arange(nk)[:,None]*np.asarray(cell.lattice)[0]
    coords=np.asarray([[.4,.5,.6],[1.8,.8,.9]])
    supercell=gto.M(atom=[('H',r+t) for t in translations for r in coords],
        a=np.diag([nk,1,1])@np.asarray(cell.lattice),unit='Bohr',mesh=(9*nk,9,9))
    data=build_inputs(supercell)
    transform=jnp.kron(jnp.exp(1j*jnp.asarray(translations)@k.T)/jnp.sqrt(nk),jnp.eye(2))
    offset=jnp.zeros((2,2*nk,2*nk),dtype=complex)
    for source,target in enumerate(transfer.partners):
        offset=offset.at[:,2*target:2*target+2,2*source:2*source+2].set(potential[:,source])
    perturbation=transform@(offset+offset.conj().swapaxes(-1,-2))@transform.conj().T
    np.testing.assert_allclose(perturbation[0],perturbation[1],atol=1e-12)
    np.testing.assert_allclose(perturbation.imag,0.,atol=1e-12)
    def solve(scale):
        return scf.run_kpoint_scf(data._replace(hcore=data.hcore+scale*perturbation[0,None].real),
            mesh=supercell.mesh,nelec=supercell.nelec,functional=model,params=.15,temperature=.12,**OPTIONS)
    plus,minus=solve(1e-4),solve(-1e-4)
    assert plus.converged and minus.converged
    difference=transform.conj().T@((plus.density_spin[:,0]-minus.density_spin[:,0])/2e-4)@transform
    expected=jnp.stack([difference[:,2*t:2*t+2,2*s:2*s+2] for s,t in enumerate(transfer.partners)],axis=1)
    assert jnp.max(jnp.abs(expected))>1e-4
    print('external GGA nk',nk,'supercell response error',float(jnp.max(jnp.abs(positive.density-expected))))
    np.testing.assert_allclose(positive.density,expected,atol=2e-7,rtol=0)


def test_external_q_screened_response_parameter_gradient_relaxes_reference():
    from gradscf import dfpt
    from gradscf.pbc.scf import run_kpoint_scf
    from gradscf.integrals.periodic.fft import build_inputs
    from gradscf.scf import SCFDifferentiationConfig
    from gradscf.solvers import LinearSolverConfig
    cell,k=_cell(3);data=build_inputs(cell,kpts=k);model=_grid_functional(3)
    transfer=momentum_transfer(np.asarray(k)@np.asarray(cell.lattice).T/(2*np.pi),[1/3,0,0])
    potential=jnp.broadcast_to(jnp.array([[.1,.03+.02j],[.01-.04j,-.2]]),data.hcore.shape)
    def solve(p):
        return run_kpoint_scf(data,mesh=cell.mesh,nelec=cell.nelec,functional=model,params=p,
            temperature=.12,**OPTIONS,
            differentiation=SCFDifferentiationConfig(mode='implicit',tolerance=1e-11,max_iter=80))
    def response(p,result=None,method='gmres'):
        reference=solve(p) if result is None else result
        problem=dfpt.finite_q_problem(data,reference,transfer,mesh=cell.mesh,functional=model,params=p,temperature=.12)
        return problem.response(potential=potential,config=LinearSolverConfig(method=method,rtol=1e-11,atol=1e-12))
    result=solve(.15);assert result.converged
    iterative=response(.15,result);direct=response(.15,result,method='direct')
    assert iterative.converged and direct.converged
    np.testing.assert_allclose(iterative.density,direct.density,atol=1e-10)
    loss=lambda p:jnp.sum(jnp.abs(response(p).density)**2)
    derivative=jax.jit(jax.grad(loss))(.15)
    expected=(loss(.15001)-loss(.14999))/2e-5
    frozen=jax.grad(lambda p:jnp.sum(jnp.abs(response(p,result).density)**2))(.15)
    print('screened parameter gradient',float(derivative),'FD',float(expected),'frozen',float(frozen))
    assert abs(derivative)>1e-6 and abs(derivative-frozen)>1e-6
    np.testing.assert_allclose(derivative,expected,atol=1e-7,rtol=2e-5)
