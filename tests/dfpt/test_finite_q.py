"""Finite-q electronic response: momentum folding, kernels and supercell oracle."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.pbc import gto,scf
from gradscf.integrals.periodic.fft import build_inputs


def test_positive_q_mapping_keeps_reciprocal_shifts():
    from gradscf.pbc.momentum import momentum_transfer
    k=np.array([[0.,0.,0.],[1/3,0.,0.],[-1/3,0.,0.]])
    transfer=momentum_transfer(k,[1/3,0.,0.])
    np.testing.assert_array_equal(transfer.partners,[1,2,0])
    np.testing.assert_allclose(k[transfer.partners],k+transfer.q+transfer.shifts,atol=1e-14)
    assert np.any(transfer.shifts!=0)
    with pytest.raises(ValueError,match='not closed'):
        momentum_transfer(k,[.25,0.,0.])


@pytest.mark.parametrize('xc',['hf','pbe'])
def test_q_zero_kernel_matches_existing_density_jvp(xc,monkeypatch):
    from gradscf.pbc import _kpoint
    from gradscf.pbc.response import q_kernel
    from gradscf.pbc.momentum import momentum_transfer
    if xc=='pbe':
        from test_pbe import pbe_energy_density
        monkeypatch.setattr(_kpoint,'_point_unrestricted_xc_value_and_grad_kernel',
            lambda xc,kind:jax.vmap(jax.value_and_grad(lambda v:pbe_energy_density(v[:2],v[2:].reshape(2,3)))))
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .6 .8',a=np.eye(3)*6,unit='Bohr',mesh=(9,)*3)
    k=cell.make_kpts((3,1,1))
    inputs=build_inputs(cell,kpts=k)
    transfer=momentum_transfer(np.asarray(k)@np.asarray(cell.lattice).T/(2*np.pi),[0.,0.,0.])
    density=jnp.broadcast_to(jnp.eye(2)*.2,(2,3,2,2)).astype(complex)
    perturbation=jnp.broadcast_to(jnp.array([[.1,.03+.02j],[.03-.02j,-.07]]),density.shape)
    apply=q_kernel(inputs,density,transfer,mesh=cell.mesh,xc=xc)
    expected=jax.jvp(lambda d:_kpoint.energy_and_fock(d,inputs,mesh=cell.mesh,xc=xc)[1],
                     (density,),(perturbation,))[1]
    np.testing.assert_allclose(apply(perturbation),expected,atol=1e-10)


@pytest.mark.parametrize('moving_metric',[False,True])
@pytest.mark.parametrize('temperature',[None,.15])
@pytest.mark.parametrize('complex_orbitals',[False,True])
def test_independent_q_response_against_full_matrix_function(temperature,moving_metric,complex_orbitals):
    from types import SimpleNamespace
    from gradscf.dfpt.finite_q import _independent_response
    from gradscf.pbc.momentum import momentum_transfer
    from gradscf.pbc.occupations import fermi_occupations
    k=np.array([[0.,0.,0.],[1/3,0.,0.],[-1/3,0.,0.]])
    transfer=momentum_transfer(k,[1/3,0.,0.])
    eps=jnp.array([[[-1.,.7],[-.8,.9],[-1.,.7]]]*2)
    occ=jnp.broadcast_to(jnp.array([1.,0.]),eps.shape)
    mu=None
    if temperature:
        pop=fermi_occupations(eps,jnp.array([1.,1.]),temperature,jnp.ones(3)/3)
        occ,mu=pop.occupations,pop.chemical_potential
    coeff=jnp.broadcast_to(jnp.eye(2),(2,3,2,2))
    if complex_orbitals:
        rng=np.random.default_rng(42)
        coeff=jnp.asarray(np.linalg.qr(rng.normal(size=(2,3,2,2))+1j*rng.normal(size=(2,3,2,2)))[0])
    result=SimpleNamespace(mo_energy_spin=eps,mo_coeff_spin=coeff,
                           mo_occ_spin=occ,chemical_potential=mu)
    chi,valid=_independent_response(result,transfer,temperature)
    assert valid
    potential=jnp.arange(12).reshape(3,2,2)*.01+.013j
    overlap=potential*.2 if moving_metric else jnp.zeros_like(potential)
    response=chi(jnp.stack([potential,potential]),overlap)[0]
    compiled=jax.jit(chi)(jnp.stack([potential,potential]),overlap)[0]
    perturbation=jnp.zeros((6,6),dtype=complex)
    for source,target in enumerate(transfer.partners):
        perturbation=perturbation.at[2*target:2*target+2,2*source:2*source+2].set(potential[source])
    perturbation+=perturbation.conj().T
    def density(t):
        s=jnp.eye(6)+(.2*t*perturbation if moving_metric else 0.)
        x=jnp.linalg.inv(jnp.linalg.cholesky(s)).conj().T
        frame=jax.scipy.linalg.block_diag(*coeff[0])
        h=(frame*eps[0].ravel())@frame.conj().T
        e,c=jnp.linalg.eigh(x.conj().T@(h+t*perturbation)@x)
        c=x@c
        f=(jnp.arange(6)<3) if temperature is None else jax.nn.sigmoid((mu[0]-e)/temperature)
        return (c*f)@c.conj().T
    fd=(density(1e-5)-density(-1e-5))/2e-5
    expected=jnp.stack([fd[2*t:2*t+2,2*s:2*s+2] for s,t in enumerate(transfer.partners)])
    np.testing.assert_allclose(response,expected,atol=1e-9)
    np.testing.assert_allclose(compiled,expected,atol=1e-9)


@pytest.mark.parametrize('mu',[jnp.nan,jnp.inf,-jnp.inf,1.])
def test_inconsistent_thermal_reference_is_rejected(mu):
    from types import SimpleNamespace
    from gradscf.dfpt.finite_q import _independent_response
    from gradscf.pbc.momentum import momentum_transfer
    transfer=momentum_transfer(np.array([[0.,0,0],[1/3,0,0],[-1/3,0,0]]),[1/3,0,0])
    result=SimpleNamespace(mo_energy_spin=jnp.zeros((2,3,2)),
        mo_coeff_spin=jnp.broadcast_to(jnp.eye(2),(2,3,2,2)),
        mo_occ_spin=jnp.full((2,3,2),.5),chemical_potential=jnp.full(2,mu))
    _,valid=_independent_response(result,transfer,.1)
    assert not valid


@pytest.mark.parametrize('xc,temperature,handwritten',[
    ('hf',None,False),('hf',.15,False),('pbe',None,True),('pbe',.15,True),
    pytest.param('pbe',None,False,id='pbe-runtime-zero'),
    pytest.param('pbe',.15,False,id='pbe-runtime-thermal'),
    pytest.param('r2scan',.15,False,id='r2scan-runtime-thermal'),
])
def test_q_response_matches_supercell_finite_difference(xc,temperature,handwritten,monkeypatch):
    from gradscf import dfpt
    from gradscf.pbc.momentum import momentum_transfer
    from gradscf.pbc.response import project_q_potential
    from gradscf.pbc import dft as pdft,_kpoint
    if xc=='pbe' and not handwritten:
        pytest.importorskip('jax_xc',reason='Production XC backend required')
    if handwritten:
        from test_pbe import pbe_energy_density
        monkeypatch.setattr(_kpoint,'_point_unrestricted_xc_value_and_grad_kernel',
            lambda xc,kind:jax.vmap(jax.value_and_grad(lambda v:pbe_energy_density(v[:2],v[2:].reshape(2,3)))))
    mesh=(15,)*3
    lattice=np.eye(3)*6
    coords=np.array([[.4,.5,.6],[1.8,.8,.9]])
    cell=gto.M(atom=[('H',r) for r in coords],a=lattice,unit='Bohr',mesh=mesh)
    k=cell.make_kpts((3,1,1))
    options=dict(conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10,max_cycle=150,damping=.1)
    cls=scf.KRHF if xc=='hf' else pdft.KRKS
    mf=cls(cell,kpts=k,xc=xc,temperature=temperature).run(**options)
    transfer=momentum_transfer(np.asarray(k)@lattice.T/(2*np.pi),[1/3,0,0])
    potential=project_q_potential(mf.inputs,transfer,jnp.ones(len(mf.inputs.weights))*.03)[0]
    response=dfpt.DFPT(mf,q=[1/3,0,0]).response(potential=potential)
    assert response.converged
    translations=np.arange(3)[:,None]*lattice[0]
    atoms=[('H',r+t) for t in translations for r in coords]
    supercell=gto.M(atom=atoms,a=np.diag([3,1,1])@lattice,unit='Bohr',mesh=(45,15,15))
    data=build_inputs(supercell)
    transform=jnp.kron(jnp.exp(1j*jnp.asarray(translations)@k.T)/jnp.sqrt(3.),jnp.eye(2))
    v=jnp.zeros((6,6),dtype=complex)
    for source,target in enumerate(transfer.partners):
        v=v.at[2*target:2*target+2,2*source:2*source+2].set(potential[source])
    v=transform@(v+v.conj().T)@transform.conj().T
    # The real supercell perturbation contains both +q and -q channels.
    np.testing.assert_allclose(v.imag,0.,atol=1e-12)
    def solve(t):
        return scf.run_kpoint_scf(data._replace(hcore=data.hcore+t*v.real[None]),
            mesh=supercell.mesh,nelec=supercell.nelec,xc=xc,temperature=temperature,**options)
    step=1e-4
    plus,minus=solve(step),solve(-step)
    assert plus.converged and minus.converged
    difference=(plus.density_spin[:,0]-minus.density_spin[:,0])/(2*step)
    difference=transform.conj().T@difference@transform
    expected=jnp.stack([difference[:,2*t:2*t+2,2*s:2*s+2] for s,t in enumerate(transfer.partners)],axis=1)
    assert jnp.max(jnp.abs(expected))>1e-4
    print(xc,temperature,'max density response error:',float(jnp.max(jnp.abs(response.density-expected))))
    np.testing.assert_allclose(response.density,expected,atol=1e-7,rtol=0)
    with pytest.raises(NotImplementedError,match='Taylor'):
        dfpt.DFPT(mf,q=[1/3,0,0]).series(potential=potential)


@pytest.mark.parametrize('nk',[2,3])
def test_shifted_mesh_q_conjugacy_and_self_inverse_channel(nk):
    from gradscf import dfpt
    from gradscf.pbc.momentum import momentum_transfer
    from gradscf.pbc.response import project_q_potential
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .6 .8',a=np.eye(3)*6,unit='Bohr',mesh=(15,)*3)
    k=cell.make_kpts((nk,1,1),shift=(.5,0,0))
    mf=scf.KRHF(cell,kpts=k,temperature=.1).run(conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10)
    q=np.array([1/nk,0,0]);transfer=momentum_transfer(np.asarray(k)@np.asarray(cell.lattice).T/(2*np.pi),q)
    inverse=np.argsort(transfer.partners)
    adjoint=lambda d:d[:,inverse].conj().swapaxes(-1,-2)
    v=(1.+.3j)*project_q_potential(mf.inputs,transfer,.03*jnp.ones(len(mf.inputs.weights)))
    if nk==2:v=.5*(v+adjoint(v))
    positive=dfpt.DFPT(mf,q=q).response(potential=v)
    negative=dfpt.DFPT(mf,q=-q).response(potential=adjoint(v))
    assert positive.converged and negative.converged
    np.testing.assert_allclose(negative.density,adjoint(positive.density),atol=1e-7,rtol=0)
    if nk==2:np.testing.assert_allclose(positive.density,adjoint(positive.density),atol=1e-7,rtol=0)


@pytest.mark.parametrize('xc',['hf',pytest.param('pbe',id='pbe-runtime-grad'),'r2scan'])
def test_q_response_loss_gradient_through_relaxed_reference(xc):
    from gradscf import dfpt
    from gradscf.pbc.momentum import momentum_transfer
    from gradscf.scf import SCFDifferentiationConfig
    from gradscf.solvers import LinearSolverConfig
    if xc=='pbe':
        pytest.importorskip('jax_xc',reason='Production XC response derivatives required')
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .6 .8',a=np.eye(3)*6,unit='Bohr',mesh=(9,)*3)
    k=cell.make_kpts((3,1,1));data=build_inputs(cell,kpts=k)
    transfer=momentum_transfer(np.asarray(k)@np.asarray(cell.lattice).T/(2*np.pi),[1/3,0,0])
    perturbation=jnp.broadcast_to(jnp.array([[.1,.03],[.03,-.2]]),data.hcore.shape)
    def response(scale,method='gmres'):
        inputs=data._replace(hcore=data.hcore+scale*perturbation)
        result=scf.run_kpoint_scf(inputs,mesh=cell.mesh,nelec=cell.nelec,xc=xc,temperature=.12,
            conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10,
            differentiation=SCFDifferentiationConfig(mode='implicit',tolerance=1e-11,max_iter=80))
        problem=dfpt.finite_q_problem(inputs,result,transfer,mesh=cell.mesh,xc=xc,temperature=.12)
        return problem.response(potential=perturbation,config=LinearSolverConfig(method=method,rtol=1e-11,atol=1e-12))
    iterative=response(.2);direct=response(.2,'direct')
    assert iterative.converged and direct.converged
    np.testing.assert_allclose(iterative.density,direct.density,atol=1e-10)
    loss=lambda scale:jnp.sum(jnp.abs(response(scale).density)**2)
    derivative=jax.jit(jax.grad(loss))(.2)
    fd=(loss(.20001)-loss(.19999))/2e-5
    assert abs(derivative)>1e-6
    print(xc,'response-loss gradient:',float(derivative),'finite difference:',float(fd))
    np.testing.assert_allclose(derivative,fd,atol=1e-7,rtol=2e-5)


def test_unrestricted_q_response_with_a_filled_spin_channel():
    from gradscf import dfpt
    cell=gto.M(atom='H .2 .3 .4; He 2. .6 .8',a=np.eye(3)*6,unit='Bohr',mesh=(15,)*3,spin=1)
    mf=scf.KUHF(cell,kpts=cell.make_kpts((3,1,1)),temperature=.1).run(
        conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10,damping=.1,max_cycle=150)
    assert mf.converged and jnp.isposinf(mf.chemical_potential[0])
    potential=jnp.broadcast_to(jnp.array([[.1,.03],[.03,-.2]]),mf.inputs.hcore.shape)
    response=dfpt.DFPT(mf,q=[1/3,0,0]).response(potential=potential)
    assert response.converged
    np.testing.assert_allclose(response.density[0],0.,atol=1e-10)
    assert jnp.max(jnp.abs(response.density[1]))>1e-4
