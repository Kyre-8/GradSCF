"""Periodic XC contracts: screened Fourier operators and full AO response."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_screened_exchange_fourier_limits_and_omega_derivative():
    from gradscf.integrals.periodic.coulomb import exchange_kernel
    lattice=jnp.eye(3)*6.
    wave=jnp.array([[0.,0.,0.],[1e-8,0.,0.],[.4,.2,-.1],[2.,-.5,.3]])
    def kernel(omega,part):
        return exchange_kernel(wave,lattice=lattice,nk=1,exxdiv=None,
                               omega=omega,range_part=part)
    short=kernel(.4,'short');long=kernel(.4,'long')
    full=exchange_kernel(wave,lattice=lattice,nk=1,exxdiv=None)
    np.testing.assert_allclose(short[0],np.pi/.4**2,atol=1e-13)
    np.testing.assert_allclose(long[0],-np.pi/.4**2,atol=1e-13)
    np.testing.assert_allclose(short+long,full,atol=2e-13,rtol=2e-15)
    np.testing.assert_allclose(short[1],short[0],atol=1e-12)
    derivative=jax.jacfwd(lambda w:kernel(w,'short'))(.4)
    finite=(kernel(.400001,'short')-kernel(.399999,'short'))/2e-6
    np.testing.assert_allclose(derivative,finite,atol=2e-8,rtol=2e-9)


@pytest.mark.parametrize('part',['short','long'])
@pytest.mark.parametrize('boundary,exxdiv',[(SimpleNamespace(dimension=2),None),(None,'vcut_sph')])
def test_screened_exchange_rejects_undefined_boundary(part,boundary,exxdiv):
    from gradscf.integrals.periodic.coulomb import exchange_kernel
    with pytest.raises(NotImplementedError,match='(3D|vcut_sph|screened)'):
        exchange_kernel(jnp.zeros((1,3)),lattice=jnp.eye(3)*6.,nk=1,
                        exxdiv=exxdiv,boundary=boundary,omega=.4,range_part=part)


@pytest.mark.parametrize('nk',[1,2])
@pytest.mark.parametrize('exxdiv',[None,'ewald'])
def test_screened_exchange_ao_decomposition_and_density_derivative(nk,exxdiv):
    from gradscf.pbc import gto
    from gradscf.integrals.periodic.fft import build_inputs,get_kpoint_jk
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,
               unit='Bohr',mesh=(9,)*3)
    inputs=build_inputs(cell,kpts=cell.make_kpts((nk,1,1)))
    density=jnp.broadcast_to(jnp.array([[.35,.04],[.04,.2]]),(2,nk,2,2))
    tangent=jnp.broadcast_to(jnp.array([[.1,.02+.03j],[.02-.03j,-.07]]),density.shape)
    def exchange(d,part='full'):
        return get_kpoint_jk(inputs,d,mesh=cell.mesh,exxdiv=exxdiv,
                            omega=None if part=='full' else .4,range_part=part)[1]
    np.testing.assert_allclose(exchange(density,'short')+exchange(density,'long'),
                               exchange(density),atol=3e-13,rtol=3e-13)
    action=jax.jvp(lambda d:exchange(d,'short'),(density.astype(complex),),(tangent,))[1]
    np.testing.assert_allclose(action,exchange(tangent,'short'),atol=2e-13,rtol=2e-13)


def test_rsh_fock_uses_both_physical_exchange_ranges_and_kweights():
    from gradscf.pbc import gto,_kpoint
    from gradscf.integrals.periodic.fft import build_inputs,get_kpoint_jk
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,
               unit='Bohr',mesh=(9,)*3)
    inputs=build_inputs(cell,kpts=cell.make_kpts((2,1,1)),kweights=[.3,.7])
    density=jnp.broadcast_to(jnp.array([[.35,.04],[.04,.2]]),(2,2,2,2)).astype(complex)
    direction=jnp.broadcast_to(jnp.array([[.1,.02+.03j],[.02-.03j,-.07]]),density.shape)
    j,short=get_kpoint_jk(inputs,density,mesh=cell.mesh,exxdiv=None,omega=.4,range_part='short')
    _,long=get_kpoint_jk(inputs,density,mesh=cell.mesh,exxdiv=None,omega=.4,range_part='long')
    exchange=.2*short+.8*long
    expected_fock=inputs.hcore+j-exchange
    expected_energy=jnp.einsum('k,skpq,kqp->',inputs.kweights,density,inputs.hcore+.5*j).real
    expected_energy-=.5*jnp.einsum('k,skpq,skqp->',inputs.kweights,density,exchange).real
    expected_energy+=inputs.nuclear_repulsion
    energy,fock=_kpoint.energy_and_fock(density,inputs,mesh=cell.mesh,
                                      xc='RSH(.4,.8,-.6)',exxdiv=None)
    np.testing.assert_allclose(energy,expected_energy,atol=2e-13)
    np.testing.assert_allclose(fock,expected_fock,atol=2e-13)
    derivative=jax.grad(lambda t:_kpoint.energy_and_fock(density+t*direction,inputs,
        mesh=cell.mesh,xc='RSH(.4,.8,-.6)',exxdiv=None)[0])(0.)
    np.testing.assert_allclose(derivative,jnp.einsum('k,skpq,skqp->',
                                inputs.kweights,direction,fock).real,atol=2e-12)


@pytest.mark.parametrize('omega',[.11,.4])
@pytest.mark.parametrize('part',['short','long'])
def test_screened_ewald_matches_independent_pyscf_probe_charge(omega,part):
    pytest.importorskip('pyscf')
    from pyscf.pbc import gto as pgto,tools
    from gradscf.pbc import gto
    from gradscf.integrals.periodic.fft import build_inputs
    from gradscf.integrals.periodic.coulomb import exchange_kernel,screened_exchange_correction
    options=dict(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,
                 unit='Bohr',basis='gth-szv',pseudo='gth-pade',mesh=(9,)*3,precision=1e-10)
    cell=gto.M(**options);points=cell.make_kpts((2,1,1));inputs=build_inputs(cell,kpts=points)
    reference=pgto.M(**options,cart=True,verbose=0)
    signed=-omega if part=='short' else omega
    probe=SimpleNamespace(kpts=np.asarray(points),exxdiv='ewald')
    with reference.with_range_coulomb(signed):
        expected=tools.get_coulG(reference,exx='ewald',mf=probe,
            Gv=np.asarray(inputs.gvectors),wrap_around=False)
    actual=exchange_kernel(inputs.gvectors,lattice=inputs.lattice,nk=2,omega=omega,
                           range_part=part,exxdiv='ewald')
    correction=screened_exchange_correction(inputs.lattice,inputs.madelung,
                                            inputs.exchange_context,omega,part)
    actual=actual.at[0].add(2*jnp.linalg.det(inputs.lattice)*correction)
    np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=2e-10)
    # The cutoff topology is fixed while both screening and live lattice vary.
    value=lambda w,s:screened_exchange_correction(inputs.lattice*s,inputs.madelung,
        inputs.exchange_context,w,part,reference_omega=omega)
    for index in (0,1):
        derivative=jax.grad(value,argnums=index)(omega,1.)
        step=1e-6
        plus=value(omega+step,1.) if index==0 else value(omega,1.+step)
        minus=value(omega-step,1.) if index==0 else value(omega,1.-step)
        np.testing.assert_allclose(derivative,(plus-minus)/(2*step),atol=2e-8,rtol=2e-7)


@pytest.mark.parametrize('xc',['lda','pbe','blyp','pbe0','b3lyp','scan','r2scan','tpss','hse06','cam-b3lyp'])
def test_complete_periodic_recipes_energy_fock_and_q_zero_hessian(xc):
    pytest.importorskip('jax_xc')
    from gradscf.pbc import gto,_kpoint
    from gradscf.pbc.momentum import momentum_transfer
    from gradscf.pbc.response import q_kernel
    from gradscf.integrals.periodic.fft import build_inputs
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,unit='Bohr',mesh=(7,)*3)
    points=cell.make_kpts((2,1,1))
    inputs=build_inputs(cell,kpts=points,kweights=[.3,.7])
    density=jnp.broadcast_to(jnp.array([[.35,.04],[.04,.2]]),(2,2,2,2)).astype(complex)
    direction=jnp.broadcast_to(jnp.array([[.1,.02+.03j],[.02-.03j,-.07]]),density.shape)
    energy=lambda d:_kpoint.energy_and_fock(d,inputs,mesh=cell.mesh,xc=xc,exxdiv=None)[0]
    fock=lambda d:_kpoint.energy_and_fock(d,inputs,mesh=cell.mesh,xc=xc,exxdiv=None)[1]
    potential=fock(density)
    expected=jnp.einsum('k,skpq,skqp->',inputs.kweights,direction,potential).real
    derivative=jax.grad(lambda t:energy(density+t*direction))(0.)
    finite=(energy(density+1e-5*direction)-energy(density-1e-5*direction))/2e-5
    assert jnp.all(jnp.isfinite(potential))
    np.testing.assert_allclose(derivative,expected,atol=2e-10,rtol=2e-10)
    np.testing.assert_allclose(derivative,finite,atol=2e-8,rtol=2e-7)
    transfer=momentum_transfer(np.asarray(points)@np.asarray(cell.lattice).T/(2*np.pi),np.zeros(3))
    action=q_kernel(inputs,density,transfer,mesh=cell.mesh,xc=xc,exxdiv=None)
    tangent=direction.at[:,0,0,1].add(.04+.03j)
    adjoint=tangent.conj().swapaxes(-1,-2)
    expected=.5*(jax.jvp(fock,(density,),(tangent+adjoint,))[1]
                    -1j*jax.jvp(fock,(density,),(1j*(tangent-adjoint),))[1])
    np.testing.assert_allclose(action(tangent),expected,atol=2e-9,rtol=2e-9)
    np.testing.assert_allclose(action(1j*tangent),1j*action(tangent),atol=2e-11,rtol=2e-11)


def _external_reference():
    from gradscf.dft import Functional
    from gradscf.pbc.scf import _params_signature
    # A deliberately nonlocal AO callback couples spins, orbitals and k points;
    # no pointwise grid Hessian can stand in for its full energy Hessian.
    ao=jnp.array([[[[.7,.2],[.3,.6],[.4,-.1]],
                   [[.1,.2],[.3,.1],[-.2,.1]],
                   [[.2,-.1],[.1,.2],[.1,.3]],
                   [[.1,.1],[-.1,.2],[.2,-.1]]]],dtype=complex)
    inputs=SimpleNamespace(ao=ao,weights=jnp.ones(3)*2.,kweights=jnp.ones(1),
        kpoints=jnp.zeros((1,3)),overlap=jnp.eye(2)[None],hcore=jnp.diag(jnp.array([-.4,.6]))[None],
        gvectors=jnp.array([[0.,0.,0.],[1.,0.,0.],[-1.,0.,0.]]),boundary=None,
        lattice=jnp.eye(3)*6.,madelung=jnp.array(.47),nuclear_repulsion=jnp.array(0.))
    matrix=jnp.array([[.3,.7],[.7,-.2]])
    model=Functional(lambda x:x.density_matrix,
        lambda p,d:p*jnp.einsum('skpq,qp->',d,matrix).real**4/4)
    density=jnp.broadcast_to(jnp.diag(jnp.array([1.,0.])),(2,1,2,2)).astype(complex)
    result=SimpleNamespace(density_spin=density,mo_coeff_spin=jnp.broadcast_to(jnp.eye(2),density.shape),
                           mo_occ_spin=jnp.broadcast_to(jnp.array([1.,0.]),(2,1,2)))
    cell=SimpleNamespace(mesh=(3,1,1),dimension=3,_version=0,nelec=(1,1))
    mf=SimpleNamespace(cell=cell,inputs=inputs,result=result,xc=None,exxdiv='ewald',params=.23,
        functional=model,converged=True,unrestricted=False,kpts=inputs.kpoints,
        _computed_xc=None,_computed_mesh=cell.mesh,_cell_version=0,
        _computed_functional=model,_computed_params=_params_signature(.23))
    return mf,matrix


def test_q_zero_external_td_uses_full_ao_energy_hessian():
    from gradscf.pbc import tdscf
    from gradscf.pbc._kpoint import energy_and_fock
    from gradscf.integrals.periodic.fft import get_kpoint_jk
    mf,matrix=_external_reference()
    a,b,gap=tdscf.response_operators(mf)
    density=mf.result.density_spin
    f0=energy_and_fock(density,mf.inputs,mesh=mf.cell.mesh,exxdiv=None,
                       functional=mf.functional,params=mf.params)[1]
    expected_gap=(f0[0,0,1,1]-f0[0,0,0,0]).real
    q=jnp.zeros_like(density).at[:,0,1,0].set(1.)
    # The callback's density Hessian is analytic: p*3*Tr(D M)^2 M Tr(q M).
    trace=jnp.einsum('skpq,qp->',density,matrix).real
    external=mf.params*3*trace**2*matrix*jnp.einsum('skpq,qp->',q,matrix)
    bare=lambda d:jnp.broadcast_to(get_kpoint_jk(mf.inputs,d,mesh=mf.cell.mesh,
                                                exxdiv=None,with_k=False)[0],density.shape)
    hermitian=q+q.conj().swapaxes(-1,-2)
    imaginary=1j*(q-q.conj().swapaxes(-1,-2))
    bare_response=.5*(jax.jvp(bare,(density,),(hermitian,))[1]
                        -1j*jax.jvp(bare,(density,),(imaginary,))[1])
    expected=(bare_response[0,0]+external)[1,0]
    np.testing.assert_allclose(gap,jnp.array([expected_gap]),atol=1e-13)
    np.testing.assert_allclose(a(jnp.ones(1)),jnp.array([expected_gap+expected]),atol=2e-13)
    np.testing.assert_allclose(b(jnp.ones(1)),jnp.array([expected]),atol=2e-13)
