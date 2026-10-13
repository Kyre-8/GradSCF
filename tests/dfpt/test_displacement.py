"""Primitive-cell geometry sources against independent supercell AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.pbc import gto,scf


@pytest.mark.parametrize('xc',['hf','pbe','pbe-runtime'])
def test_primitive_displacement_matches_supercell(xc,monkeypatch):
    from gradscf.dfpt import displacement_source
    from gradscf.pbc import _kpoint,dft
    from gradscf.pbc.phonon import Phonons
    if xc=='pbe-runtime':
        pytest.importorskip('jax_xc',reason='Production XC backend required')
        xc='pbe'
    elif xc=='pbe':
        from test_pbe import pbe_energy_density
        monkeypatch.setattr(_kpoint,'_point_unrestricted_xc_value_and_grad_kernel',
            lambda xc,kind:jax.vmap(jax.value_and_grad(lambda v:pbe_energy_density(v[:2],v[2:].reshape(2,3)))))
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6,unit='Bohr',mesh=(15,)*3)
    mf=dft.KRKS(cell,kpts=cell.make_kpts((3,1,1)),xc=xc,temperature=.1).run(
        conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10,damping=.1)
    assert mf.converged
    direction=jnp.array([[.1,-.2,.3],[1.,.2,-.1]])
    q=np.array([1/3,0.,0.])
    expected=Phonons(mf,masses=[1.,1.]).displacement(q,direction,method='supercell')
    actual=displacement_source(mf,q,direction)
    for name,a,b in [('hcore',actual.hcore,expected.hcore),
                     ('overlap',actual.perturbation.overlap,expected.perturbation.overlap),
                     ('rho',actual.perturbation.rho_spin,expected.perturbation.rho_spin),
                     ('fock',actual.perturbation.fock,expected.perturbation.fock)]:
        print(name,float(jnp.max(jnp.abs(a-b))))
        np.testing.assert_allclose(a,b,atol=2e-6,rtol=0,err_msg=name)
    if xc=='hf':
        shifted_q=q+np.array([1.,0.,0.])
        shifted=Phonons(mf,masses=[1.,1.]).displacement(shifted_q,direction,method='supercell')
        primitive=displacement_source(mf,shifted_q,direction)
        phase=jnp.exp(-1j*mf.inputs.coords@cell.reciprocal_vectors()[0])
        np.testing.assert_allclose(shifted.perturbation.rho_spin,expected.perturbation.rho_spin*phase,atol=1e-12)
        np.testing.assert_allclose(primitive.perturbation.rho_spin,shifted.perturbation.rho_spin,atol=2e-6)


def test_nonlocal_primitive_source_matches_supercell_ad():
    from gradscf.dfpt.displacement import _nonlocal_source
    from gradscf.pbc.phonon import _replica
    from gradscf.pbc.momentum import momentum_transfer
    from gradscf.integrals.periodic.fft import build_inputs
    from gradscf.integrals.periodic.ao import fourier_ao,reciprocal_grid
    from gradscf.integrals.periodic.pseudo import nonlocal_matrix
    cell=gto.M(atom='C .4 .5 .6; H 1.8 .8 .9',spin=1,a=np.eye(3)*6,unit='Bohr',mesh=(21,)*3)
    k=cell.make_kpts((3,1,1));data=build_inputs(cell,kpts=k)
    transfer=momentum_transfer(np.asarray(k)@np.asarray(cell.lattice).T/(2*np.pi),[1/3,0,0])
    direction=jnp.array([[.3,-.2,.1],[-.4,.1,.2]])
    def coefficients(r):
        return jnp.stack([fourier_ao(cell.topology,cell.parameters_at(r),data.gvectors+ki) for ki in k])
    ft,dft=jax.jvp(coefficients,(cell.coords,),(direction,))
    actual=_nonlocal_source(cell,data,ft,dft,direction,transfer.partners)
    sup,indices,translations=_replica(cell,np.array([3,1,1]))
    vectors=reciprocal_grid(sup.lattice,sup.mesh)
    def matrix(r):
        params=sup.parameters_at(r)
        return nonlocal_matrix(vectors,fourier_ao(sup.topology,params,vectors),r,sup.pseudopotentials,sup.volume)
    phase=jnp.exp(2j*jnp.pi*indices[:,0]/3)
    pattern=(phase[:,None,None]*direction[None]).reshape(-1,3)
    full=jax.jvp(matrix,(sup.coords,),(pattern.real,))[1]+1j*jax.jvp(matrix,(sup.coords,),(pattern.imag,))[1]
    n=cell.topology.nao
    b=jnp.kron(jnp.exp(1j*translations@k.T)/jnp.sqrt(3.),jnp.eye(n))
    full=b.conj().T@full@b
    expected=jnp.stack([full[n*p:n*(p+1),n*s:n*(s+1)] for s,p in enumerate(transfer.partners)])
    print('nonlocal source error',float(jnp.max(jnp.abs(actual-expected))))
    np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=0)


def test_primitive_source_shifted_even_mesh_adjoint_and_jvp(monkeypatch):
    from gradscf.dfpt import displacement_source
    from gradscf.pbc import phonon
    from gradscf.pbc.momentum import momentum_transfer
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6,unit='Bohr',mesh=(41,)*3)
    mf=scf.KRHF(cell,kpts=cell.make_kpts((2,1,1),shift=(.5,0,0)),temperature=.1).run(
        conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10)
    def forbidden(*args,**kwargs):raise AssertionError('Primitive source must not build a supercell')
    monkeypatch.setattr(phonon,'_replica',forbidden)
    direction=jnp.array([[.1,-.2,.3],[1.,.2,-.1]])
    source=displacement_source(mf,[-.5,0,0],direction)
    transfer=momentum_transfer(np.asarray(mf.kpts)@np.asarray(cell.lattice).T/(2*np.pi),[-.5,0,0])
    p=transfer.partners
    np.testing.assert_allclose(source.hcore,source.hcore[p].conj().swapaxes(-1,-2),atol=1e-8)
    np.testing.assert_allclose(source.perturbation.fock,source.perturbation.fock[:,p].conj().swapaxes(-1,-2),atol=1e-7)
    linear=jax.jit(jax.jvp,static_argnums=0)(lambda u:displacement_source(mf,[-.5,0,0],u),
                                          (direction,),(direction*.2,))[1]
    np.testing.assert_allclose(linear.perturbation.fock,.2*source.perturbation.fock,atol=1e-10)


def test_displacement_source_input_validation_and_integer_direction():
    from gradscf.dfpt import displacement_source
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6,unit='Bohr',mesh=(7,)*3)
    mf=scf.KRHF(cell,kpts=cell.make_kpts((3,1,1))).run()
    for q in (.3,[.2,.1],[np.nan,0.,0.]):
        with pytest.raises(ValueError,match='three-vector'):
            displacement_source(mf,q,[[1,0,0],[0,0,0]])
    with pytest.raises(ValueError,match='finite'):
        displacement_source(mf,[1/3,0,0],[[np.nan,0,0],[0,0,0]])
    integer=displacement_source(mf,[1/3,0,0],[[1,0,0],[0,0,0]])
    floats=displacement_source(mf,[1/3,0,0],jnp.array([[1.,0,0],[0,0,0]]))
    np.testing.assert_allclose(integer.hcore,floats.hcore,atol=1e-14)
    # Keep the caller's q representative: DFPT uses it for density envelopes.
    lower=displacement_source(mf,[-1/3,0,0],[[1,0,0],[0,0,0]])
    upper=displacement_source(mf,[2/3,0,0],[[1,0,0],[0,0,0]])
    phase=jnp.exp(-1j*mf.inputs.coords@cell.reciprocal_vectors()[0])
    np.testing.assert_allclose(upper.perturbation.rho_spin,lower.perturbation.rho_spin*phase,atol=1e-12)


@pytest.mark.parametrize('xc,temperature',[('hf',None),('hf',.1),('pbe',.1),('pbe-runtime',.1)])
def test_primitive_electronic_response_force_constants_match_full_supercell(monkeypatch,xc,temperature):
    import jax
    from gradscf.pbc import phonon,dft,_kpoint
    if xc=='pbe-runtime':
        pytest.importorskip('jax_xc',reason='Production XC backend required')
        xc='pbe'
    elif xc=='pbe':
        from test_pbe import pbe_energy_density
        monkeypatch.setattr(_kpoint,'_point_unrestricted_xc_value_and_grad_kernel',
            lambda xc,kind:jax.vmap(jax.value_and_grad(lambda v:pbe_energy_density(v[:2],v[2:].reshape(2,3)))))
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6,unit='Bohr',mesh=(15,)*3)
    mf=dft.KRKS(cell,kpts=cell.make_kpts((3,1,1)),xc=xc,temperature=temperature).run(
        conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10)
    driver=phonon.Phonons(mf,masses=[1.,2.])
    q=[1/3,0.,0.]
    expected=driver.kernel(q,method='supercell')
    def forbidden(*args,**kwargs):raise AssertionError('Primitive response must not build supercell atomic_problem')
    monkeypatch.setattr(phonon,'atomic_problem',forbidden)
    actual=driver.kernel(q,method='primitive')
    assert actual.converged
    print(xc,temperature,'maximum force-constant error:',float(jnp.max(jnp.abs(actual.force_constants-expected.force_constants))))
    np.testing.assert_allclose(actual.force_constants,expected.force_constants,atol=2e-6,rtol=0)
    np.testing.assert_allclose(actual.frequencies,expected.frequencies,atol=2e-7,rtol=0)
