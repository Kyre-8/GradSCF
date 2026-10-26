"""Bounded commensurate phonon reference path with analytic response sources."""
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.pbc import gto,scf


def test_gamma_phonon_mass_scaling_and_optional_asr():
    from gradscf.pbc.phonon import Phonons
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6,unit='Bohr',mesh=(7,)*3)
    mf=scf.RHF(cell).run(conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10)
    phonons=Phonons(mf,masses=[1.,1.])
    result=phonons.kernel(asr=True)
    assert result.converged and result.frequencies.shape==(6,)
    translations=jnp.tile(jnp.eye(3),(2,1))
    np.testing.assert_allclose(result.dynamical_matrix@translations,0.,atol=1e-12)
    assert result.asr_residual>=0
    # Reusing force constants is enough to test mass conversion, with no SCF.
    from gradscf.pbc.phonon import _modes
    _,frequencies,_=_modes(result.force_constants,jnp.array([4.,4.]),True)
    np.testing.assert_allclose(frequencies,result.frequencies/2,atol=1e-10)
    matrix,_,_=_modes(result.force_constants,jnp.array([1.,2.]),True)
    np.testing.assert_allclose(matrix@(jnp.repeat(jnp.sqrt(jnp.array([1.,2.])),3)[:,None]*translations),0.,atol=1e-12)


@pytest.mark.parametrize('temperature',[None,.1])
def test_finite_q_displacement_sources_and_force_constant(temperature):
    from gradscf import dfpt
    from gradscf.pbc.phonon import Phonons
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6,unit='Bohr',mesh=(15,)*3)
    mf=scf.KRHF(cell,kpts=cell.make_kpts((3,1,1)),temperature=temperature).run(
        conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10,damping=.1)
    driver=Phonons(mf,masses=[1.,1.])
    q=np.array([1/3,0,0]);direction=jnp.zeros_like(cell.coords).at[1,0].set(1.)
    source=driver.displacement(q,direction)
    response=dfpt.DFPT(mf,q=q).response(tangent=source.perturbation)
    assert response.converged
    assert jnp.linalg.norm(source.hcore)>1e-3
    assert jnp.linalg.norm(source.perturbation.overlap)>1e-3
    result=driver.kernel(q)
    assert result.converged and result.hermiticity_error<1e-6
    # Independent finite displacement uses re-converged supercell energies.
    sup=driver.super_reference.cell
    translations=np.arange(3)[:,None]*np.asarray(cell.lattice)[0]
    phase=np.exp(2j*np.pi*np.arange(3)/3)
    pattern=phase[:,None,None]*np.asarray(direction)[None]
    def solve(t,part):
        r=np.asarray(sup.coords)+t*part.reshape(-1,3)
        moved=gto.M(atom=[('H',row) for row in r],a=sup.lattice,unit='Bohr',mesh=sup.mesh)
        calc=scf.RHF(moved,temperature=temperature).run(conv_tol=1e-13,conv_tol_density=1e-11,conv_tol_grad=1e-10,damping=.1)
        assert calc.converged
        return calc
    step=3e-4;base=solve(0.,pattern.real).e_free
    pairs=[(solve(step,p),solve(-step,p)) for p in (pattern.real,pattern.imag)]
    curvature=sum((a.e_free+b.e_free-2*base)/step**2 for a,b in pairs)/3
    np.testing.assert_allclose(result.force_constants[3,3].real,curvature,atol=3e-5,rtol=0)
    b=jnp.kron(jnp.exp(1j*jnp.asarray(translations)@mf.kpts.T)/jnp.sqrt(3.),jnp.eye(2))
    full=sum(factor*(a.result.density_spin[:,0]-c.result.density_spin[:,0])/2/step
             for factor,(a,c) in zip((1.,1j),pairs))
    full=b.conj().T@full@b
    expected=jnp.stack([full[:,2*p:2*p+2,2*k:2*k+2] for k,p in enumerate([1,2,0])],axis=1)
    np.testing.assert_allclose(response.density,expected,atol=5e-6,rtol=0)
    from gradscf.pbc.functional import density_inputs
    rho=sum(factor*(density_inputs(a.inputs,a.result.density_spin).rho_spin
                   -density_inputs(c.inputs,c.result.density_spin).rho_spin)/2/step
            for factor,(a,c) in zip((1.,1j),pairs))
    blocks=rho.reshape(2,3,15,15,15).reshape(2,3,-1)
    envelope=jnp.einsum('r,srg->sg',jnp.asarray(phase).conj()/3,blocks)
    envelope*=jnp.exp(-1j*mf.inputs.coords@(jnp.array([1/3,0,0])@cell.reciprocal_vectors()))
    np.testing.assert_allclose(response.values['rho_spin'],envelope,atol=5e-6,rtol=0)


@pytest.mark.parametrize('representation',['periodic','periodic_gaussian'])
def test_ofdft_finite_q_phonon_reference(representation):
    from gradscf import ofdft
    from gradscf.pbc.phonon import Phonons
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6,unit='Bohr',mesh=(5,)*3)
    calc=ofdft.OFDFT(cell,representation=representation,xc=None,tolerance=1e-9,maxiter=500).run()
    driver=Phonons(calc,masses=[1.,1.],supercell=(3,1,1))
    result=driver.kernel([1/3,0,0])
    assert result.converged
    assert result.hermiticity_error<1e-6
    assert np.isfinite(result.frequencies).all()
    sup=driver.super_reference.system
    pattern=np.zeros((3,2,3),complex);pattern[:,1,0]=np.exp(2j*np.pi*np.arange(3)/3)
    def energy(t,part):
        positions=np.asarray(sup.coords)+t*part.reshape(-1,3)
        moved=gto.M(atom=[('H',r) for r in positions],a=sup.lattice,unit='Bohr',mesh=sup.mesh)
        out=ofdft.OFDFT(moved,representation=representation,xc=None,tolerance=1e-10,maxiter=500).run()
        assert out.converged
        return out.e_tot
    base=energy(0.,pattern.real);step=1e-3
    fd=sum((energy(step,p)+energy(-step,p)-2*base)/step**2 for p in (pattern.real,pattern.imag))/3
    np.testing.assert_allclose(result.force_constants[3,3].real,fd,atol=2e-4,rtol=0)
    # A stale primitive state cannot silently be replaced by a new supercell root.
    calc.kinetic_params={'vw':.5}
    with pytest.raises(ValueError,match='stale'):
        driver.kernel([1/3,0,0])
    assert driver.result is None and driver.super_reference is None


def test_gamma_limits_and_invalid_masses_are_checked_before_response(monkeypatch):
    from gradscf.pbc import phonon
    cell=gto.M(atom='H 0 0 0; H 1.4 0 0',a=np.eye(3)*6,unit='Bohr',mesh=(7,)*3)
    mf=scf.RHF(cell)
    with pytest.raises(ValueError,match='mass'):
        phonon.Phonons(mf,masses=[1.,-1.])
    def forbidden(*args,**kwargs):raise AssertionError('Response must not be built beyond the size gate')
    monkeypatch.setattr(phonon,'atomic_problem',forbidden)
    with pytest.raises(ValueError,match='limit'):
        phonon.Phonons(mf,masses=[1.,1.],max_atoms=1).kernel()
