"""IBZ target contractions, full-grid equivalence and unrestricted backward space."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import dft
from gradscf.pbc import gto,scf
from gradscf.pbc.symmetry import Operation
from gradscf.integrals.periodic.fft import build_inputs,get_kpoint_jk
from gradscf.pbc._kpoint import energy_and_fock,run_kpoint_scf
from gradscf.pbc.occupations import fermi_density
from gradscf.scf import SCFDifferentiationConfig

CONTROLS=dict(max_cycle=100,damping=.1,conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10)


@pytest.fixture(scope='module')
def setup():
    from gradscf.pbc.reduction import SpatialReduction
    cell=gto.M(atom='H -.7 0 0; H .7 0 0',a=np.eye(3)*6.,unit='Bohr',mesh=(9,)*3)
    points=cell.make_kpts((3,1,1));data=build_inputs(cell,kpts=points)
    plan=SpatialReduction.from_cell(cell,points,operations=[Operation.from_cell(cell,-np.eye(3))])
    f=jnp.broadcast_to(data.hcore,(2,)+data.hcore.shape)
    density=fermi_density(f,data.overlap,jnp.array(cell.nelec),jnp.array(.2),data.kweights)[0]
    assert len(plan.representatives)==2 and plan.consistent_density(density)
    return cell,points,data,plan,density


def test_jk_contracts_only_representative_targets(setup,monkeypatch):
    import gradscf.integrals.periodic.fft as fft
    cell,_,data,plan,density=setup
    full_j,full_k=get_kpoint_jk(data,density,mesh=cell.mesh)
    calls=[];original=fft._pair_potential
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(fft,'_pair_potential',counted)
    j,k=get_kpoint_jk(data,density,mesh=cell.mesh,targets=plan.representatives)
    assert len(calls)==len(plan.representatives)*len(data.kpoints)
    np.testing.assert_allclose(j,full_j[plan.representatives],atol=1e-12)
    np.testing.assert_allclose(k,full_k[:,plan.representatives],atol=1e-12)


def test_j_only_preserves_input_spin_axis(setup):
    cell,_,data,plan,density=setup
    density=density[:1]
    full,_=get_kpoint_jk(data,density,mesh=cell.mesh,with_k=False)
    selected,zero=get_kpoint_jk(data,density,mesh=cell.mesh,with_k=False,targets=plan.representatives)
    assert zero.shape==(1,)+selected.shape
    np.testing.assert_allclose(selected,full[plan.representatives],atol=1e-12)


@pytest.mark.parametrize('xc,exxdiv',[('hf',None),('hf','ewald'),('pbe',None)])
def test_irreducible_energy_and_fock_equal_full(setup,xc,exxdiv):
    cell,_,data,plan,density=setup
    expected=energy_and_fock(density,data,mesh=cell.mesh,xc=xc,exxdiv=exxdiv)
    actual=energy_and_fock(density,data,mesh=cell.mesh,xc=xc,exxdiv=exxdiv,integration=plan)
    np.testing.assert_allclose(actual[0],expected[0],atol=1e-10,rtol=0)
    np.testing.assert_allclose(actual[1],expected[1],atol=1e-10,rtol=0)


@pytest.mark.parametrize('temperature',[None,.2])
def test_spatial_scf_matches_full_and_preserves_reference_guards(setup,temperature):
    cell,points,_,_,_=setup
    full=scf.KRHF(cell,kpts=points,temperature=temperature,exxdiv=None).run(**CONTROLS)
    reduced=scf.KRHF(cell,kpts=points,temperature=temperature,exxdiv=None,
                     symmetry=True,time_reversal=True).run(**CONTROLS)
    assert full.converged and reduced.converged
    assert len(reduced.reduction.representatives)<len(points)
    np.testing.assert_allclose(reduced.e_free,full.e_free,atol=1e-10)
    np.testing.assert_allclose(reduced.make_rdm1(),full.make_rdm1(),atol=1e-9)
    reduced.symmetry=False
    with pytest.raises(ValueError,match='symmetry'):
        reduced.to_reference()


def test_symmetry_breaking_implicit_response_uses_full_residual(setup):
    cell,_,data,plan,_=setup
    b=jnp.broadcast_to(jnp.array([[.1,.03j],[-.03j,-.2]]),data.hcore.shape)
    def solve(t,reduction):
        return run_kpoint_scf(data._replace(hcore=data.hcore+t*b),mesh=cell.mesh,
            nelec=cell.nelec,temperature=.2,exxdiv=None,reduction=reduction,**CONTROLS,
            differentiation=SCFDifferentiationConfig(mode='implicit',tolerance=1e-11,max_iter=80))
    assert solve(0.,plan).converged
    loss=lambda t:jnp.real(solve(t,plan).density_spin[0,0,0,0])
    derivative=jax.jit(jax.grad(loss))(0.)
    expected=jnp.real((solve(1e-5,None).density_spin[0,0,0,0]-solve(-1e-5,None).density_spin[0,0,0,0])/2e-5)
    assert abs(float(derivative))>1e-5
    np.testing.assert_allclose(derivative,expected,atol=1e-7,rtol=1e-5)


@pytest.mark.parametrize('mode',['explicit','implicit'])
def test_geometry_and_external_parameter_gradients(setup,mode):
    cell,points,_,plan,_=setup
    model=dft.Functional(lambda s:(s.rho,s.weights),lambda p,x:p*jnp.dot(x[1],x[0]**2))
    def energy(scale,parameter,reduction):
        data=build_inputs(cell,kpts=points,parameters=cell.parameters_at(cell.coords*scale),
                          lattice=cell.lattice*scale)
        result=run_kpoint_scf(data,mesh=cell.mesh,nelec=cell.nelec,functional=model,
            params=parameter,temperature=.2,exxdiv=None,reduction=reduction,**CONTROLS,
            differentiation=SCFDifferentiationConfig(mode=mode,tolerance=1e-11,max_iter=80))
        return result.free_energy
    value,derivatives=jax.jit(jax.value_and_grad(lambda a,p:energy(a,p,plan),argnums=(0,1)))(1.,.05)
    full=jax.jit(lambda a,p:energy(a,p,None))
    np.testing.assert_allclose(value,full(1.,.05),atol=1e-10)
    finite=[(full(1.+1e-5,.05)-full(1.-1e-5,.05))/2e-5,
            (full(1.,.05001)-full(1.,.04999))/2e-5]
    np.testing.assert_allclose(derivatives,finite,atol=1e-6,rtol=1e-5)


def test_broken_external_symmetry_is_not_silently_projected(setup):
    cell,_,data,plan,_=setup
    model=dft.Functional(lambda s:s.density_matrix,lambda p,d:.1*jnp.real(d[:,:,0,0].sum()))
    result=run_kpoint_scf(data,mesh=cell.mesh,nelec=cell.nelec,functional=model,
        temperature=.2,exxdiv=None,reduction=plan,max_cycle=3)
    assert not result.converged


def test_unrestricted_spatial_scf_matches_full():
    cell=gto.M(atom='H -.9 0 0; H 0 0 0; H .9 0 0',a=np.eye(3)*6.,
               spin=1,unit='Bohr',mesh=(9,)*3)
    points=cell.make_kpts((3,1,1))
    full=scf.KUHF(cell,kpts=points,temperature=.2,exxdiv=None).run(**CONTROLS)
    reduced=scf.KUHF(cell,kpts=points,temperature=.2,exxdiv=None,symmetry=True).run(**CONTROLS)
    assert full.converged and reduced.converged
    np.testing.assert_allclose(reduced.e_free,full.e_free,atol=1e-9)
    np.testing.assert_allclose(reduced.make_rdm1(),full.make_rdm1(),atol=1e-8)
