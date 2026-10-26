"""The shared Trainer accepts periodic snapshots without a second SCF solver."""
import copy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import dft
from gradscf.pbc import gto, dft as pdft
from gradscf.model.neural_xc import training
from gradscf.pbc._kpoint import energy_and_fock, run_kpoint_scf


def functional():
    return dft.Functional(lambda s:(s.rho,s.weights),
        lambda p,x:p['scale']*jnp.dot(x[1],x[0]**2),
        init_fn=lambda key,x:{'scale':jnp.array(.15)})


@pytest.fixture(scope='module')
def reference():
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,
               unit='Bohr',mesh=(7,)*3)
    return pdft.KRKS(cell,kpts=cell.make_kpts((2,1,1),shift=(.5,0,0)),
        kweights=[.3,.7],functional=functional(),params={'scale':jnp.array(.2)},
        temperature=.1,exxdiv=None).run(max_cycle=100,damping=.1,
        conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10)


def trainer(mode,params=None):
    t=training.Trainer(functional(),params=params)
    t.mode=mode
    t.loss={'energy':{'mse':1.,'mae':1.}}
    t.learning_rate=.002
    t.scf=dict(max_cycle=100,damping=.1,conv_tol_energy=1e-12,
               conv_tol_density=1e-11,conv_tol_grad=1e-10)
    t.adjoint=dict(tolerance=1e-11,max_iter=80)
    return t


def test_sample_is_an_independent_traceable_periodic_snapshot(reference):
    assert reference.converged
    mf=copy.copy(reference)
    sample=training.Sample(mf,energy=mf.e_tot)
    state=sample.molecule
    assert state is not mf
    assert state.rdm1.shape==mf.result.density_spin.shape
    assert all(hasattr(x,'dtype') for x in jax.tree.leaves(sample))
    rebuilt=jax.tree.unflatten(jax.tree.structure(sample),jax.tree.leaves(sample))
    rho=training.density_on_grid(rebuilt.molecule)
    np.testing.assert_allclose(jnp.dot(state.grid.weights,rho),sum(mf.cell.nelec),atol=1e-9)
    before=trainer('fixed_density').evaluate(sample,params={'scale':jnp.array(.15)})['energy']
    mf.temperature=.2
    with pytest.raises(ValueError,match='temperature'):
        training.Sample(mf,energy=mf.e_tot)
    after=trainer('fixed_density').evaluate(sample,params={'scale':jnp.array(.15)})['energy']
    np.testing.assert_array_equal(before,after)


@pytest.mark.parametrize('mode',['fixed_density','explicit','implicit'])
def test_three_modes_match_array_energy_and_gradient(reference,mode):
    sample=training.Sample(reference,energy=reference.e_tot)
    t=trainer(mode)
    def energy(scale):
        return t.predict(sample.molecule,params={'scale':scale})[0]
    value,gradient=jax.jit(jax.value_and_grad(energy))(.15)
    def direct(scale):
        if mode=='fixed_density':
            return energy_and_fock(reference.result.density_spin,reference.inputs,
                mesh=reference.cell.mesh,functional=t.functional,params={'scale':scale},exxdiv=None)[0]
        return run_kpoint_scf(reference.inputs,mesh=reference.cell.mesh,nelec=reference.cell.nelec,
            functional=t.functional,params={'scale':scale},temperature=.1,exxdiv=None,
            init_density=reference.result.density_spin,max_cycle=100,damping=.1,
            conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10).total_energy
    expected=jax.jit(direct)
    np.testing.assert_allclose(value,expected(.15),atol=1e-10,rtol=0)
    fd=(expected(.15001)-expected(.14999))/2e-5
    np.testing.assert_allclose(gradient,fd,atol=2e-8,rtol=2e-5)
    assert abs(float(gradient))>1e-5
    t.run(sample,steps=2)
    assert t.history['loss'][-1]<t.history['loss'][0]
    assert t.history['optimizer_step']==[0,1,2]
    assert all(t.history['update_accepted'][1:])
    if mode!='fixed_density':
        assert all(t.history['scf_converged'])
    # The energy label always denotes internal E, including at finite T.
    at_reference=t.evaluate(sample,params=reference.params)
    np.testing.assert_allclose(at_reference['energy'],reference.e_tot,atol=1e-10)
    assert abs(float(reference.result.total_energy-reference.result.free_energy))>1e-4


def test_fixed_density_never_runs_scf(reference,monkeypatch):
    import gradscf.pbc._kpoint as kpoint
    sample=training.Sample(reference,energy=reference.e_tot)
    def forbidden(*args,**kwargs):
        raise AssertionError('Fixed-density training must not run SCF')
    monkeypatch.setattr(kpoint,'run_kpoint_scf',forbidden)
    trainer('fixed_density').run(sample,steps=1)


@pytest.mark.parametrize('mode',['explicit','implicit'])
def test_unconverged_periodic_training_does_not_update(reference,mode):
    t=trainer(mode,{'scale':jnp.array(.8)})
    t.scf['max_cycle']=1
    t.run(training.Sample(reference,energy=reference.e_tot),steps=1)
    assert not t.history['update_accepted'][-1]
    assert t.history['optimizer_step']==[0,0]
    np.testing.assert_array_equal(t.params['scale'],.8)


def test_explicit_opt_out_retains_shared_convergence_policy(reference):
    t=trainer('explicit',{'scale':jnp.array(.8)})
    t.scf.update(max_cycle=1,require_converged=False)
    t.run(training.Sample(reference,energy=reference.e_tot),steps=1)
    assert not t.history['scf_converged'][0]
    assert t.history['update_accepted'][-1]
    assert t.history['optimizer_step']==[0,1]
    assert t.params['scale']!=.8


@pytest.mark.parametrize('mode',['explicit','implicit'])
def test_density_supervision_has_scf_response(reference,mode):
    sample=training.Sample(reference,density=training.density_on_grid(reference.to_reference()))
    t=trainer(mode,{'scale':jnp.array(.15)})
    t.loss={'density':{'mse':1.}}
    from gradscf.model.neural_xc.training.targets import molecular_loss
    loss=lambda p:molecular_loss({'scale':p},t.functional,sample,training_config=t._configuration())[0]
    value,derivative=jax.jit(jax.value_and_grad(loss))(.15)
    fd=(loss(.15001)-loss(.14999))/2e-5
    assert value>0 and abs(float(derivative))>1e-8
    np.testing.assert_allclose(derivative,fd,atol=1e-9,rtol=2e-4)


def test_unsupported_periodic_targets_and_options_are_explicit(reference):
    with pytest.raises(NotImplementedError,match='energy.*density'):
        training.Sample(reference,excitation_energies=jnp.array([.1]))
    t=trainer('implicit')
    t.scf['iterate_selection']='best_rms'
    with pytest.raises(NotImplementedError,match='iterate_selection'):
        t.evaluate(training.Sample(reference,energy=reference.e_tot))


@pytest.mark.parametrize('unrestricted',[False,True])
def test_gamma_and_unrestricted_training(unrestricted):
    atom='H .2 .3 .4; H 1.6 .3 .4'+('; H 2.8 .7 .9' if unrestricted else '')
    cell=gto.M(atom=atom,a=np.eye(3)*6.,unit='Bohr',mesh=(7,)*3,spin=int(unrestricted))
    options=dict(functional=functional(),params={'scale':jnp.array(.2)},exxdiv=None)
    if unrestricted:
        mf=pdft.KUKS(cell,kpts=cell.make_kpts((2,1,1),shift=(.5,0,0)),
                     kweights=[.4,.6],temperature=.1,**options)
    else:
        mf=pdft.RKS(cell,**options)
    mf.run(max_cycle=100,damping=.1,conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10)
    assert mf.converged
    sample=training.Sample(mf,energy=mf.e_tot)
    t=trainer('implicit')
    np.testing.assert_allclose(t.evaluate(sample,params=mf.params)['energy'],mf.e_tot,atol=1e-10)
    t.run(sample,steps=1)
    assert t.history['update_accepted'][-1]
    assert t.history['loss'][-1]<t.history['loss'][0]
    spin_numbers=jnp.einsum('k,skij,kji->s',mf.inputs.kweights,
                            sample.molecule.rdm1,mf.inputs.overlap).real
    np.testing.assert_allclose(spin_numbers,mf.cell.nelec,atol=1e-9)
