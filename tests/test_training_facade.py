"""Short public training API; numerical paths remain in the shared trainer."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_external_neural_xc import molecule, functional


def test_sample_reuses_existing_pytree_record():
    from gradscf.model.neural_xc.training import Sample, MolecularTrainingDatum
    mol = molecule()
    sample = Sample(mol, energy=-2.1, density=jnp.ones(3))
    assert isinstance(sample, MolecularTrainingDatum)
    assert sample.molecule is mol
    assert sample.target_e0_total_h == -2.1
    np.testing.assert_array_equal(sample.target_grid_density, jnp.ones(3))
    rebuilt = jax.tree.unflatten(jax.tree.structure(sample), jax.tree.leaves(sample))
    assert rebuilt.target_e0_total_h == -2.1


def test_run_returns_self_and_history_includes_initial_and_post_update_losses():
    from gradscf.model.neural_xc.training import Sample, Trainer
    data = [Sample(molecule(), energy=-2.1)]
    trainer = Trainer(functional(), params={'scale':jnp.array(.3)})
    trainer.loss = {'energy': {'mse':1., 'mae':1.}}
    trainer.learning_rate = .002
    initial = trainer.evaluate(data)
    assert trainer.run(data, steps=5) is trainer
    assert trainer.history['step'] == list(range(6))
    np.testing.assert_allclose(trainer.history['loss'][0], initial['loss'])
    after = trainer.evaluate(data)
    np.testing.assert_allclose(trainer.history['loss'][-1], after['loss'])
    assert trainer.history['loss'][-1] < trainer.history['loss'][0]
    np.testing.assert_allclose(trainer.history['loss'],
        np.array(trainer.history['energy_mse'])+trainer.history['energy_mae'])
    assert trainer.history['update_accepted'] == [None]+[True]*5
    assert trainer.history['scf_cycles'] == [0]*6


def test_sequential_runs_preserve_optimizer_and_history():
    from gradscf.model.neural_xc.training import Sample, Trainer
    data=[Sample(molecule(),energy=-2.1)]
    a=Trainer(functional(),params={'scale':jnp.array(.3)})
    b=Trainer(functional(),params={'scale':jnp.array(.3)})
    a.run(data,steps=3).run(data,steps=2)
    b.run(data,steps=5)
    np.testing.assert_allclose(a.params['scale'],b.params['scale'],atol=1e-14)
    np.testing.assert_allclose(a.history['loss'],b.history['loss'],atol=1e-14)
    assert a.history['optimizer_step'] == list(range(6))


def test_fixed_density_does_not_silently_enter_scf_for_density_labels():
    from gradscf.model.neural_xc.training import Sample, Trainer
    trainer=Trainer(functional(),params={'scale':jnp.array(.3)})
    trainer.loss={'density':{'mse':1.}}
    with pytest.raises(ValueError,match='self.consistent'):
        trainer.run([Sample(molecule(),density=jnp.ones(3))],steps=1)


def test_loss_typo_rejected_before_training():
    from gradscf.model.neural_xc.training import Sample, Trainer
    trainer=Trainer(functional(),params={'scale':jnp.array(.3)})
    trainer.loss={'energy':{'msee':1.}}
    with pytest.raises(ValueError,match='msee'):
        trainer.run([Sample(molecule(),energy=-2.)],steps=1)


def test_nonfinite_update_has_no_optimizer_step_but_is_recorded():
    from gradscf.model.neural_xc.training import Sample, Trainer
    from gradscf.dft import Functional
    f=Functional(lambda s:s.rho, lambda p,x:jnp.sqrt(p['x'])*x.sum())
    trainer=Trainer(f,params={'x':jnp.array(-1.)})
    trainer.run([Sample(molecule(),energy=-2.)],steps=1)
    assert trainer.history['update_accepted']==[None,False]
    assert trainer.history['optimizer_step']==[0,0]
    assert trainer.params['x']==-1.


def test_explicit_name_preserves_existing_scf_gradient_behavior():
    from gradscf.scf import SCFDifferentiationConfig
    from gradscf.scf.autodiff import normalize_scf_gradient_mode
    from gradscf.model.neural_xc.training import Sample, Trainer
    assert normalize_scf_gradient_mode('explicit') == 'explicit'
    assert normalize_scf_gradient_mode('expl') == 'explicit'
    assert normalize_scf_gradient_mode('unrolled') == 'explicit'
    assert SCFDifferentiationConfig(mode='explicit').mode == 'explicit'
    trainer=Trainer(functional(),params={'scale':jnp.array(.3)})
    trainer.mode='explicit'
    trainer.scf=dict(max_cycle=60,damping=0.,conv_tol_density=1e-10,
                     conv_tol_energy=1e-12,eigenvalue_jitter=0.)
    trainer.run([Sample(molecule(),energy=-2.1)],steps=2)
    assert len(trainer.history['loss'])==3
    assert all(trainer.history['scf_converged'])
    assert all(trainer.history['update_accepted'][1:])


@pytest.mark.parametrize('rate', [float('nan'), float('inf'), -float('inf'), 0., [.1]])
def test_learning_rate_must_be_finite_positive_scalar(rate):
    from gradscf.model.neural_xc.training import Sample, Trainer
    trainer=Trainer(functional(),params={'scale':jnp.array(.3)})
    trainer.learning_rate=rate
    with pytest.raises(ValueError,match='finite positive scalar'):
        trainer.run([Sample(molecule(),energy=-2.1)],steps=1)


def test_weighted_metrics_match_weighted_loss():
    from gradscf.model.neural_xc.training import Sample, Trainer
    trainer=Trainer(functional(),params={'scale':jnp.array(.3)})
    trainer.loss={'energy':{'mse':.2,'mae':.7}}
    data=[Sample(molecule(),energy=-2.1,weight=2.),Sample(molecule(),energy=-1.5,weight=1.)]
    result=trainer.evaluate(data)
    errors=result['energy']-jnp.array([-2.1,-1.5])
    mse=jnp.sum(jnp.array([2.,1.])*errors**2)/3
    mae=jnp.sum(jnp.array([2.,1.])*abs(errors))/3
    np.testing.assert_allclose(result['energy_mse'],mse)
    np.testing.assert_allclose(result['energy_mae'],mae)
    np.testing.assert_allclose(result['loss'],.2*mse+.7*mae)
