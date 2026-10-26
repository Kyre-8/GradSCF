from importlib import import_module
from pathlib import Path
import runpy
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('tolerance', [0., -1., float('nan'), float('inf')])
def test_methane_rejects_invalid_implicit_tolerance(tolerance):
    module=vars(import_module("gradscf.model.nnao.training.objective"))
    with pytest.raises(ValueError,match='implicit_tolerance must be finite and positive'):
        module['BasisRHF'](implicit_tolerance=tolerance)


@pytest.mark.parametrize('shift',[-.1,float('nan'),float('inf')])
def test_methane_rejects_invalid_scf_rescue_level_shift(shift):
    module=vars(import_module("gradscf.model.nnao.training.objective"))
    with pytest.raises(ValueError,match='scf_rescue_level_shift must be finite and nonnegative'):
        module['BasisRHF'](scf_rescue_level_shift=shift)


@pytest.mark.parametrize('primary_converged',[True,False])
def test_scf_rescue_only_runs_after_failure_and_polishes_consistent_orbitals(primary_converged):
    module=vars(import_module("gradscf.model.nnao.training.objective"));events=[]
    class Result(NamedTuple):
        converged:object
        cycles:object
        density_matrix:object
        mo_coeff:object
        mo_energy:object
    expected_density=jnp.eye(2)*2.;expected_coeff=jnp.eye(2)*3.;expected_energy=jnp.arange(2.)
    def record(stage,error):events.append((int(stage),float(error)))
    def solver(*,config,**kwargs):
        shifted=config.level_shift>0
        polished='init_density' in kwargs
        stage=1 if shifted else 2 if polished else 0
        error=jnp.asarray(0.)
        if polished:
            error=(jnp.linalg.norm(kwargs['init_density']-expected_density)
                +jnp.linalg.norm(kwargs['init_mo_coeff']-expected_coeff)
                +jnp.linalg.norm(kwargs['init_mo_energy']-expected_energy))
        jax.debug.callback(record,jnp.asarray(stage),error,ordered=True)
        return Result(jnp.asarray(primary_converged or stage>0),jnp.asarray([7,5,2][stage]),
            expected_density,expected_coeff,expected_energy)
    cfg=module['RKSConfig'](xc_spec='hf')
    result,used,total=jax.jit(lambda:module['_run_scf_with_rescue'](
        solver,{},cfg,rescue_level_shift=.2))()
    jax.block_until_ready((result,used,total))
    assert events==([(0,0.)] if primary_converged else [(0,0.),(1,0.),(2,0.)])
    assert bool(used)==(not primary_converged)
    assert int(total)==(7 if primary_converged else 14)
    assert int(result.cycles)==(7 if primary_converged else 2)


def test_checked_scf_rejects_non_aufbau_stationary_density():
    module=vars(import_module("gradscf.model.nnao.training.objective"))
    info=dict(converged=True,scf_cycles=2,orbital_residual=1e-12,
        min_overlap_eigenvalue=.1,reconstruction_error=0.,fixed_point_residual=.5)
    with pytest.raises(RuntimeError,match='fixed point'):
        module['BasisRHF']._checked_info(-1.,info)


def test_nonfinite_coefficient_gradient_retains_outputs_and_valid_forward_diagnostics():
    module=vars(import_module("gradscf.model.nnao.training.objective"))
    experiment=object.__new__(module['BasisRHF'])
    experiment.ps=experiment.ph=experiment.rep=None
    outputs=np.array([[[.2,.3]]])
    gradient=np.array([[[np.nan,np.inf]]])
    info=dict(converged=True,scf_cycles=3,orbital_residual=1e-12,
              min_overlap_eigenvalue=.1,reconstruction_error=0.,fixed_point_residual=0.)
    experiment._value_grad=lambda *args:((-1.,info),gradient)
    with pytest.raises(RuntimeError) as caught:
        experiment.evaluate(outputs)
    error=caught.value
    assert str(error)=='Nonfinite energy or gradient'  # Existing per-molecule Adam contract.
    assert hasattr(error,'failure_arrays'), 'The actual failed coefficient backward was discarded.'
    np.testing.assert_array_equal(error.failure_arrays['basis_outputs'],outputs)
    np.testing.assert_array_equal(error.failure_arrays['coefficient_gradient'],gradient)
    assert error.failure_details==dict(energy_hartree=-1.,scf=info)


@pytest.mark.parametrize('bad_info,energy',[
    ({'converged':False},-1.),({'orbital_residual':1e-2},-1.),
    ({'min_overlap_eigenvalue':1e-12},-1.),({'reconstruction_error':1e-5},-1.),
    ({'orbital_residual':float('nan')},-1.),({},float('nan')),
])
def test_value_only_rejects_invalid_scf_without_calling_backward(bad_info,energy):
    module=vars(import_module("gradscf.model.nnao.training.objective"))
    experiment=object.__new__(module['BasisRHF'])
    experiment.ps=experiment.ph=experiment.rep=None
    info=dict(converged=True,scf_cycles=3,orbital_residual=1e-12,
              min_overlap_eigenvalue=.1,reconstruction_error=0.,fixed_point_residual=0.)
    info.update(bad_info)
    experiment._value=lambda *args:(energy,info)
    experiment._value_grad=lambda *args:pytest.fail('Value-only evaluation called backward.')
    with pytest.raises(RuntimeError):
        experiment.evaluate_value(jnp.zeros((1,1)))


def test_hydrogen_value_only_matches_implicit_energy_and_fd():
    module=vars(import_module("gradscf.model.nnao.training.objective"))
    experiment=module['BasisRHF'](basis_family='szp663_direct',core_primitives=6,
        jk_backend='df',implicit_tolerance=1e-8,geometry=dict(name='H2',charge=0,spin=0,
            symbols=['H','H'],coords_angstrom=[[0.,0.,0.],[0.,0.,.74]]))
    outputs=experiment.layout.reference_outputs()
    expected,gradient,_=experiment.evaluate(outputs)
    experiment._value_grad=lambda *args:pytest.fail('FD evaluation called backward.')
    actual,info=experiment.evaluate_value(outputs)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0.)
    assert info['converged'] and info['orbital_residual']<1e-7
    check=vars(import_module("gradscf.model.nnao.training.validation"))['directional_check']
    direction=np.asarray(gradient)/np.linalg.norm(np.asarray(gradient))
    result=check(experiment.evaluate_value,outputs,gradient,direction,
                 step=1e-4,atol=2e-6,rtol=2e-5)
    assert result['passed']


def test_energy_only_fd_rejects_an_unconverged_perturbation():
    module=vars(import_module("gradscf.model.nnao.training.objective"))
    experiment=object.__new__(module['BasisRHF'])
    experiment.ps=experiment.ph=experiment.rep=None
    info=dict(converged=False,scf_cycles=150,orbital_residual=1e-12,
              min_overlap_eigenvalue=.1,reconstruction_error=0.,fixed_point_residual=0.)
    experiment._value=lambda *args:(-1.,info)
    experiment._value_grad=lambda *args:pytest.fail('FD evaluation called backward.')
    check=vars(import_module("gradscf.model.nnao.training.validation"))['directional_check']
    with pytest.raises(RuntimeError,match='SCF is not stationary'):
        check(experiment.evaluate_value,np.zeros(1),np.zeros(1),np.ones(1),
              step=1e-4,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('implicit_tolerance',[1e-9,1e-8])
def test_methane_implicit_energy_gradient_matches_reconverged_scf(implicit_tolerance):
    script=Path('tools/optimize_methane_nnao.py')
    assert script.is_file(), 'Methane implicit-SCF experiment is missing'
    module=vars(import_module("gradscf.model.nnao.training.objective"))
    experiment=module['BasisRHF'](basis_family='szp3',implicit_tolerance=implicit_tolerance)
    assert experiment.implicit_tolerance==implicit_tolerance
    x=jnp.zeros((5,2,2))
    energy,grad,info=experiment.evaluate(x)
    assert info['converged'] and info['min_overlap_eigenvalue']>1e-5
    # Carbon s/p and symmetry-shared H s cover every independent shell channel.
    direction=np.zeros((5,2,2));direction[0]=[[.2,-.3],[.4,.1]];direction[1:,0]=[.15,-.2]
    step=1e-4
    e=lambda t:experiment.evaluate(x+t*direction)[0]
    fd=(8*(e(step)-e(-step))-e(2*step)+e(-2*step))/(12*step)
    np.testing.assert_allclose(np.sum(np.asarray(grad)*direction),fd,atol=2e-6,rtol=1e-5)
    assert experiment.evaluate(x-1e-3*grad)[0] < energy


def test_methane_implicit_scf_hessian_matches_reconverged_gradient():
    module=vars(import_module("gradscf.model.nnao.training.objective"))
    experiment=module['BasisRHF'](basis_family='szp3')
    x=jnp.zeros((5,2,2),dtype=jnp.float64)
    direction=jnp.zeros_like(x)
    direction=direction.at[0].set(jnp.array([[.2,-.3],[.4,.1]]))
    direction=direction.at[1:,0].set(jnp.array([.15,-.2]))
    value=lambda outputs:experiment._implicit_value(
        outputs,experiment.ps,experiment.ph,experiment.rep,
    )[0]
    gradient=jax.jit(jax.grad(value))
    implicit_hvp=jax.jit(
        lambda outputs,tangent:jax.jvp(gradient,(outputs,),(tangent,))[1]
    )(x,direction)
    step=1e-4
    finite_difference=(gradient(x+step*direction)-gradient(x-step*direction))/(2*step)
    np.testing.assert_allclose(implicit_hvp,finite_difference,atol=3e-5,rtol=3e-4)


def test_adam_rejects_invalid_trial_and_halves_learning_rate():
    module=vars(import_module("gradscf.model.nnao.training.objective"))
    run_adam=import_module("gradscf.model.nnao.workflows.optimize")._run_adam_steps
    attempts=[];records=[]

    def evaluate(vector):
        value=float(np.asarray(vector)[0]);attempts.append(value)
        if value>.75:raise RuntimeError('Nonfinite energy or gradient')
        energy=(value-2.)**2
        gradient=np.asarray([2.*(value-2.)])
        return energy,gradient,{},np.asarray(vector)

    final,stats=run_adam(
        np.asarray([0.]),evaluate,
        lambda vector,**metadata:records.append((np.asarray(vector),metadata)),
        steps=1,initial_learning_rate=1.,final_learning_rate=1.,
        gradient_clip_norm=1.,max_retries=3,
    )

    np.testing.assert_allclose(final,.5,atol=1e-7)
    assert attempts==pytest.approx([0.,1.,.5])
    assert stats==dict(iterations=1,evaluations=3,rejected_trials=1)
    assert records[-1][1]==dict(learning_rate=.5,rejected_trials=1)
