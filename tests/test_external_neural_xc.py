"""External XC: current physical inputs, one energy, three training modes."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from gradscf.scf import RestrictedMolecule, QuadratureGrid


def molecule():
    ao = jnp.array([[1., .2], [.3, .9], [.6, -.4]])
    return RestrictedMolecule(
        ao=ao, ao_deriv1=jnp.stack([ao, .2*ao, -.1*ao, .3*ao]),
        ao_laplacian=.14*ao,
        grid=QuadratureGrid(jnp.array([.3, .4, .3]), jnp.zeros((3, 3))),
        dipole_integrals=jnp.zeros((3, 2, 2)), rep_tensor=jnp.zeros((2,)*4),
        h1e=jnp.array([[-1., .08], [.08, .6]]), overlap_matrix=jnp.eye(2),
        mo_coeff=jnp.stack([jnp.eye(2)]*2), mo_occ=jnp.array([[1., 0.]]*2),
        mo_energy=jnp.array([[-1., .6]]*2), rdm1=jnp.array([[[1., 0.], [0., 0.]]]*2),
        nuclear_repulsion=0., nocc=1,
    )


def inputs(state):
    # External feature schema is a nested PyTree, not a prescribed feature vector.
    return {'local': (state.rho, state.grad_rho, state.tau), 'weights': state.weights}


def energy(params, features):
    rho, grad, tau = features['local']
    return params['scale'] * jnp.sum(features['weights'] * (
        .1*rho**2 + .01*jnp.sum(grad**2, axis=-1) + .03*tau)) + .01*params['scale']*jnp.sum(features['weights']*rho)**2


def functional():
    from gradscf.dft import Functional
    return Functional(input_fn=inputs, energy_fn=energy)


@pytest.mark.parametrize('mode', ['explicit', 'implicit'])
def test_dfpt_functional_parameter_gradient_includes_relaxed_reference(mode):
    from gradscf import dfpt
    from gradscf.scf import DifferentiableSCF, DifferentiableSCFConfig
    f, mol = functional(), molecule()
    solver = DifferentiableSCF(DifferentiableSCFConfig(mode='self_consistent',
        gradient_mode=mode, max_cycle=60, damping=0., conv_tol_density=1e-11,
        conv_tol_energy=1e-12, eigenvalue_jitter=0.,
        implicit_diff_tolerance=1e-11, implicit_diff_max_iter=40))
    b = jnp.array([[.2, .1], [.1, -.1]])
    def solve(scale):
        return solver.run(mol, f, {'scale': scale})
    def response(scale):
        state, info = solve(scale)
        return dfpt.DFPT(state, functional=f, params={'scale': scale},
                         converged=info.converged).response(potential=b)
    out = response(.3)
    assert out.converged
    # Independent forward oracle: solve both externally perturbed ground states.
    plus, ip = solver.run(replace(mol, h1e=mol.h1e+1e-4*b), f, {'scale':.3})
    minus, im = solver.run(replace(mol, h1e=mol.h1e-1e-4*b), f, {'scale':.3})
    assert ip.converged and im.converged
    np.testing.assert_allclose(out.density, (plus.rdm1-minus.rdm1).sum(axis=0)/2e-4, atol=1e-8)
    value = lambda scale: jnp.sum(response(scale).density**2)
    actual = jax.jit(jax.grad(value))(.3)
    expected = (value(.30001)-value(.29999))/2e-5
    assert abs(actual) > 1e-5
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-8)


def test_response_training_api_and_fixed_density_boundary():
    from gradscf.model.neural_xc.training import Sample, Trainer
    datum = Sample(molecule(), response=(jnp.eye(2), jnp.zeros((2,2))))
    trainer = Trainer(functional(), params={'scale':jnp.array(.3)})
    trainer.loss = {'response': {'mse':1., 'mae':.1}}
    with pytest.raises(ValueError, match='self-consistent'):
        trainer._configuration()
    trainer.mode = 'implicit'
    cfg = trainer._configuration()
    assert cfg.response_mse_weight == 1.
    np.testing.assert_array_equal(datum.response_potential, jnp.eye(2))


@pytest.mark.parametrize('mode', ['explicit', 'implicit'])
def test_response_loss_training_gradient_and_accepted_step(mode):
    from gradscf.model.neural_xc.training import Sample, Trainer, molecular_loss
    f = functional()
    datum = Sample(molecule(), response=(jnp.array([[.2,.1],[.1,-.1]]), jnp.zeros((2,2))))
    trainer = Trainer(f, params={'scale':jnp.array(.3)})
    trainer.mode = mode
    trainer.loss = {'response': {'mse':1., 'mae':.1}}
    trainer.scf = dict(max_cycle=60, damping=0., conv_tol_density=1e-11,
        conv_tol_energy=1e-12, eigenvalue_jitter=0., require_converged=True)
    trainer.adjoint = dict(tolerance=1e-11, max_iter=40)
    cfg = trainer._configuration()
    loss = lambda s: molecular_loss({'scale':s}, f, datum, training_config=cfg)[0]
    value, gradient = jax.jit(jax.value_and_grad(loss))(.3)
    finite_difference = (loss(.30001)-loss(.29999))/2e-5
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-5, atol=1e-9)
    assert abs(gradient) > 1e-6
    trainer.run(datum, steps=2)
    assert trainer.metrics['response_converged']
    assert trainer.metrics['loss'] < value
    assert trainer.history['optimizer_step'][-1] == 2


def test_failed_response_training_preserves_parameters_and_optimizer():
    from flax.training.train_state import TrainState
    from gradscf.model.neural_xc.training import Sample, MolecularTrainingConfig
    from gradscf.model.neural_xc.training.trainer import make_molecular_train_step
    f = functional()
    cfg = MolecularTrainingConfig(mode='self_consistent', scf_gradient_mode='explicit',
        scf_max_cycle=1, scf_require_converged=False, response_mse_weight=1.)
    state = TrainState.create(apply_fn=f.apply, params={'scale':jnp.array(.3)}, tx=optax.adam(.01))
    sample = Sample(molecule(), response=(jnp.eye(2), jnp.zeros((2,2))))
    updated, metrics = jax.jit(make_molecular_train_step(f, cfg))(state, sample)
    assert not metrics['update_accepted']
    assert not metrics['response_converged'][0]
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(updated)):
        np.testing.assert_array_equal(before, after)


def test_response_target_shape_and_reference_occupation_guards():
    from gradscf import dfpt
    from gradscf.model.neural_xc.training import Sample
    mol = molecule()
    with pytest.raises(ValueError, match='AO matrices'):
        Sample(mol, response=(jnp.eye(2), jnp.zeros(2)))
    fractional = replace(mol, mo_occ=mol.mo_occ*.9)
    response = dfpt.DFPT(fractional, functional=functional(), params={'scale':.3}).response(potential=jnp.eye(2))
    assert not response.converged
    assert np.isnan(response.density).all()


@pytest.mark.parametrize('layout', ['packed', 'df'])
def test_dfpt_functional_integral_layouts_match_full(layout):
    from gradscf import dfpt
    from gradscf.scf import DifferentiableSCF, DifferentiableSCFConfig
    factors = jnp.array([[[.2,.1],[.1,.3]], [[.08,0.],[0.,.12]]])
    eri = jnp.einsum('Pij,Pkl->ijkl', factors, factors)
    full = replace(molecule(), rep_tensor=eri)
    i,j = np.tril_indices(2)
    compressed = replace(full, rep_tensor=jnp.zeros((0,)),
        eri_pair_matrix=eri[i[:,None],j[:,None],i[None,:],j[None,:]] if layout == 'packed' else None,
        df_factors=factors if layout == 'df' else None)
    solver = DifferentiableSCF(DifferentiableSCFConfig(mode='self_consistent',
        max_cycle=60, damping=0., eigenvalue_jitter=0., conv_tol_density=1e-11,
        conv_tol_energy=1e-12))
    def run(mol):
        state, info = solver.run(mol, functional(), {'scale':.3})
        assert info.converged
        out = dfpt.DFPT(state, functional=functional(), params={'scale':.3},
                        converged=info.converged).response(potential=jnp.array([[.2,.1],[.1,-.1]]))
        assert out.converged
        return out
    expected, actual = run(full), run(compressed)
    np.testing.assert_allclose(actual.density, expected.density, atol=1e-10)
    np.testing.assert_allclose(actual.energy, expected.energy, atol=1e-10)


def test_dfpt_functional_orthonormal_state_without_overlap():
    from gradscf import dfpt
    from gradscf.scf import DifferentiableSCF, DifferentiableSCFConfig
    f = functional()
    solver = DifferentiableSCF(DifferentiableSCFConfig(mode='self_consistent',
        max_cycle=60, damping=0., eigenvalue_jitter=0., conv_tol_density=1e-11))
    state, info = solver.run(replace(molecule(), overlap_matrix=None), f, {'scale':.3})
    assert info.converged
    b = jnp.array([[.2,.1],[.1,-.1]])
    def run(mol):
        return dfpt.DFPT(mol, functional=f, params={'scale':.3}).response(potential=b)
    actual, expected = run(state), run(replace(state, overlap_matrix=jnp.eye(2)))
    assert actual.converged
    np.testing.assert_allclose(actual.density, expected.density, atol=1e-12)


def test_current_density_inputs_and_potential_kernel_fd():
    f, mol, p = functional(), molecule(), {'scale': jnp.array(.3)}
    dm = mol.rdm1.sum(axis=0)
    tangent = jnp.array([[.1, -.2], [-.2, .05]])
    # The molecule deliberately retains its initial rdm1. Each callback must
    # use its supplied trial density for ALL density-dependent quantities.
    def value(d):
        return f.energy_for_density(p, mol, d)
    current = f.inputs_for_density(mol, dm + tangent)
    expected_rho = jnp.einsum('pq,gp,gq->g', dm+tangent, mol.ao, mol.ao)
    np.testing.assert_allclose(current['local'][0], expected_rho)
    eps = 1e-5
    potential = f.potential(p, mol, dm)
    np.testing.assert_allclose(jnp.sum(potential*tangent),
        (value(dm+eps*tangent)-value(dm-eps*tangent))/(2*eps), atol=1e-10)
    kernel = jax.jit(lambda d, t: f.kernel_action(p, mol, d, t))(dm, tangent)
    fd = (f.potential(p,mol,dm+eps*tangent)-f.potential(p,mol,dm-eps*tangent))/(2*eps)
    np.testing.assert_allclose(kernel, fd, atol=1e-10)
    # Mixed network/kernel derivative must survive the response boundary.
    assert abs(float(jax.grad(lambda s:jnp.sum(f.kernel_action({'scale':s},mol,dm,tangent)**2))(.3))) > 1e-6


def test_spin_context_and_missing_physical_quantity():
    from gradscf.dft import Functional
    mol = molecule()
    f = Functional(lambda s: (s.rho_spin, s.laplacian_rho, s.weights),
        lambda p, x: p*jnp.sum(x[2]*(x[0][0]**2+2*x[0][1]**2)))
    dm = mol.rdm1.at[1].multiply(.4)
    pot = f.potential(.2,mol,dm)
    assert pot.shape == dm.shape
    np.testing.assert_allclose(pot[1], .8*pot[0], atol=1e-12)
    with pytest.raises(AttributeError, match='ao_laplacian'):
        f.energy_for_density(.2, replace(mol,ao_laplacian=None), dm)


@pytest.mark.parametrize('mode, backward', [
    ('fixed_density','implicit'), ('self_consistent','explicit'), ('self_consistent','implicit')])
def test_three_training_modes_have_fd_gradients_and_accept_external_params(mode, backward):
    from gradscf.model.neural_xc.training import MolecularTrainingConfig, MolecularTrainingDatum, molecular_loss, Trainer
    f, mol = functional(), molecule()
    cfg = MolecularTrainingConfig(mode=mode,scf_gradient_mode=backward,
        e0_total_mse_weight=1.,scf_max_cycle=60,scf_damping=0.,
        scf_conv_tol_density=1e-10,scf_conv_tol_energy=1e-12,
        scf_eigenvalue_jitter=0.,scf_implicit_diff_tolerance=1e-10,
        scf_implicit_diff_max_iter=40,scf_require_converged=True)
    datum = MolecularTrainingDatum(mol,target_e0_total_h=jnp.array(-2.1))
    loss = lambda s:molecular_loss({'scale':s},f,datum,training_config=cfg)[0]
    value, derivative = jax.jit(jax.value_and_grad(loss))(.3)
    fd = (loss(.30001)-loss(.29999))/2e-5
    assert np.isfinite(value) and abs(float(derivative))>1e-5
    np.testing.assert_allclose(derivative,fd,atol=1e-8,rtol=1e-5)
    result = Trainer(f, params={'scale':jnp.array(.3)})
    result.mode = 'fixed_density' if mode == 'fixed_density' else backward
    result.learning_rate = 1e-3
    result.scf = dict(max_cycle=60,damping=0.,conv_tol_density=1e-10,
                      conv_tol_energy=1e-12,eigenvalue_jitter=0.,require_converged=True)
    result.adjoint = dict(tolerance=1e-10,max_iter=40)
    result.run([datum],steps=1)
    assert float(loss(result.params['scale'])) < float(value)


def test_parameter_gradient_includes_self_consistent_density_response():
    from gradscf.scf import DifferentiableSCF, DifferentiableSCFConfig, SCFDifferentiationConfig
    f, mol = functional(), molecule()
    def solve(s, mode):
        solver = DifferentiableSCF(DifferentiableSCFConfig(mode='self_consistent',
            max_cycle=70,damping=0.,conv_tol_density=1e-11,conv_tol_energy=1e-12,
            eigenvalue_jitter=0.,differentiation=SCFDifferentiationConfig(mode=mode,tolerance=1e-10)))
        state, info = solver.run(mol,f,{'scale':s})
        return state.rdm1[:,0,1].sum(), info.converged
    derivatives=[]
    for mode in ('implicit','explicit'):
        val, grad = jax.jit(jax.value_and_grad(lambda s:solve(s,mode)[0]))(.3)
        assert bool(solve(.3,mode)[1])
        fd=(solve(.30001,mode)[0]-solve(.29999,mode)[0])/2e-5
        assert abs(float(grad))>1e-4
        np.testing.assert_allclose(grad,fd,rtol=1e-5,atol=1e-8)
        derivatives.append(grad)
    np.testing.assert_allclose(*derivatives,rtol=1e-5,atol=1e-8)


@pytest.mark.parametrize('spin', [False, True])
def test_ao_response_is_used_by_tda_and_tddft(spin):
    from gradscf.scf import UnrestrictedMolecule
    from gradscf.tddft.response import build_restricted_tda_operator, build_restricted_tdhf_operator
    from gradscf.tddft.unrestricted import build_unrestricted_tda_operator, build_unrestricted_tdhf_operator
    f, mol, p = functional(), molecule(), {'scale':jnp.array(.3)}
    if spin:
        fields=UnrestrictedMolecule.__dataclass_fields__
        mol=UnrestrictedMolecule(**{k:v for k,v in vars(mol).items() if k in fields},nocc_alpha=1,nocc_beta=1)
        tda=build_unrestricted_tda_operator
        rpa=build_unrestricted_tdhf_operator
        dm=mol.rdm1
        t=jnp.zeros_like(dm).at[0,0,1].set(.5).at[0,1,0].set(.5)
        delta=f.kernel_action(p,mol,dm,t)
        expected=jnp.array([[1.6+delta[0,0,1],delta[1,0,1]],
                            [delta[1,0,1],1.6+delta[1,0,1]]])
    else:
        tda=build_restricted_tda_operator
        rpa=build_restricted_tdhf_operator
        dm=mol.rdm1.sum(axis=0)
        t=jnp.array([[0.,1.],[1.,0.]])
        delta=f.kernel_action(p,mol,dm,t)
        expected=jnp.array([[1.6+delta[0,1]]])
    operator=tda(mol,f,xc_params=p)
    matrix=operator[0](jnp.eye(expected.shape[0]))
    np.testing.assert_allclose(matrix,expected,atol=1e-11)
    # Full response must consume the same kernel, including its B block.
    full=rpa(mol,f,xc_params=p)
    action=full[0] if spin else full
    coupling=expected-1.6*jnp.eye(expected.shape[0])
    np.testing.assert_allclose(action(jnp.eye(2*expected.shape[0])),
        jnp.block([[expected,-coupling],[coupling,-expected]]),atol=1e-11)


def test_nonfinite_training_step_is_rejected_without_changing_optimizer_state():
    from flax.training.train_state import TrainState
    from gradscf.model.neural_xc.training import MolecularTrainingConfig, MolecularTrainingDatum
    from gradscf.model.neural_xc.training.trainer import make_molecular_train_step
    from gradscf.dft import Functional
    f=Functional(lambda s:s.rho,lambda p,x:jnp.sqrt(p["scale"])*jnp.sum(x))
    cfg=MolecularTrainingConfig(e0_total_mse_weight=1.)
    state=TrainState.create(apply_fn=f.apply,params={"scale":jnp.array(-1.)},tx=optax.adam(.01))
    datum=MolecularTrainingDatum(molecule(),target_e0_total_h=jnp.array(-2.))
    updated,metrics=make_molecular_train_step(f,cfg)(state,datum)
    for before,after in zip(jax.tree.leaves(state),jax.tree.leaves(updated)):
        np.testing.assert_array_equal(before,after)
    assert not bool(metrics['update_accepted'])


@pytest.mark.parametrize('backward', ['implicit', 'explicit'])
def test_external_unrestricted_scf_density_response(backward):
    from gradscf.scf import UnrestrictedMolecule, DifferentiableSCF, DifferentiableSCFConfig, SCFDifferentiationConfig
    mol=molecule()
    fields=UnrestrictedMolecule.__dataclass_fields__
    values={k:v for k,v in vars(mol).items() if k in fields}
    values['mo_occ']=jnp.array([[1.,0.],[0.,0.]])
    values['rdm1']=mol.rdm1.at[1].set(0.)
    mol=UnrestrictedMolecule(**values,nocc_alpha=1,nocc_beta=0)
    f=functional()
    solver=DifferentiableSCF(DifferentiableSCFConfig(mode='self_consistent',max_cycle=65,
        damping=0.,eigenvalue_jitter=0.,conv_tol_density=1e-10,conv_tol_energy=1e-12,
        differentiation=SCFDifferentiationConfig(mode=backward,tolerance=1e-10)))
    def probe(s):
        result,info=solver.run(mol,f,{'scale':s})
        return result.rdm1[0,0,1],info.converged
    value,derivative=jax.jit(jax.value_and_grad(lambda s:probe(s)[0]))(.3)
    assert bool(probe(.3)[1]) and np.isfinite(value)
    fd=(probe(.30001)[0]-probe(.29999)[0])/2e-5
    assert abs(float(derivative))>1e-5
    np.testing.assert_allclose(derivative,fd,rtol=1e-5,atol=1e-8)
    from gradscf import dfpt
    b = jnp.array([[.2,.1],[.1,-.1]])
    def response(scale):
        state, info = solver.run(mol, f, {'scale':scale})
        return dfpt.DFPT(state, functional=f, params={'scale':scale},
                         converged=info.converged).response(potential=b)
    out = response(.3)
    assert out.converged
    np.testing.assert_allclose(out.values['density_spin'][1], 0., atol=1e-12)
    plus, ip = solver.run(replace(mol, h1e=mol.h1e+1e-4*b), f, {'scale':.3})
    minus, im = solver.run(replace(mol, h1e=mol.h1e-1e-4*b), f, {'scale':.3})
    assert ip.converged and im.converged
    np.testing.assert_allclose(out.density, (plus.rdm1-minus.rdm1).sum(axis=0)/2e-4, atol=1e-8)
    objective = lambda scale:jnp.sum(response(scale).density**2)
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(.3),
        (objective(.30001)-objective(.29999))/2e-5, rtol=2e-5, atol=1e-8)


def test_external_implicit_training_requires_converged_state_by_default():
    from gradscf.model.neural_xc.training import MolecularTrainingConfig, MolecularTrainingDatum
    from gradscf.model.neural_xc.training.trainer import make_molecular_train_step
    from flax.training.train_state import TrainState
    f=functional()
    cfg=MolecularTrainingConfig(mode='self_consistent',scf_gradient_mode='implicit',
        scf_max_cycle=1,e0_total_mse_weight=1.)
    state=TrainState.create(apply_fn=f.apply,params={'scale':jnp.array(.3)},tx=optax.adam(.01))
    datum=MolecularTrainingDatum(molecule(),target_e0_total_h=jnp.array(-2.1))
    updated,metrics=make_molecular_train_step(f,cfg)(state,datum)
    assert not bool(metrics['update_accepted'])
    assert int(updated.step)==0


def test_external_init_fn_can_initialize_pytree_inputs():
    from gradscf.dft import Functional
    from gradscf.model.neural_xc.training.trainer import create_train_state_from_molecule
    f=Functional(inputs,energy,init_fn=lambda key,x:{'scale':jnp.sum(x['weights'])/10})
    state=create_train_state_from_molecule(f,jax.random.PRNGKey(0),molecule(),optax.adam(.01))
    np.testing.assert_allclose(state.params['scale'],.1)


def test_response_binding_does_not_replace_the_scf_functional():
    from gradscf.model.neural_xc.training.targets import _freeze_functional_for_fractional_path
    f, mol, p = functional(), molecule(), {'scale':jnp.array(.3)}
    frozen, params = _freeze_functional_for_fractional_path(p, f, mol)
    assert frozen is f
    np.testing.assert_allclose(frozen.energy_from_molecule(params,mol),f.energy_from_molecule(p,mol))


def test_explicit_explicit_convergence_policy_rejects_finite_unconverged_updates():
    from flax.training.train_state import TrainState
    from gradscf.model.neural_xc.training import MolecularTrainingConfig, MolecularTrainingDatum
    from gradscf.model.neural_xc.training.trainer import make_molecular_train_step
    f=functional()
    cfg=MolecularTrainingConfig(mode='self_consistent',scf_gradient_mode='explicit',
        scf_require_converged=True,scf_max_cycle=1,e0_total_mse_weight=1.)
    state=TrainState.create(apply_fn=f.apply,params={'scale':jnp.array(.3)},tx=optax.adam(.01))
    datum=MolecularTrainingDatum(molecule(),target_e0_total_h=jnp.array(-2.1))
    updated,metrics=make_molecular_train_step(f,cfg)(state,datum)
    assert bool(jnp.isfinite(metrics['total_loss']))
    assert not bool(metrics['update_accepted'])
    assert int(updated.step)==0
