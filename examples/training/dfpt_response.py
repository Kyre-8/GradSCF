"""Train a four-layer external XC network against a static density response.

H2/6-31G*, CPU float64, seed 0. Reuses the network from density_matrix_mlp.py.
Synthetic teacher labels demonstrate the differentiable chain, not accuracy.
The AO response is dD/dF for h(F)=h(0)+F*z, in atomic units, at fixed geometry.
"""
import jax
import jax.numpy as jnp
from density_matrix_mlp import model, functional
from gradscf import gto, scf, dfpt
from gradscf.model.neural_xc import training

jax.config.update('jax_enable_x64', True)

mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='6-31g*')
mf = scf.RHF(mol, grids_level=0, conv_tol=1e-12).run()
reference = mf.to_reference()
params = model.init(jax.random.PRNGKey(0), mf.make_rdm1())
potential = reference.dipole_integrals[2]

teacher = jax.tree.map(lambda x:x+.02, params)
trainer = training.Trainer(functional, params=params)
trainer.mode = 'implicit'
trainer.scf = dict(max_cycle=80, damping=.2, conv_tol_energy=1e-12,
                   conv_tol_density=1e-10, eigenvalue_jitter=0., require_converged=True)
trainer.adjoint = dict(tolerance=1e-10, max_iter=60)
_, teacher_state = trainer.predict(reference, params=teacher)
target = dfpt.DFPT(teacher_state, functional=functional, params=teacher).response(potential=potential)
assert target.converged
sample = training.Sample(reference, response=(potential, jax.lax.stop_gradient(target.density)))

for mode in ('explicit', 'implicit'):
    trainer.params = params  # Reset parameters/Adam state for a fair comparison.
    trainer.mode = mode
    trainer.loss = {'response': {'mse':1., 'mae':.1}}
    trainer.learning_rate = .002
    trainer.run(sample, steps=3)
    before, after = trainer.history['loss'][0], trainer.history['loss'][-1]
    print(mode, 'response loss:', before, '->', after)
    print('accepted updates:', trainer.history['update_accepted'][1:])
    print('response converged:', trainer.metrics['response_converged'])
    assert jnp.isfinite(after) and after < before
    assert all(trainer.history['update_accepted'][1:])

# Measured CPU float64 output, JAX 0.8.1, seed 0:
# explicit response loss: 0.005450575906029184 -> 0.0011762965376142746
# accepted updates: [True, True, True]
# response converged: True
# implicit response loss: 0.0054505755938781195 -> 0.0011762966344321727
# accepted updates: [True, True, True]
# response converged: True
