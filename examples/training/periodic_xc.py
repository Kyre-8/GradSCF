"""Train an external four-layer XC network on a periodic H2 cell.

CPU float64, seed 0, GTH-SZV/GTH-Pade, a 6-Bohr cubic cell, shifted 2x1x1
k mesh, 9x9x9 FFT grid, kBT=0.1 Eh. This coarse synthetic teacher experiment
illustrates derivatives, not material accuracy. All three modes fit the same
teacher internal energy per cell; fixed_density freezes the initial density.
SCF minimizes free energy at finite temperature; the energy label remains E.
The model and its input features are ordinary user-defined JAX functions.
"""
import jax
import jax.numpy as jnp
from gradscf import dft, dfpt
from gradscf.pbc import gto, dft as pdft
from gradscf.model.neural_xc import training

jax.config.update('jax_enable_x64', True)


def features(state):
    x = jnp.stack((state.rho, jnp.sum(state.grad_rho**2, axis=-1), state.tau), axis=-1)
    return x, state.weights * state.rho


def energy(params, inputs):
    x, weights = inputs
    for weight, bias in params[:-1]:
        x = jnp.tanh(x @ weight + bias)
    weight, bias = params[-1]
    return .1 * jnp.dot(weights, (x @ weight + bias)[:, 0])


functional = dft.Functional(features, energy)
sizes = (3, 8, 8, 8, 1)
keys = jax.random.split(jax.random.PRNGKey(0), 4)
params = tuple((.1 * jax.random.normal(key, (n, m)), jnp.zeros(m))
               for key, n, m in zip(keys, sizes[:-1], sizes[1:]))
cell = gto.M(atom='H .2 .3 .4; H 1.6 .3 .4', a=jnp.eye(3)*6.,
             unit='Bohr', basis='gth-szv', pseudo='gth-pade', mesh=(9,)*3)
kpoints = cell.make_kpts((2, 1, 1), shift=(.5, 0, 0))
controls = dict(max_cycle=80, damping=.1, conv_tol=1e-12,
                conv_tol_density=1e-10, conv_tol_grad=1e-9)
reference = pdft.KRKS(cell, kpts=kpoints, functional=functional, params=params,
                     temperature=.1, exxdiv=None).run(**controls)
teacher = tuple((weight + .02, bias + .03) for weight, bias in params)
teacher_state = pdft.KRKS(cell, kpts=kpoints, functional=functional, params=teacher,
                         temperature=.1, exxdiv=None).run(**controls)
assert reference.converged and teacher_state.converged
sample = training.Sample(reference, energy=teacher_state.e_tot)

for mode in ('fixed_density', 'explicit', 'implicit'):
    trainer = training.Trainer(functional, params=params)
    trainer.mode = mode
    trainer.learning_rate = .002
    trainer.loss = {'energy': {'mse': 1.}}
    trainer.scf = dict(max_cycle=80, damping=.1, conv_tol_energy=1e-12,
                       conv_tol_density=1e-10, conv_tol_grad=1e-9)
    trainer.adjoint = dict(tolerance=1e-10, max_iter=60)
    trainer.run(sample, steps=3)
    before, after = trainer.history['loss'][0], trainer.history['loss'][-1]
    assert all(trainer.history['update_accepted'][1:]) and after < before
    print(mode, 'MSE (Eh^2):', before, '->', after)

# The independent sample retains its initial density after the facade changes.
# Static q=0 DFPT is a separate calculation, not a Trainer response target yet.
reference.run(params=trainer.params)
potential = jnp.broadcast_to(jnp.diag(jnp.array([.1, -.1])), reference.inputs.hcore.shape)
response = dfpt.DFPT(reference).response(potential=potential)
assert response.converged
print('Response electron-number derivative:', float(response.values['electron_number']))

# Measured on jy_gpu_1, CPU float64, JAX 0.10.0, seed 0, three Adam steps:
# fixed_density MSE (Eh^2): 8.03967805450204e-05 -> 4.1388056248653946e-05
# explicit MSE (Eh^2): 8.039678062157627e-05 -> 4.14214479636731e-05
# implicit MSE (Eh^2): 8.039678062157627e-05 -> 4.142144694063198e-05
# Response electron-number derivative: -2.7755575615628914e-17
