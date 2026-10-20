"""A four-layer MLP: AO density matrix -> total XC energy.

H2/6-31G*, CPU float64. Four Dense layers mean three hidden layers and one
scalar output layer. The network predicts the entire XC energy in Hartree;
there is no additional XC baseline or grid integration of its output.

This fixed-basis example has synthetic teacher targets. A flattened AO matrix
is basis/order dependent and the MLP imposes no size consistency or exact XC
constraints. It demonstrates the external-functional interface, not accuracy.

Run: PYTHONPATH=src JAX_PLATFORMS=cpu python examples/training/density_matrix_mlp.py
"""
import jax
import jax.numpy as jnp
from flax import linen as nn

jax.config.update('jax_enable_x64', True)

from gradscf import gto, scf, dft
from gradscf.model.neural_xc import training


# The input is the CURRENT spin-summed AO density matrix, supplied by GradSCF.
def make_inputs(state):
    return state.total_density_matrix


class XCNetwork(nn.Module):
    @nn.compact
    def __call__(self, density_matrix):
        x = density_matrix.reshape(-1)
        for width in (16, 16, 16):
            x = nn.tanh(nn.Dense(width, dtype=jnp.float64, param_dtype=jnp.float64)(x))
        x = nn.Dense(1, dtype=jnp.float64, param_dtype=jnp.float64)(x)
        return .05 * x[0]  # Scalar XC energy in Ha; a small initial energy scale.


model = XCNetwork()
functional = dft.custom_xc(input_fn=make_inputs, energy_fn=model.apply,
                          init_fn=model.init)

if __name__ == '__main__':
    # GradSCF supplies integrals and an HF initial density. Learned SCF then uses
    # only the MLP for XC; the HF exchange energy is not retained as a baseline.
    mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='6-31g*')
    mf = scf.RHF(mol, grids_level=0, integral_backend='native').run()
    reference = mf.to_reference()
    dm = mf.make_rdm1()
    params = model.init(jax.random.PRNGKey(0), dm)
    print('Density-matrix shape:', dm.shape)
    print('Initial XC energy / Ha:', float(functional.energy_from_molecule(params, reference)))
    print('XC potential shape:', functional.potential(params, reference, dm).shape)
    delta_dm = jnp.eye(dm.shape[0])
    response = functional.kernel_action(params, reference, dm, delta_dm)
    print('XC kernel action norm:', float(jnp.linalg.norm(response)))

    # Both energy and density targets come from another parameter set, solved
    # self-consistently. Replace them with reference labels for real training.
    trainer = training.Trainer(functional, params=params)
    trainer.mode = 'implicit'
    trainer.loss = {'energy': {'mse': 1.}, 'density': {'mse': 1.}}
    trainer.learning_rate = .002
    trainer.scf = dict(max_cycle=80, damping=.2, conv_tol_energy=1e-11,
                       conv_tol_density=1e-9, eigenvalue_jitter=0.)
    trainer.adjoint = dict(tolerance=1e-9, max_iter=40)
    teacher = jax.tree.map(lambda x: x+.02, params)
    target_energy, target_state = trainer.predict(mf, params=teacher)
    data = [training.Sample(mf,
        energy=jax.lax.stop_gradient(target_energy),
        density=jax.lax.stop_gradient(training.density_on_grid(target_state)))]

    trainer.run(data, steps=5)
    before, after = trainer.history['loss'][0], trainer.history['loss'][-1]
    print('Training loss:', before, '->', after)
    print('Accepted updates:', trainer.history['update_accepted'][1:])
    print('SCF converged:', trainer.history['scf_converged'][1:])
    if not all(trainer.history['update_accepted'][1:]) or not jnp.isfinite(after) or after >= before:
        raise RuntimeError('Expected five valid updates and a finite loss reduction')

# Measured output: CPU float64, JAX 0.8.1, random seed 0.
# Density-matrix shape: (4, 4)
# Initial XC energy / Ha: -0.0078110503267572184
# XC potential shape: (4, 4)
# XC kernel action norm: 0.01573187494899862
# Training loss: 2.15340029562302e-06 -> 1.8271876314836187e-06
# Accepted updates: [True, True, True, True, True]
# SCF converged: [True, True, True, True, True]
