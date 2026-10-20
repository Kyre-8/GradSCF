"""An external Flax XC network, trained in all three existing modes.

H2/6-31G*, native CPU integrals, float64. The initial HF orbitals and grid
come from GradSCF; every learned SCF uses the external energy functional.
The network adds a correction to Dirac exchange; correlation is omitted.
Teacher-generated targets demonstrate AD/training, not chemical accuracy.
Run: PYTHONPATH=src JAX_PLATFORMS=cpu python examples/training/external_functional.py
"""
from time import perf_counter

import jax
import jax.numpy as jnp
from flax import linen as nn

jax.config.update('jax_enable_x64', True)

from gradscf import gto, scf, dft
from gradscf.model.neural_xc import training


# Both the feature schema and the architecture belong to this example.
def make_inputs(state):
    return {'x': jnp.log1p(state.rho[:, None]),
            'rho': state.rho, 'weights': state.weights}


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(4, dtype=jnp.float64, param_dtype=jnp.float64)(x))
        return nn.Dense(1, dtype=jnp.float64, param_dtype=jnp.float64)(x)[:, 0]


model = Network()


def energy(params, inputs):
    rho = inputs['rho']
    baseline = -.75*(3/jnp.pi)**(1/3)*jnp.maximum(rho, 1e-18)**(4/3)
    correction = .02*rho*model.apply(params, inputs['x'])
    return jnp.sum(inputs['weights']*(baseline+correction))


functional = dft.custom_xc(input_fn=make_inputs, energy_fn=energy,
    init_fn=lambda key, x: model.init(key, x['x']))
mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='6-31g*')
mf = scf.RHF(mol, grids_level=0, integral_backend='native').run()
reference = mf.to_reference()
params = functional.init_from_molecule(jax.random.PRNGKey(0), reference)
teacher = jax.tree.map(lambda x: x+.03, params)
scf_settings = dict(max_cycle=80, damping=.2, conv_tol_energy=1e-11,
                    conv_tol_density=1e-9, eigenvalue_jitter=0.)
adjoint_settings = dict(tolerance=1e-9, max_iter=40)
reference_trainer = training.Trainer(functional)
reference_trainer.mode = 'implicit'
reference_trainer.scf = scf_settings
reference_trainer.adjoint = adjoint_settings
target_energy, _ = reference_trainer.predict(mf, params=teacher)
data = [training.Sample(mf, energy=jax.lax.stop_gradient(target_energy))]
print('Self-consistent teacher energy / Ha:', float(target_energy))

for mode in ('fixed_density', 'explicit', 'implicit'):
    started = perf_counter()
    trainer = training.Trainer(functional, params=params)
    trainer.mode = mode
    trainer.learning_rate = .002
    trainer.scf = scf_settings
    trainer.adjoint = adjoint_settings
    trainer.run(data, steps=3)
    before, after = trainer.history['loss'][0], trainer.history['loss'][-1]
    print(mode, 'loss:', before, '->', after,
          'seconds:', round(perf_counter()-started, 2))
    if not jnp.isfinite(after) or after >= before:
        raise RuntimeError('Expected a finite loss reduction in this demonstration')

# Measured CPU float64 output (Apple M4 Pro, JAX 0.8.1):
# Self-consistent teacher energy / Ha: -1.0400905116545243
# fixed_density         4.1417802781e-06 -> 9.2387785597e-07
# self_consistent/explicit 5.6129071029e-06 -> 1.6771517640e-06
# self_consistent/implicit 5.6129071029e-06 -> 1.6771517640e-06
# Three Adam updates per mode; seed 0. Targets are synthetic.
