"""Train an external four-layer XC network on a periodic H2 cell.

CPU float64, seed 0, GTH-SZV/GTH-Pade, a 6-Bohr cubic cell, shifted 2x1x1
k mesh, 9x9x9 FFT grid, kBT=0.1 Eh. This coarse synthetic teacher experiment
illustrates derivatives, not material accuracy. All three modes fit the same
teacher free energy; fixed_density freezes the initial electronic density.
The model and its input features are ordinary user-defined JAX functions.
"""
import jax
import jax.numpy as jnp
import optax
from gradscf import dft, dfpt
from gradscf.pbc import gto
from gradscf.pbc.scf import run_kpoint_scf
from gradscf.integrals.periodic.fft import build_inputs
from gradscf.scf import SCFDifferentiationConfig

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
inputs = build_inputs(cell, kpts=cell.make_kpts((2, 1, 1), shift=(.5, 0, 0)))


def solve(parameters, mode):
    return run_kpoint_scf(inputs, mesh=cell.mesh, nelec=cell.nelec,
        functional=functional, params=parameters, temperature=.1, exxdiv=None,
        max_cycle=80, damping=.1, conv_tol=1e-12, conv_tol_density=1e-10,
        differentiation=SCFDifferentiationConfig(mode=mode, tolerance=1e-10, max_iter=60))


reference = solve(params, 'implicit')
teacher = tuple((weight + .02, bias + .03) for weight, bias in params)
teacher_state = solve(teacher, 'implicit')
assert reference.converged and teacher_state.converged
target = jax.lax.stop_gradient(teacher_state.free_energy)
fixed_density = jax.lax.stop_gradient(reference.density_spin)
constant = reference.free_energy - functional.energy_for_density(params, inputs, fixed_density)

for mode in ('fixed_density', 'explicit', 'implicit'):
    def loss(parameters):
        if mode == 'fixed_density':
            prediction = constant + functional.energy_for_density(parameters, inputs, fixed_density)
            converged = True
        else:
            result = solve(parameters, mode)
            prediction, converged = result.free_energy, result.converged
        return (prediction - target)**2, converged

    value_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))
    optimizer = optax.adam(.002)
    fitted = params
    state = optimizer.init(fitted)
    before = float(loss(fitted)[0])
    for step in range(3):
        (value, converged), gradient = value_and_grad(fitted)
        assert converged and jnp.isfinite(value)
        updates, state = optimizer.update(gradient, state, fitted)
        fitted = optax.apply_updates(fitted, updates)
    after, converged = loss(fitted)
    assert converged and jnp.isfinite(after) and after < before
    print(mode, 'MSE (Eh^2):', before, '->', float(after))

# Build inside a differentiated function when fitting a response observable;
# the array factory retains both model parameters and relaxed SCF dependence.
result = solve(fitted, 'implicit')
problem = dfpt.periodic_problem(inputs, result, mesh=cell.mesh, nelec=cell.nelec,
    functional=functional, params=fitted, temperature=.1, exxdiv=None)
potential = jnp.broadcast_to(jnp.diag(jnp.array([.1, -.1])), inputs.hcore.shape)
response = problem.response(potential=potential)
assert response.converged
print('Response electron-number derivative:', float(response.values['electron_number']))

# Measured CPU float64 output, JAX 0.8.1, seed 0:
# fixed_density MSE (Eh^2): 8.052880096068146e-05 -> 4.148262531563322e-05
# explicit MSE (Eh^2): 8.052880096068146e-05 -> 4.1482626116580674e-05
# implicit MSE (Eh^2): 8.052880096068146e-05 -> 4.1482626116580674e-05
# Response electron-number derivative: 3.885780586188048e-16
