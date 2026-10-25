"""An oblique spectral projector through a non-Hermitian eigenvalue crossing.

The parameter and matrices are dimensionless. AD acts on a complete isolated
rank-two cluster; the individual sorted eigenvalues have no derivative at zero.
No SCF or molecular geometry response is implied by this small matrix example.
"""
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from gradscf.solvers import (EigenResponseConfig, LinearSolverConfig,
                             NonHermitianSolverConfig, solve_nonhermitian)


config = NonHermitianSolverConfig(method='davidson', nroots=2, max_space=4,
                                  atol=1e-11, seed=42)
response = EigenResponseConfig(target='subspace', linear_config=LinearSolverConfig(
    method='direct', max_dense=32, rtol=1e-11, atol=1e-12))
probe = jnp.array([.2, .4, -.3, .5])
reference = jnp.array([[1., .3, .2, 0.], [0., 1., 0., -.1],
                       [0., 0., 1., .4], [0., 0., 0., 1.]])
generator = jnp.array([[0., .1, .2, -.1], [.2, 0., -.1, .3],
                       [.3, -.2, 0., .1], [-.1, .2, .1, 0.]])


def states(parameter):
    change = reference @ (jnp.eye(4)+parameter*generator)
    values = jnp.array([-1.+.6*parameter+.2*parameter**2,
                       -1.-.7*parameter+.3*parameter**2, 1.5, 3.])
    matrix = change @ (values[:, None]*jnp.linalg.inv(change))
    return solve_nonhermitian(matrix, config=config, response=response,
                              probes=probe, max_memory=32.)


def observable(parameter):
    result = states(parameter)
    return probe @ result.projection + .1*result.eigenvalue_sum


result = jax.jit(states)(0.)
assert bool(result.response_valid)
# Raw eigenvectors are stopped diagnostics. Only these invariants carry AD.
assert result.values is None and result.right_vectors is None
value, derivative = jax.jit(jax.value_and_grad(observable))(0.)
curvature = jax.jit(lambda t: jax.jvp(jax.grad(observable), (t,), (1.,))[1])(0.)
forward = jax.jit(lambda t: jax.jvp(observable, (t,), (1.,))[1])(0.)
np.testing.assert_allclose(forward, derivative, atol=2e-10, rtol=0.)
trace = lambda t: states(t).eigenvalue_sum
np.testing.assert_allclose(jax.jit(jax.grad(trace))(0.), -.1, atol=2e-10, rtol=0.)
np.testing.assert_allclose(jax.jit(jax.grad(jax.grad(trace)))(0.), 1., atol=2e-9, rtol=0.)
for step in (3e-4, 1e-4):
    difference = (observable(step)-observable(-step))/(2*step)
    second = (jax.grad(observable)(step)-jax.grad(observable)(-step))/(2*step)
    np.testing.assert_allclose(difference, derivative, atol=2e-8, rtol=0.)
    np.testing.assert_allclose(second, curvature, atol=2e-8, rtol=0.)
print('Cluster trace:', float(result.eigenvalue_sum))
print('Observable:', float(value))
print('JVP / VJP:', float(forward), float(derivative))
print('Hessian-vector product:', float(curvature))
print('External boundary gap:', float(result.boundary_gap))

# Delta n006, CPU/float64, JAX 0.10.0:
# Cluster trace: -2.0
# Observable: 0.03999999999999998
# JVP / VJP: -0.03648000000000002 -0.03647999999999999
# Hessian-vector product: 0.025991999999999973
# External boundary gap: 2.500000000000001
