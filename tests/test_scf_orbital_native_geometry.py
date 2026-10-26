"""The SCF backward boundary composes with the native coordinate VJP."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.scf import nuclear_repulsion_energy
from gradscf.scf.orbital_optimization import minimize_uks_from_integrals


@pytest.mark.parametrize('mode', ['implicit', 'unrolled'])
def test_h2_native_coordinate_backward_matches_finite_difference(mode):
    from scipy.linalg import eigh
    topology, parameters = integrals.prepare_basis('H 0 0 0; H 0 0 .74', basis='3-21g')
    plan = integrals.make_plan(topology, backend='native')
    owners = np.argmin(np.linalg.norm(np.asarray(parameters.centers)[:,None,:]
        -np.asarray(parameters.nuclear_coords)[None,:,:],axis=2),axis=1)
    s = plan.evaluate('overlap',parameters)
    h = plan.evaluate('kinetic',parameters)+plan.evaluate('nuclear',parameters)
    _, c = eigh(np.asarray(h),np.asarray(s))
    seed = np.stack([c,c]); occ = np.zeros((2,len(c)));occ[:,:1]=1.

    def calculate(distance):
        coordinates = parameters.nuclear_coords.at[1,2].set(distance)
        moving = replace(parameters,nuclear_coords=coordinates,centers=coordinates[owners])
        overlap = plan.evaluate('overlap',moving)
        core = plan.evaluate('kinetic',moving)+plan.evaluate('nuclear',moving)
        eri = plan.evaluate('eri',moving)
        return minimize_uks_from_integrals(overlap=overlap,hcore=core,eri=eri,
            nuclear_repulsion=nuclear_repulsion_energy(coordinates,jnp.asarray(topology.nuclear_charges)),
            ao=jnp.zeros((0,len(c))),ao_deriv1=jnp.zeros((4,0,len(c))),grid_weights=jnp.zeros(0),
            mo_coeff=seed,mo_occ=occ,max_iterations=100,gradient_tolerance=1e-9,
            gradient_mode=mode,orthonormalize_initial=True)

    def observable(distance):
        result = calculate(distance)
        return result.total_energy + .1*result.density_matrix[0,0,1]

    distance, step = float(parameters.nuclear_coords[1,2]), 1e-4
    value = jax.jit(observable)
    actual = jax.jit(jax.grad(observable))(distance)
    finite_difference = (8*(value(distance+step)-value(distance-step))
                         -value(distance+2*step)+value(distance-2*step))/(12*step)
    result = calculate(distance)
    assert result.stationary
    np.testing.assert_allclose(actual,finite_difference,atol=3e-7,rtol=0)
    if mode == 'implicit':
        # Differentiate the converged orbitals as well as the native integrals.
        # A frozen-density second derivative would omit this orbital response.
        energy = lambda r: calculate(r).total_energy
        gradient = jax.jit(jax.grad(energy))
        curvature = jax.jit(jax.grad(jax.grad(energy)))(distance)
        fd_curvature = (8*(gradient(distance+step)-gradient(distance-step))
                        -gradient(distance+2*step)+gradient(distance-2*step))/(12*step)
        np.testing.assert_allclose(curvature,fd_curvature,atol=2e-6,rtol=0)
