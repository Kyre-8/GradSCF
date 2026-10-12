"""Force-loss mixed derivatives also cross the orbital-minimization boundary."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import integrals
from gradscf.scf import minimize_uks_from_integrals, nuclear_repulsion_energy
from gradscf.model.neural_xc.training import energy_and_forces, make_force_loss_and_grad


@pytest.mark.parametrize('mode',['implicit','unrolled'])
def test_native_orbital_force_loss_parameter_gradient(mode):
    topology,p=integrals.prepare_basis('H 0 0 0; H 0 0 .8',basis='sto-3g')
    plan=integrals.make_plan(topology,backend='native')
    s=np.asarray(plan.evaluate('overlap',p))
    seed=np.linalg.inv(np.linalg.cholesky(s).T)
    occ=np.array([[1.,0.],[1.,0.]])
    def energy(theta,r):
        moving=replace(p,centers=r,nuclear_coords=r)
        h=plan.evaluate('kinetic',moving)+plan.evaluate('nuclear',moving)
        h=h+theta*jnp.diag(jnp.array([1.,-1.]))
        result=minimize_uks_from_integrals(overlap=plan.evaluate('overlap',moving),hcore=h,
            eri=plan.evaluate('eri',moving),nuclear_repulsion=nuclear_repulsion_energy(r,jnp.array([1.,1.])),
            ao=jnp.zeros((0,2)),ao_deriv1=jnp.zeros((4,0,2)),grid_weights=jnp.zeros(0),
            mo_coeff=np.stack([seed,seed]),mo_occ=occ,gradient_mode=mode,
            orthonormalize_initial=True,max_iterations=60,gradient_tolerance=1e-10)
        return result.total_energy
    r=p.nuclear_coords;theta=jnp.array(.2)
    predict=jax.jit(lambda t:energy_and_forces(energy,t,r))
    target=jax.lax.stop_gradient(predict(jnp.array(.25)).forces)
    value_grad=jax.jit(make_force_loss_and_grad(energy))
    loss,gradient=value_grad(theta,r,target)
    fn=jax.jit(lambda t:.5*jnp.mean((predict(t).forces-target)**2))
    step=1e-4
    finite_difference=(fn(theta+step)-fn(theta-step))/(2*step)
    assert float(loss)>0
    assert abs(float(finite_difference))>1e-6
    np.testing.assert_allclose(gradient,finite_difference,atol=2e-7,rtol=1e-4)
