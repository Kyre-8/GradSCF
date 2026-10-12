"""Force-supervision regression: native H2 integrals and a tiny neural XC energy.

This finite, co-moving quadrature is a derivative test, not an accurate DFT grid.
Run with JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src python <this-file>.
Coordinates are Bohr; energies Hartree; forces Hartree/Bohr.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular
import numpy as np
import optax

from gradscf import integrals
from gradscf.scf import (
    DifferentiableSCF, DifferentiableSCFConfig, SCFDifferentiationConfig,
    QuadratureGrid, RestrictedMolecule, nuclear_repulsion_energy,
)
from gradscf.model.neural_xc.training import energy_and_forces, make_force_loss_and_grad


class TinyNeuralXC:
    """A trainable 1-3-1 tanh network defining a variational density functional."""
    def scf_xc_energy_and_alpha_for_density(self, params, molecule, density):
        rho = jnp.einsum('pq,gp,gq->g', density, molecule.ao, molecule.ao)
        hidden = jnp.tanh(jnp.log1p(rho)[:,None]*params['w']+params['b'])
        exc = .05*jnp.sum(molecule.grid.weights*rho*(hidden@params['v']))
        return exc, jnp.asarray(1.,density.dtype)  # HF plus the learned correction.


def make_experiment(mode='implicit', *, basis_name='3-21g', max_cycle=40):
    atom='H 0 0 -.38; H .06 .03 .38'
    topology, basis_parameters = integrals.prepare_basis(atom,basis=basis_name)
    plan=integrals.make_plan(topology,backend='native')
    basis=integrals.basis_from_spec(atom,basis=basis_name,precompute_eri_groups=False)
    coordinates=basis_parameters.nuclear_coords
    owners=np.argmin(np.linalg.norm(np.asarray(basis_parameters.centers)[:,None,:]
        -np.asarray(coordinates)[None,:,:],axis=2),axis=1)
    ao_owners=np.argmin(np.linalg.norm(np.stack([np.asarray(ao.center) for ao in basis.aos])[:,None,:]
        -np.asarray(coordinates)[None,:,:],axis=2),axis=1)
    s0=np.asarray(plan.evaluate('overlap',basis_parameters))
    h0=np.asarray(plan.evaluate('kinetic',basis_parameters)+plan.evaluate('nuclear',basis_parameters))
    x=np.linalg.inv(np.linalg.cholesky(s0).T)
    _,u=np.linalg.eigh(x.T@h0@x)
    seed=jnp.asarray(x@u)
    n=len(seed)
    occ=jnp.zeros((2,n)).at[:,0].set(1.)
    offsets=jnp.asarray([[-.8,.3,-.6],[.4,-.7,.5],[.6,.5,-.2],[-.3,-.4,.8],
                         [.2,.8,.4],[-.7,-.2,-.9],[.9,.1,.7],[-.2,.4,-.3]])
    weights=jnp.linspace(.12,.24,len(offsets))
    functional=TinyNeuralXC()
    solver=DifferentiableSCF(DifferentiableSCFConfig(
        mode='self_consistent',max_cycle=max_cycle,damping=0.,conv_tol_energy=1e-13,
        conv_tol_density=1e-12,eigenvalue_jitter=0.,
        differentiation=SCFDifferentiationConfig(mode=mode,tolerance=1e-11,max_iter=20,
                                                  require_converged=True)))

    def state(params, coords):
        moving=replace(basis_parameters,nuclear_coords=coords,centers=coords[owners])
        s=plan.evaluate('overlap',moving)
        h=plan.evaluate('kinetic',moving)+plan.evaluate('nuclear',moving)
        eri=plan.evaluate('eri',moving)
        enuc=nuclear_repulsion_energy(coords,jnp.asarray(topology.nuclear_charges))
        grid_coords=coords.mean(axis=0)+offsets
        moving_basis=integrals.CartesianBasis(
            tuple(replace(ao,center=coords[owner]) for ao,owner in zip(basis.aos,ao_owners)),
            precompute_eri_groups=False,atom_coords=coords,atom_charges=jnp.asarray(topology.nuclear_charges))
        ao=integrals.evaluate_cartesian_ao(moving_basis,grid_coords)
        lower=jnp.linalg.cholesky(seed.T@s@seed)
        coeff=solve_triangular(lower,seed.T,lower=True).T
        half=coeff[:,:1]@coeff[:,:1].T
        molecule=RestrictedMolecule(ao=ao,grid=QuadratureGrid(weights=weights,coords=grid_coords),
            dipole_integrals=jnp.zeros((3,n,n)),rep_tensor=eri,mo_coeff=jnp.stack([coeff,coeff]),
            mo_occ=occ,mo_energy=jnp.stack([jnp.diag(coeff.T@h@coeff)]*2),
            rdm1=jnp.stack([half,half]),h1e=h,nuclear_repulsion=enuc,overlap_matrix=s,nocc=1,
            atom_coords=coords,atom_charges=jnp.asarray(topology.nuclear_charges),exact_exchange_fraction=1.)
        result,info=solver.run(molecule,functional,params)
        density=result.rdm1.sum(axis=0)
        j=jnp.einsum('pqrs,rs->pq',eri,density)
        k=jnp.einsum('prqs,rs->pq',eri,density)
        exc,_=functional.scf_xc_energy_and_alpha_for_density(params,result,density)
        energy=jnp.sum(density*(h+.5*j-.25*k))+exc+enuc
        return energy,info.converged,info.cycles

    params={'w':jnp.array([.22,-.31,.17]),'b':jnp.array([.13,-.08,.19]),'v':jnp.array([.3,-.2,.25])}
    return dict(energy=lambda p,r:state(p,r)[0],state=state,params=params,coordinates=coordinates,
                mode=mode,basis=basis_name)


def validate_and_train(mode='implicit'):
    started=time.perf_counter()
    experiment=make_experiment(mode)
    energy,params,r=experiment['energy'],experiment['params'],experiment['coordinates']
    predictor=jax.jit(lambda p,r:energy_and_forces(energy,p,r))
    vector,unflatten=ravel_pytree(params)
    teacher=unflatten(vector+jnp.linspace(-.06,.08,len(vector)))
    target=jax.lax.stop_gradient(predictor(teacher,r).forces)
    prediction=predictor(params,r)
    objective=jax.jit(make_force_loss_and_grad(energy))
    before,grad=objective(params,r,target)
    flat_grad,_=ravel_pytree(grad)
    h=1e-4
    scalar_loss=jax.jit(lambda v:.5*jnp.mean((predictor(unflatten(v),r).forces-target)**2))
    fd=jnp.stack([(scalar_loss(vector+h*d)-scalar_loss(vector-h*d))/(2*h) for d in jnp.eye(len(vector))])
    force_fd=[]
    forward=jax.jit(energy)
    for d in jnp.eye(r.size).reshape((-1,)+r.shape):
        force_fd.append(-(forward(params,r+h*d)-forward(params,r-h*d))/(2*h))
    force_fd=jnp.stack(force_fd).reshape(r.shape)
    optimizer=optax.adam(.01)
    updates,_=optimizer.update(grad,optimizer.init(params),params)
    updated=optax.apply_updates(params,updates)
    after=scalar_loss(ravel_pytree(updated)[0])
    _,converged,cycles=jax.jit(experiment['state'])(params,r)
    summary=dict(mode=mode,basis=experiment['basis'],backend='native',dtype='float64',
        energy=float(prediction.energy),forces=np.asarray(prediction.forces).tolist(),
        coordinates_bohr=np.asarray(r).tolist(),target_forces=np.asarray(target).tolist(),
        force_units='Hartree/Bohr',converged=bool(converged),cycles=int(cycles),
        parameter_gradient=np.asarray(flat_grad).tolist(),parameter_fd=np.asarray(fd).tolist(),
        max_parameter_fd_error=float(jnp.max(jnp.abs(flat_grad-fd))),
        max_force_fd_error=float(jnp.max(jnp.abs(prediction.forces-force_fd))),
        force_sum=np.asarray(prediction.forces.sum(axis=0)).tolist(),
        parameter_gradient_norm=float(jnp.linalg.norm(flat_grad)),loss_before=float(before),loss_after=float(after),
        parameter_update_norm=float(jnp.linalg.norm(ravel_pytree(updated)[0]-vector)),
        finite_difference_step=h,elapsed_seconds=time.perf_counter()-started,jax=jax.__version__)
    print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['implicit','explicit'],default='implicit')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=validate_and_train(args.mode)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
