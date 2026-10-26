"""Single-molecule NNAO coefficient optimization."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np

from ..training.objective import BasisRHF, embedded_outputs

def _run_adam_steps(initial,evaluate,record,*,steps,initial_learning_rate,
                    final_learning_rate,gradient_clip_norm,max_retries=12):
    """Run accepted Adam steps, shrinking the learning rate at invalid trials."""
    import optax
    if steps<1:raise ValueError('Adam steps must be positive.')
    if min(initial_learning_rate,final_learning_rate,gradient_clip_norm)<=0:
        raise ValueError('Adam learning rates and gradient clip norm must be positive.')
    if max_retries<1:raise ValueError('Adam max_retries must be positive.')
    vector=jnp.asarray(initial)
    transform=optax.chain(optax.clip_by_global_norm(float(gradient_clip_norm)),optax.scale_by_adam())
    state=transform.init(vector)
    energy,gradient,_,_=evaluate(np.asarray(vector));evaluations=1;rejected_total=0
    record(np.asarray(vector),learning_rate=0.,rejected_trials=0)
    multiplier=1.
    for step in range(steps):
        updates,next_state=transform.update(jnp.asarray(gradient),state,vector)
        fraction=step/max(steps-1,1)
        base_lr=initial_learning_rate*(final_learning_rate/initial_learning_rate)**fraction
        last_error=None
        for retry in range(max_retries):
            learning_rate=base_lr*multiplier
            candidate=vector-learning_rate*updates
            try:
                candidate_result=evaluate(np.asarray(candidate));evaluations+=1
            except RuntimeError as error:
                if str(error)!='Nonfinite energy or gradient':raise
                evaluations+=1;rejected_total+=1;multiplier*=.5;last_error=error
                continue
            vector=candidate;state=next_state
            energy,gradient=candidate_result[:2]
            record(np.asarray(vector),learning_rate=float(learning_rate),rejected_trials=retry)
            break
        else:
            raise RuntimeError('Adam failed to find a finite trial point.') from last_error
    return np.asarray(vector),dict(iterations=steps,evaluations=evaluations,
                                   rejected_trials=rejected_total)

def optimize(*,bond=1.09,maxiter=80,basis_family='szp442_direct',core_primitives=None,
             initial_summary=None,initial_checkpoint=None,geometry=None,jk_backend='direct',auxbasis='def2-universal-jkfit',
             integral_cache=None,optimizer='lbfgs',learning_rate=1e-4,
             final_learning_rate=1e-5,gradient_clip_norm=1.,
             output_dir=Path('artifacts/methane-nnao'),implicit_tolerance=1e-9,
             scf_rescue_level_shift=0.0):
    from flax import nnx,serialization
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import minimize
    from gradscf.model.nnao import MACEBasisModel,build_graph
    jax.config.update('jax_enable_x64',True)
    start=time.perf_counter();output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    from gradscf.data.molecule import atomic_number
    experiment=BasisRHF(bond,basis_family,core_primitives,geometry,jk_backend,auxbasis,integral_cache,
                         implicit_tolerance=implicit_tolerance,scf_rescue_level_shift=scf_rescue_level_shift)
    atomic_numbers=[atomic_number(s) for s in experiment.symbols]
    model_config=dict(elements=tuple(sorted(set(atomic_numbers))),channels=8,num_interactions=2,max_ell=1,correlation=2,zero_init=True,basis_family=basis_family)
    model=MACEBasisModel(**model_config,rngs=nnx.Rngs(0))
    warm=None
    if initial_summary is not None:
        warm=json.loads(Path(initial_summary).read_text())
        if basis_family not in {'szp3_direct','szp442_direct','szp663_direct'}:
            raise ValueError('Warm start requires an all-electron direct basis family.')
        assert tuple(warm['symbols'])==experiment.symbols and not warm['cartesian'] and not warm['ecp']
        assert warm['charge']==warm['spin']==0
        np.testing.assert_allclose(warm['coords_angstrom'],experiment.coords,atol=1e-14,rtol=0)
        outputs=embedded_outputs(experiment.layout,warm['final_basis'])
        from gradscf.data.molecule import atomic_number
        bias=np.asarray(model.head_bias[...]).copy();seen={}
        for i,symbol in enumerate(experiment.symbols):
            z=atomic_number(symbol);vector=np.asarray(outputs[i]).ravel()
            if z in seen:np.testing.assert_allclose(vector,seen[z],atol=1e-10,rtol=0)
            seen[z]=vector;bias[model.elements.index(z)]=vector
        model.head_bias[...]=jnp.asarray(bias)
    graph=build_graph(atomic_numbers,experiment.coords,element_order=model.elements)
    graphdef,parameters,other=nnx.split(model,nnx.Param,...)
    initial,unravel=ravel_pytree(parameters)
    if initial_checkpoint is not None:
        with np.load(initial_checkpoint,allow_pickle=False) as checkpoint:
            key='parameters' if 'parameters' in checkpoint else 'final_parameters'
            restored=np.asarray(checkpoint[key])
        if restored.shape!=initial.shape:raise ValueError('Checkpoint parameter shape does not match the model.')
        initial=jnp.asarray(restored,dtype=initial.dtype)
    initial_parameter_tree=unravel(initial)
    @jax.jit
    def predict(vector):
        return nnx.merge(graphdef,unravel(vector),other)(graph)
    @jax.jit
    def backward(vector,cotangent):
        return jax.vjp(predict,vector)[1](cotangent)[0]
    cache={};history=[]
    def evaluate(vector):
        key=np.asarray(vector).tobytes()
        if key not in cache:
            outputs=predict(jnp.asarray(vector))
            energy,go,info=experiment.evaluate(outputs)
            gradient=np.asarray(backward(jnp.asarray(vector),go))
            row=dict(energy_hartree=energy,max_parameter_gradient=float(np.max(np.abs(gradient))),
                     max_output_gradient=float(np.max(np.abs(go))),
                     max_abs_output=float(np.max(np.abs(np.asarray(outputs)))),**info)
            if basis_family=='szp3':row['max_tanh_output']=float(np.max(np.abs(np.tanh(np.asarray(outputs)))))
            cache[key]=(energy,gradient,row,np.asarray(outputs))
        return cache[key]
    def record(vector,**metadata):
        _,_,row,outputs=evaluate(vector)
        row=dict(iteration=len(history),**row,**metadata);history.append(row)
        (output_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        temporary=output_dir/'checkpoint-latest.tmp.npz'
        np.savez(temporary,parameters=np.asarray(vector),outputs=outputs,iteration=row['iteration'])
        os.replace(temporary,output_dir/'checkpoint-latest.npz')
        print(f"iter={row['iteration']:3d} E={row['energy_hartree']:.12f} "
              f"max|dE/dtheta|={row['max_parameter_gradient']:.3e} SCF={row['scf_cycles']} "
              f"Smin={row['min_overlap_eigenvalue']:.3e}",flush=True)
    def finite_difference(vector):
        energy,g,_,_=evaluate(vector)
        direction=g/np.linalg.norm(g) if np.linalg.norm(g)>1e-12 else np.ones_like(g)/np.sqrt(g.size)
        step=1e-4
        def e(scale):return evaluate(np.asarray(vector)+scale*direction)[0]
        fd=(8*(e(step)-e(-step))-e(2*step)+e(-2*step))/(12*step)
        ad=float(g@direction)
        np.testing.assert_allclose(ad,fd,atol=2e-6,rtol=2e-5)
        return dict(ad=ad,finite_difference=fd,absolute_error=abs(ad-fd),step=step)
    print(f'{experiment.molecule} RHF {basis_family}: AO={experiment.layout.topology.nao}, '
          f'primitive AO={experiment.primitive_topology.nao}, MACE parameters={initial.size}',flush=True)
    same_hamiltonian = (warm is not None and warm.get('jk_backend')==jk_backend
                        and (jk_backend!='df' or warm.get('auxbasis')==auxbasis))
    if optimizer=='lbfgs':
        record(initial)
        if same_hamiltonian:
            np.testing.assert_allclose(history[0]['energy_hartree'],warm['final_energy_hartree'],atol=1e-9,rtol=0)
        initial_fd=finite_difference(np.asarray(initial))
        print('Initial gradient check:',initial_fd,flush=True)
        result=minimize(lambda v:evaluate(v)[:2],np.asarray(initial),jac=True,method='L-BFGS-B',callback=record,
                        options=dict(maxiter=maxiter,gtol=5e-7,ftol=1e-14,maxls=30))
        final_vector=np.asarray(result.x);optimizer_success=bool(result.success);message=str(result.message)
        iterations=int(result.nit);evaluations=int(result.nfev);rejected_trials=0
        final_e,final_gradient,final_row,final_outputs=evaluate(final_vector)
        if history[-1]['energy_hartree']!=final_e:record(final_vector)
        final_fd=finite_difference(final_vector)
    elif optimizer=='adam':
        initial_fd=final_fd=None
        final_vector,stats=_run_adam_steps(
            np.asarray(initial),evaluate,record,steps=maxiter,
            initial_learning_rate=learning_rate,final_learning_rate=final_learning_rate,
            gradient_clip_norm=gradient_clip_norm,
        )
        final_e,final_gradient,final_row,final_outputs=evaluate(final_vector)
        optimizer_success=True;message=f'Completed {maxiter} Adam steps'
        iterations=stats['iterations'];evaluations=stats['evaluations'];rejected_trials=stats['rejected_trials']
    else:raise ValueError("optimizer must be 'lbfgs' or 'adam'.")
    def state_norm_difference(first,last):
        a=jax.tree_util.tree_leaves(nnx.to_pure_dict(first));b=jax.tree_util.tree_leaves(nnx.to_pure_dict(last))
        return float(np.sqrt(sum(float(jnp.sum((x-y)**2)) for x,y in zip(a,b))))
    final_parameters=unravel(jnp.asarray(final_vector))
    summary=dict(molecule=experiment.molecule,method='RHF',basis=basis_family,cartesian=experiment.layout.topology.cart,nelectron=experiment.nelectron,
        geometry_metadata=geometry,jk_backend=jk_backend,auxbasis=auxbasis if jk_backend=='df' else None,
        integral_cache=None if integral_cache is None else str(integral_cache),
        implicit_config=dict(tolerance=experiment.implicit_tolerance,max_iter=100,restart=40),
        scf_rescue_level_shift=experiment.scf_rescue_level_shift,
        rep_shape=None if experiment.rep is None else experiment.rep.shape,
        rep_bytes=0 if experiment.rep is None else int(experiment.rep.size*experiment.rep.dtype.itemsize),
        initial_summary=str(initial_summary) if initial_summary is not None else None,
        initial_summary_sha256=hashlib.sha256(Path(initial_summary).read_bytes()).hexdigest() if initial_summary is not None else None,
        initial_checkpoint=str(initial_checkpoint) if initial_checkpoint is not None else None,
        initial_checkpoint_sha256=hashlib.sha256(Path(initial_checkpoint).read_bytes()).hexdigest() if initial_checkpoint is not None else None,
        core_primitive_counts=[n for n,r in zip(experiment.layout.topology.primitive_counts,experiment.layout.roles) if r=='core'],
        ecp=[e.as_raw() if e else None for e in experiment.layout.ecps] if basis_family=='qvszps' else None,bond_angstrom=experiment.bond,
        symbols=experiment.symbols,coords_angstrom=experiment.coords.tolist(),charge=0,spin=0,
        optimized='all MACE and basis-head trainable parameters',fixed='geometry, primitive exponents and ECP' if basis_family=='qvszps' else 'geometry, primitive exponents, core and single-primitive polarization shells' if basis_family in {'szp442_direct','szp663_direct'} else 'geometry, primitive exponents, core and polarization shells',
        gradient='native fixed primitive '+('DF factors' if jk_backend=='df' else 'shell-direct J/K')+' + JAX contraction + implicit RHF density fixed point',
        optimizer='L-BFGS-B' if optimizer=='lbfgs' else 'Adam',optimizer_success=optimizer_success,message=message,
        learning_rate=learning_rate if optimizer=='adam' else None,
        final_learning_rate=final_learning_rate if optimizer=='adam' else None,
        gradient_clip_norm=gradient_clip_norm if optimizer=='adam' else None,
        rejected_trials=rejected_trials,iterations=iterations,evaluations=evaluations,parameter_count=int(initial.size),
        model_config=model_config,seed=0,initial_energy_hartree=history[0]['energy_hartree'],final_energy_hartree=final_e,
        energy_decrease_hartree=history[0]['energy_hartree']-final_e,initial_fd=initial_fd,final_fd=final_fd,
        backbone_parameter_change_norm=state_norm_difference(initial_parameter_tree.backbone,final_parameters.backbone),
        final=final_row,initial_basis=experiment.layout.atom_shells(experiment.layout.bind(predict(initial))),
        final_basis=experiment.layout.atom_shells(experiment.layout.bind(final_outputs)),
        hydrogen_output_symmetry_error=float(np.max(np.abs(final_outputs[1:]-final_outputs[1]))) if geometry is None else None,
        history=history,elapsed_seconds=time.perf_counter()-start,jax_version=jax.__version__,
        devices=[str(x) for x in jax.devices()],dtype='float64',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output_dir/'optimization.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    np.savez(output_dir/'checkpoint.npz',initial_parameters=np.asarray(initial),final_parameters=final_vector,
             initial_outputs=np.asarray(predict(initial)),final_outputs=final_outputs)
    (output_dir/'parameters.msgpack').write_bytes(serialization.to_bytes(nnx.to_pure_dict(final_parameters)))
    print('Final:',message,'energy decrease / Ha:',summary['energy_decrease_hartree'],flush=True)
    if final_fd is not None:print('Final gradient check:',final_fd,flush=True)
    print('Backbone parameter change:',summary['backbone_parameter_change_norm'],flush=True)
    return summary

def main(argv=None):
    jax.config.update("jax_enable_x64", True)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bond',type=float,default=1.09)
    parser.add_argument('--maxiter',type=int,default=80)
    parser.add_argument('--basis-family',choices=('szp663_direct','szp442_direct','szp3_direct','szp3','qvszps'),default='szp442_direct')
    parser.add_argument('--core-primitives',type=int,choices=(3,6),default=None)
    parser.add_argument('--initial-summary',type=Path,default=None,help='Initialize trainable output bias from saved nested contractions; backbone uses seed 0.')
    parser.add_argument('--initial-checkpoint',type=Path,default=None,help='Resume the complete flattened MACE parameter vector.')
    parser.add_argument('--geometry',type=Path,help='JSON with name, symbols and coords_angstrom; default is methane.')
    parser.add_argument('--jk-backend',choices=('direct','df'),default='direct')
    parser.add_argument('--auxbasis',default='def2-universal-jkfit')
    parser.add_argument('--integral-cache',type=Path,help='Load a CPU-prepared DF cache; required to keep native setup out of a GPU process.')
    parser.add_argument('--prepare-integral-cache',type=Path,help='Build a DF cache and exit without training.')
    parser.add_argument('--optimizer',choices=('lbfgs','adam'),default='lbfgs')
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--final-learning-rate',type=float,default=1e-5)
    parser.add_argument('--gradient-clip-norm',type=float,default=1.)
    parser.add_argument('--implicit-tolerance',type=float,default=1e-9)
    parser.add_argument('--scf-rescue-level-shift',type=float,default=0.0)
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/methane-nnao'))
    args=parser.parse_args(argv)
    geometry=json.loads(args.geometry.read_text()) if args.geometry else None
    if args.prepare_integral_cache is not None:
        experiment=BasisRHF(args.bond,args.basis_family,args.core_primitives,geometry,args.jk_backend,args.auxbasis,
                             implicit_tolerance=args.implicit_tolerance,
                             scf_rescue_level_shift=args.scf_rescue_level_shift)
        print(experiment.write_integral_cache(args.prepare_integral_cache),flush=True)
    else:
        optimize(bond=args.bond,maxiter=args.maxiter,basis_family=args.basis_family,core_primitives=args.core_primitives,
                 initial_summary=args.initial_summary,initial_checkpoint=args.initial_checkpoint,
                 geometry=geometry,jk_backend=args.jk_backend,
                 auxbasis=args.auxbasis,integral_cache=args.integral_cache,optimizer=args.optimizer,
                 learning_rate=args.learning_rate,final_learning_rate=args.final_learning_rate,
                 gradient_clip_norm=args.gradient_clip_norm,output_dir=args.output_dir,
                 implicit_tolerance=args.implicit_tolerance,
                 scf_rescue_level_shift=args.scf_rescue_level_shift)


if __name__ == "__main__":
    raise SystemExit(main())
