"""Large-system implicit coefficient gradient and independent RHF checks."""
from importlib import import_module
import argparse
import json
from pathlib import Path
import runpy
import time
import os
import platform
import resource
import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto,scf


def run(args):
    geometry=json.loads(Path(args.geometry).read_text());start=time.perf_counter()
    cls=vars(import_module("gradscf.model.nnao.training.objective"))['BasisRHF']
    print('Building primitive J/K backend',args.family,flush=True)
    ex=cls(geometry=geometry,basis_family=args.family,jk_backend=args.jk_backend,auxbasis=args.auxbasis)
    x=ex.layout.reference_outputs()
    energy,g,info=ex.evaluate(x)
    print('Initial implicit evaluation',energy,info,flush=True)
    rng=np.random.default_rng(73)
    direction=jnp.asarray(rng.normal(size=x.shape))
    mask=jnp.zeros_like(x)
    for atom,slot,n in zip(ex.layout.shell_atoms,ex.layout.slots,ex.layout.topology.primitive_counts):
        if slot>=0:mask=mask.at[atom,slot,:n].set(1.)
    direction=direction*mask;direction=direction/jnp.linalg.norm(direction)
    step=1e-4
    values={a:ex.evaluate(x+a*step*direction)[0] for a in [-2,-1,1,2]}
    fd=(8*(values[1]-values[-1])-values[2]+values[-2])/(12*step)
    ad=float(jnp.sum(g*direction))
    np.testing.assert_allclose(ad,fd,atol=2e-6,rtol=2e-5)
    labels=[f'{s}{i}' for i,s in enumerate(ex.symbols)]
    basis=dict(zip(labels,ex.layout.atom_shells(ex.layout.bind(x))))
    mol=gto.M(atom=list(zip(labels,ex.coords)),basis=basis,cart=False,unit='Angstrom',verbose=0)
    mf=scf.RHF(mol)
    if args.jk_backend=='df':mf=mf.density_fit(auxbasis=args.auxbasis)
    mf.init_guess='1e';mf.conv_tol=1e-12;mf.conv_tol_grad=1e-9;mf.max_cycle=200
    expected=float(mf.kernel());assert mf.converged
    np.testing.assert_allclose(energy,expected,atol=1e-8,rtol=0)
    result=dict(family=args.family,jk_backend=args.jk_backend,auxbasis=args.auxbasis,
        rep_shape=None if ex.rep is None else ex.rep.shape,
        rep_bytes=0 if ex.rep is None else int(ex.rep.size*ex.rep.dtype.itemsize),
        nao=ex.layout.topology.nao,primitive_nao=ex.primitive_topology.nao,
        gradscf_energy=energy,pyscf_energy=expected,energy_error=abs(energy-expected),
        gradient_ad=ad,gradient_fd=fd,gradient_error=abs(ad-fd),step=step,
        info=info,elapsed_seconds=time.perf_counter()-start,status='pass')
    if args.jk_backend=='df':
        def exact(v):
            mol.basis=dict(zip(labels,ex.layout.atom_shells(ex.layout.bind(v))))
            mol.build(False,False)
            ref=scf.RHF(mol);ref.init_guess='1e';ref.conv_tol=1e-12;ref.conv_tol_grad=1e-9;ref.max_cycle=200
            value=ref.kernel();assert ref.converged
            return float(value)
        exact_energy=exact(x)
        fd_exact=(exact(x+step*direction)-exact(x-step*direction))/(2*step)
        result.update(exact_energy=exact_energy,fitting_energy_error=abs(energy-exact_energy),
                      exact_gradient_fd=fd_exact,fitting_gradient_error=abs(ad-fd_exact))
        assert result['fitting_energy_error']<1e-3,result
        assert result['fitting_gradient_error']<1e-4,result
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.update(elapsed_seconds=time.perf_counter()-start,host=platform.node(),jax_version=jax.__version__,
                  peak_rss_mib=peak/(2**20 if platform.system()=='Darwin' else 1024),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('geometry');p.add_argument('--family',choices=['szp442_direct','szp663_direct'],default='szp442_direct')
    p.add_argument('--output',required=True)
    p.add_argument('--jk-backend',choices=['direct','df'],default='df')
    p.add_argument('--auxbasis',default='def2-universal-jkfit')
    run(p.parse_args())
