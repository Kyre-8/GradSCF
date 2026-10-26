"""Localize large-ERI failures before SCF: native samples, layouts and J/K."""
import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto
from gradscf import integrals
from gradscf.integrals.basis.contraction import primitive_basis
from gradscf.model.nnao import prepare_direct_basis
from gradscf.scf.rks import _build_jk


def diagnose(geometry,basis,output):
    start=time.perf_counter();data=json.loads(Path(geometry).read_text())
    atom=list(zip(data['symbols'],data['coords_angstrom']))
    if basis=='primitive442':
        layout=prepare_direct_basis(atom,basis_family='szp442_direct',core_primitives=6)
        top,params=primitive_basis(layout.topology,layout.parameters)
        shells=layout.atom_shells(params)
    else:
        top,params=integrals.prepare_basis(atom,basis,cart=False)
        shells=None
    labels=[f'{s}{i}' for i,s in enumerate(data['symbols'])]
    mol=gto.M(atom=list(zip(labels,data['coords_angstrom'])),basis=dict(zip(labels,shells)) if shells else basis,
              cart=False,unit='Angstrom',spin=0,verbose=0)
    assert mol.nao_nr()==top.nao
    print('Building native ERI',basis,'nao',top.nao,flush=True)
    eri=integrals.make_plan(top).evaluate('eri',params);eri.block_until_ready()
    host=np.asarray(eri);n=top.nao
    print('ERI ready',time.perf_counter()-start,'seconds',host.shape,host.strides,flush=True)
    rng=np.random.default_rng(31)
    indices=rng.integers(0,n,size=(300,4))
    indices=np.vstack((indices,[[0,0,0,0],[n-1,n-1,n-1,n-1],[0,n-1,0,n-1]]))
    loc=mol.ao_loc_nr();cache={};actual=[];reference=[];symmetry=[]
    for idx in indices:
        owners=np.searchsorted(loc,idx,side='right')-1
        key=tuple(owners)
        if key not in cache:cache[key]=mol.intor_by_shell('int2e_sph',key)
        actual.append(host[tuple(idx)])
        reference.append(cache[key][tuple(idx-loc[owners])])
        symmetry.append(host[tuple(idx)]-host[tuple(idx[[2,3,0,1]])])
    report=dict(basis=basis,nao=n,eri_size=int(eri.size),strides=host.strides,
                native_sample_error=float(np.max(np.abs(np.asarray(actual)-reference))),
                permutation_sample_error=float(np.max(np.abs(symmetry))))
    print(report,flush=True)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    v=rng.normal(size=(n,3));d=v@v.T/n
    j,k=jax.jit(_build_jk)(eri,jnp.asarray(d));j.block_until_ready();k.block_until_ready()
    picks=[0,n//2,n-1]
    jr=np.stack([np.einsum('qrs,rs->q',host[p],d) for p in picks])
    kr=np.stack([np.einsum('rqs,rs->q',host[p],d) for p in picks])
    report.update(j_error=float(np.max(np.abs(np.asarray(j)[picks]-jr))),
                  k_error=float(np.max(np.abs(np.asarray(k)[picks]-kr))),
                  j_norm=float(np.linalg.norm(j)),k_norm=float(np.linalg.norm(k)),
                  elapsed_seconds=time.perf_counter()-start)
    print(report,flush=True);output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('geometry');parser.add_argument('--basis',default='primitive442')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();diagnose(args.geometry,args.basis,args.output)
