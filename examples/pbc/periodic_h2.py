"""Reproducible periodic HF/DFT and q=0 excitation calculation (atomic units)."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import jax
import numpy as np
jax.config.update('jax_enable_x64',True)
from gradscf.pbc import gto,scf,dft,tdscf


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xc',default='hf',choices=['hf','svwn','pbe','pbe0'])
    parser.add_argument('--kmesh',nargs=3,type=int,default=[1,1,1])
    parser.add_argument('--mesh',type=int,default=31)
    parser.add_argument('--unrestricted',action='store_true')
    parser.add_argument('--response',choices=['none','tda','full'],default='tda')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();start=time.perf_counter()
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6,unit='Bohr',
        basis='gth-szv',pseudo='gth-pade',mesh=(args.mesh,)*3,precision=1e-10)
    kpts=cell.make_kpts(args.kmesh);multi=len(kpts)>1
    name=('K' if multi else '')+('U' if args.unrestricted else 'R')+('HF' if args.xc=='hf' else 'KS')
    cls=getattr(scf if args.xc=='hf' else dft,name)
    mf=cls(cell,kpts=kpts,xc=args.xc).run()
    output=dict(method=name,xc=args.xc,energy_hartree=float(mf.e_tot),converged=mf.converged,
        cycles=mf.cycles,electrons=cell.nelectron,mesh=cell.mesh,kpoints_bohr_inverse=np.asarray(kpts).tolist(),
        basis=cell.basis,pseudo=cell.pseudo,exxdiv=mf.exxdiv,
        lattice_bohr=np.asarray(cell.lattice).tolist(),coordinates_bohr=np.asarray(cell.coords).tolist(),
        hostname=platform.node(),backend=jax.default_backend(),dtype='float64/complex128',
        python=platform.python_version(),jax=jax.__version__,command=sys.argv)
    if args.response!='none':
        td=(tdscf.TDA if args.response=='tda' else tdscf.TDDFT)(mf,nstates=1)
        energy,_=td.kernel()
        output.update(response=args.response,excitation_hartree=np.asarray(energy).tolist(),
                      response_converged=np.asarray(td.converged).tolist())
    output['elapsed_seconds']=time.perf_counter()-start
    text=json.dumps(output,indent=2)+'\n'
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    return 0 if output['converged'] and all(output.get('response_converged',[True])) else 1


if __name__=='__main__':sys.exit(main())
