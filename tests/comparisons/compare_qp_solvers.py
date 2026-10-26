"""QP-only secant/Newton/hybrid timing, one fresh CPU process per case/method.

G/W and the mean-field reference are prepared outside the measured solve.
Peak RSS includes that preparation and compiler/runtime memory. It is not
isolated solver storage or total-GW throughput. All values are in Hartree.
"""
from pathlib import Path
import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time

CASES={
    'h2_sto3g':('H 0 0 0; H 0 0 .74','sto-3g',None),
    'water_sto3g':('O 0 0 0; H 0 -.757 .587; H 0 .757 .587','sto-3g',None),
    'lih_sto3g':('Li 0 0 0; H 0 0 1.6','sto-3g',None),
    'water_631g':('O 0 0 0; H 0 -.757 .587; H 0 .757 .587','6-31g','frontier'),
}


def measure(case,method):
    import jax
    import jax.numpy as jnp
    import numpy as np
    jax.config.update('jax_enable_x64',True)
    from gradscf import gto,scf,gw
    from gradscf.gw import g0w0
    from gradscf.gw.qp import qp_residual_batch
    from gradscf.solvers.nonlinear import ScalarRootConfig,solve_scalar_roots
    atom,basis,window=CASES[case]
    mf=scf.RHF(gto.M(atom=atom,basis=basis),conv_tol=1e-12,max_cycle=100).run()
    if not mf.converged:raise ArithmeticError('SCF not converged')
    nocc=int(np.count_nonzero(mf.mo_occ));nmo=len(mf.mo_energy)
    orbs=tuple(range(nmo)) if window is None else tuple(range(nocc-2,min(nocc+2,nmo)))
    context={}
    class Prepared(Exception):pass
    original=g0w0.solve_qp_batch
    def capture(e_mf,delta_v,shared,stacked,**kw):
        context.update(e_mf=e_mf,delta_v=delta_v,shared=shared,stacked=stacked,occupied=kw['occupied'])
        raise Prepared()
    # Capture the real driver's already assembled context, never duplicate
    # self-energy preparation or time an arbitrary proxy polynomial.
    g0w0.solve_qp_batch=capture
    try:
        gw.GW(mf,nw=100,eta=1e-3).run(orbs=orbs)
    except Prepared:pass
    finally:g0w0.solve_qp_batch=original
    if not context:raise RuntimeError('QP context capture failed')
    jax.block_until_ready(context)
    f=lambda w:qp_residual_batch(w,context['e_mf'],context['delta_v'],context['shared'],context['stacked'])
    x0=context['e_mf']+jnp.where(context['occupied'],-1e-2,1e-2)
    cfg=ScalarRootConfig(method=method,ftol=1e-8,xtol=1e-8,maxiter=100,step_cap=.05)
    solve=jax.jit(lambda x:solve_scalar_roots(f,x,config=cfg))
    started=time.perf_counter()
    compiled=solve.lower(x0).compile()
    out=compiled(x0);jax.block_until_ready(out)
    first=time.perf_counter()-started
    times=[]
    for _ in range(7):
        started=time.perf_counter();out=compiled(x0);jax.block_until_ready(out)
        times.append(time.perf_counter()-started)
    memory=compiled.memory_analysis()
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    return dict(case=case,method=method,atom=atom,basis=basis,orbs=orbs,nw=100,eta_hartree=.001,
                backend=jax.default_backend(),jax=jax.__version__,python=platform.python_version(),
                platform=platform.platform(),dtype='float64',ftol=cfg.ftol,xtol=cfg.xtol,
                maxiter=cfg.maxiter,step_cap=cfg.step_cap,hybrid_every=cfg.hybrid_every,max_backtrack=cfg.max_backtrack,
                compile_and_first_solve_seconds=first,steady_median_seconds=float(np.median(times)),
                steady_seconds=times,peak_rss_bytes=rss,xla_temporary_bytes=memory.temp_size_in_bytes,
                roots=np.asarray(out.roots).tolist(),residual=np.asarray(out.residual).tolist(),
                converged=np.asarray(out.converged).tolist(),iterations=np.asarray(out.iterations).tolist(),
                function_evaluations=int(out.function_evaluations),derivative_evaluations=int(out.derivative_evaluations),
                backtracks=int(out.backtracks),newton_steps=np.asarray(out.newton_steps).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',nargs=2,metavar=('CASE','METHOD'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--case',action='append',choices=list(CASES))
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(measure(*args.worker)));return
    if args.output is None:parser.error('--output is required')
    results=[]
    for case in args.case or CASES:
        for method in ('secant','newton','hybrid'):
            command=[sys.executable,str(Path(__file__).resolve()),'--worker',case,method]
            worker=subprocess.run(command,env=dict(os.environ,JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1'),
                                  capture_output=True,text=True,check=True)
            item=json.loads(worker.stdout);results.append(item)
            print(case,method,item['converged'],item['iterations'],item['steady_median_seconds'],flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            summary=[]
            for name in dict.fromkeys(x['case'] for x in results):
                rows=[x for x in results if x['case']==name]
                base=next(x for x in rows if x['method']=='secant')
                for row in rows:
                    common=all(base['converged']) and all(row['converged'])
                    difference=max(abs(a-b) for a,b in zip(base['roots'],row['roots'])) if common else None
                    summary.append(dict(case=name,method=row['method'],all_converged=common,
                        max_root_difference_hartree=difference,
                        same_roots=common and difference<1e-6,
                        time_ratio_to_secant=row['steady_median_seconds']/base['steady_median_seconds']))
            args.output.write_text(json.dumps(dict(records=results,comparisons=summary),indent=2)+'\n')


if __name__=='__main__':main()
