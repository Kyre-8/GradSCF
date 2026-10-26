"""Reproducible alpha/beta-string FCI forward and energy-response checks.

Each case runs in a fresh CPU process. The scalar derivative scales h1 in
Hartree at fixed ERIs and orbital topology; it is not a nuclear derivative.
"""
from pathlib import Path
import json
import platform
import resource
import subprocess
import sys
import time


def measure(norb):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    import numpy as np
    import pyscf
    from pyscf import lib
    from pyscf.fci import direct_spin1
    from gradscf import fci
    from gradscf.solvers import EigenSolverConfig
    lib.num_threads(1)
    seed=20260926
    rng=np.random.default_rng(seed)
    h=rng.normal(size=(norb,norb))*.025
    h=(h+h.T)*.5+np.diag(np.linspace(-2.,2.,norb))
    l=rng.normal(size=(12,norb,norb))*.035
    l=(l+l.transpose(0,2,1))*.5
    g=np.einsum('Lpq,Lrs->pqrs',l,l)
    nelec=(norb//2,norb//2)
    space=fci.make_fci_space(norb,nelec)
    cfg=EigenSolverConfig(method='davidson',atol=1e-9,maxiter=120,max_subspace=20)
    def energy(scale):
        return fci.solve_fci(jnp.asarray(h)*scale,jnp.asarray(g),space,config=cfg).total_energies[0]
    begin=time.perf_counter()
    compiled=jax.jit(jax.value_and_grad(energy)).lower(1.).compile()
    value,grad=compiled(1.);jax.block_until_ready((value,grad))
    elapsed=time.perf_counter()-begin
    begin=time.perf_counter();jax.block_until_ready(compiled(1.));steady=time.perf_counter()-begin
    fd=float((compiled(1.0001)[0]-compiled(.9999)[0])/2e-4)
    reference=direct_spin1.FCISolver();reference.pspace_size=0
    reference.conv_tol=1e-12;reference.max_cycle=120
    expected,c=reference.kernel(h,g,norb,nelec)
    assert reference.converged
    error=abs(float(value)-expected)
    assert error<1e-8 and abs(float(grad)-fd)<1e-7, (norb,float(value),float(expected),float(grad),fd)
    memory=compiled.memory_analysis()
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    return dict(norb=norb,nelec=nelec,determinants=space.size,strings=space.shape,
                energy_hartree=float(value),pyscf_energy_hartree=float(expected),energy_error_hartree=error,
                derivative=float(grad),finite_difference=fd,derivative_error=abs(float(grad)-fd),
                compile_first_gradient_seconds=elapsed,steady_seconds=steady,
                peak_rss_bytes=rss,xla_temporary_bytes=memory.temp_size_in_bytes,
                backend=jax.default_backend(),dtype='float64',seed=seed,
                platform=platform.platform(),python=platform.python_version(),jax=jax.__version__,
                pyscf=pyscf.__version__,max_space=20,conv_tol_hartree=1e-9,max_cycle=120)


if __name__=='__main__':
    code="import runpy,json,sys; print(json.dumps(runpy.run_path(sys.argv[1])['measure'](int(sys.argv[2]))),flush=True)"
    for norb in (8,10):
        subprocess.run([sys.executable,'-c',code,str(Path(__file__).resolve()),str(norb)],check=True)
