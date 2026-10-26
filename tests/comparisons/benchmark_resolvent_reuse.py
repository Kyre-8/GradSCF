"""Serial dense-reference/factor-reuse comparison with the same finite-eta CD response."""
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
import json, time, platform, os, hashlib
import jax.numpy as jnp
import jax,numpy as np
jax.config.update('jax_enable_x64',True)
from gradscf import scf,gto,gw
from gradscf.gw import g0w0, self_energy
from gradscf.gw.poles import _rpa_matrices


def dense_model(energy, ov, factors, *, eta=0.):
    matrix, coupling = _rpa_matrices(energy, ov, factors)
    real_matrix, real_coupling = _rpa_matrices(energy, ov, factors, eta=eta)
    return dict(matrix=matrix, coupling=coupling, eta=eta,
                retarded=dict(matrix=real_matrix, coupling=real_coupling))


def dense_w(model, frequencies, *, imaginary=False, pairs=None):
    # Evaluate all MO pairs with a per-frequency dense solve. This is
    # a benchmark reference only; production uses the shared shifted solver.
    eta = model['eta']
    state = model if imaginary else model['retarded']
    matrix, coupling = state['matrix'], state['coupling']
    def one(w):
        z = 1j*w if imaginary else w+1j*eta
        solution = jnp.linalg.solve(matrix-z*z*jnp.eye(matrix.shape[0]), coupling)
        return -4*jnp.sum(coupling*solution, axis=0)
    flat = jax.vmap(one)(frequencies)
    if pairs is not None:
        return flat[jnp.arange(len(frequencies)), pairs]
    nmo = int(coupling.shape[1]**.5)
    return flat.reshape(len(frequencies), nmo, nmo)


geometry=json.loads(Path('reproducibility/gw_bse/cycloalkane_scaling/geometries.json').read_text())[0]
mf=scf.RHF(gto.M(atom=geometry['atom'],basis='sto-3g'),conv_tol=1e-10).density_fit('weigend').run()
assert mf.converged

report={'system':geometry,'backend':jax.default_backend(),'device':str(jax.devices()[0]),'jax':jax.__version__,
'python':platform.python_version(),'platform':platform.platform(),'basis':'sto-3g','auxbasis':'weigend','nw':64,
'eta_ha':1e-5,'conv_tol_ha':1e-7,'damping':.3,'nmo':len(mf.mo_energy),'naux':mf._scf_inputs.df_factors.shape[0],
'repeats':3,'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
'response_convention':'Legacy CD real-axis 2i*eta; separate real imaginary-axis and complex retarded RPA models',
'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('src/gradscf/gw/poles.py','src/gradscf/solvers/linear/shifted.py')},
'note':'Serial in one process; first call per method/path is not a fresh-process cold start. Warm timings are repeats 2 and 3. No BSE or SCF included in GW timings.', 'results':{}}
energies={}
for mode in ('dense_reference','reused_factorization'):
    report['results'][mode]={}
    with ExitStack() as context:
        if mode=='dense_reference':
            context.enter_context(patch.object(g0w0,'rpa_resolvent',dense_model))
            context.enter_context(patch.object(g0w0,'screened_w_imag_resolvent',lambda m,w: dense_w(m,w,imaginary=True)))
            context.enter_context(patch.object(self_energy,'screened_w_real_resolvent',dense_w))
        for method in ('evgw0','evgw'):
            times=[]
            for repeat in range(3):
                start=time.perf_counter()
                result=gw.GW(mf,method=method,nw=64,eta=1e-5,max_cycle=40,conv_tol=1e-7,damp=.3).run()
                jax.block_until_ready(result.result.mo_energy)
                times.append(time.perf_counter()-start)
                assert result.converged
                print(mode,method,repeat,times[-1],flush=True)
            e=np.asarray(result.mo_energy)
            if mode=='dense_reference':energies[method]=e.copy()
            error=float(max(abs(e-energies[method])))
            assert error<1e-9
            report['results'][mode][method]={'seconds':times,'warm_median_s':float(np.median(times[1:])),
                'max_energy_difference_ha':error,'qp_residual_max_ha':float(max(abs(np.asarray(result.result.qp_residual))))}
            Path('reproducibility/gw_bse/cycloalkane_scaling/resolvent_reuse.json').write_text(json.dumps(report,indent=2)+'\n')
