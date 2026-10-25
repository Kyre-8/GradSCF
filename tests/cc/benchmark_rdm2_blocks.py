"""One fixed CC-density component point, not a complete-CC scaling benchmark.

Run by a bounded remote wrapper. Historical inputs are the exact 26ab61e
properties.py and _spin_density.py files supplied alongside this thin runner.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import resource
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.cc import properties
from gradscf._memory import tree_storage_bytes
from test_rdm2_blocks import _states


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ready(value):
    return jax.tree.map(lambda x:x.block_until_ready(),value)


algorithm = sys.argv[1]
directory = Path(__file__).resolve().parents[2]
baseline = directory/'baseline'
start = perf_counter()
n,nocc,state,left = _states(False,None)
operator = jnp.arange(n**4,dtype=jnp.float64).reshape((n,)*4)/n**4+.2
selection = ((3,0),(1,0),(0,3),(3,0))
setup_seconds = perf_counter()-start
if algorithm == 'legacy':
    old = load('gradscf.cc._benchmark_old_properties',baseline/'properties.py')
    gamma = load('gradscf.cc._benchmark_old_density',baseline/'_spin_density.py')
    old.active_density_parts = gamma.active_density_parts
    function = lambda r,l,g:old._rdm2_from_states(r,l,nmo=n,nocc=nocc,dtype=jnp.float64)
elif algorithm in ('full','requested'):
    indices = selection if algorithm == 'requested' else None
    function = lambda r,l,g:properties._rdm2_from_states(r,l,nmo=n,nocc=nocc,
        dtype=jnp.float64,indices=indices,max_memory=32.)
elif algorithm == 'contract':
    function = lambda r,l,g:properties._contract_rdm2_from_states(g,r,l,
        nmo=n,nocc=nocc,dtype=jnp.float64,max_memory=32.)
else:
    raise ValueError('Expected legacy, full, requested or contract')

compiled = jax.jit(function)
start = perf_counter()
lowered = compiled.lower(state,left,operator)
lower_seconds = perf_counter()-start
start = perf_counter()
executable = lowered.compile()
compile_seconds = perf_counter()-start
start = perf_counter()
output = ready(executable(state,left,operator))
first_seconds = perf_counter()-start
samples = []
for _ in range(7):
    start = perf_counter()
    ready(executable(state,left,operator))
    samples.append(perf_counter()-start)
analysis = executable.memory_analysis()
host_output = np.asarray(output)
report = dict(algorithm=algorithm,point='R nmo4 nocc2 seed713',dtype='float64',
    backend=jax.default_backend(),device=str(jax.devices()[0]),jax=jax.__version__,
    affinity=sorted(os.sched_getaffinity(0)),setup_seconds=setup_seconds,
    lower_seconds=lower_seconds,compile_seconds=compile_seconds,first_seconds=first_seconds,
    warm_median_seconds=float(np.median(samples)),warm_samples_seconds=samples,
    child_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    input_arrays_bytes=tree_storage_bytes(state,left,operator),
    xla={name:getattr(analysis,name) for name in ('argument_size_in_bytes',
        'output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')},
    checksum=float(host_output.sum()),
    inputs={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__),baseline/'properties.py',baseline/'_spin_density.py')})
# These host checks do not compile further JAX kernels after the timed sample.
# Requested and scalar execution never run a complete dense oracle.
if algorithm in ('legacy','full'):
    report['requested_checksum'] = float(host_output[np.ix_(*selection)].sum())
    report['contract_checksum'] = float(.5*np.sum(np.asarray(operator)*host_output))
print(json.dumps(report,sort_keys=True),flush=True)
