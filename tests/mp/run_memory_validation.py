"""Bound one isolated MP validation process and record its exact evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--seconds', type=float, default=570.)
    parser.add_argument('--script', action='store_true')
    parser.add_argument('nodes', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.seconds <= 570:
        parser.error('--seconds must be in (0, 570]')
    if not args.nodes:
        parser.error('Supply pytest nodes or a benchmark script')
    stage = Path.cwd()

    def manifest():
        return {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
                for root in ('src', 'tests/mp') for p in (stage/root).rglob('*.py')}

    before = manifest()
    Path(args.name+'-source-before.json').write_text(json.dumps(before, indent=2)+'\n')
    env = os.environ.copy()
    env.update(PYTHONPATH=str(stage/'src'), CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
        JAX_ENABLE_X64='true', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', PYTHONPYCACHEPREFIX=str(stage/'pycache'),
        GRADSCF_VALIDATION_METADATA=str(stage/(args.name+'-backend.json')))
    code = ("import jax,jaxlib,numpy,json,os,platform; from pathlib import Path; "
            "m=dict(backend=jax.default_backend(),devices=[str(d) for d in jax.devices()],"
            "jax=jax.__version__,jaxlib=jaxlib.__version__,numpy=numpy.__version__,"
            "x64=jax.config.x64_enabled,affinity=sorted(os.sched_getaffinity(0)),"
            "hostname=platform.node(),exclusive_node=False,"
            "cpu_model=next((s.split(':',1)[1].strip() for s in "
            "Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),None),"
            "compilation_cache_dir=jax.config.jax_compilation_cache_dir); "
            "Path(os.environ['GRADSCF_VALIDATION_METADATA']).write_text(json.dumps(m,indent=2)); "
            "print(json.dumps(m),flush=True); ")
    code += ("import runpy,sys; sys.argv="+repr(args.nodes)+
             "; sys.path.insert(0,str(Path(sys.argv[0]).resolve().parent)); "
             "runpy.run_path(sys.argv[0],run_name='__main__')" if args.script else
             "import pytest; raise SystemExit(pytest.main("+
             repr(['-vv', '-s', *args.nodes, '--junitxml='+args.name+'.xml'])+"))")
    command = [sys.executable, '-c', code]
    start = time.perf_counter()
    timed_out = False
    with Path(args.name+'.log').open('w') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            code = child.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = 124
    after = manifest()
    Path(args.name+'-source-after.json').write_text(json.dumps(after, indent=2)+'\n')
    result = dict(exit_code=code, child_returncode=child.returncode, timed_out=timed_out,
        process_seconds=time.perf_counter()-start, source_unchanged=before == after,
        child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        memory_scope='single_validation_child_peak_excludes_runner_not_whole_job',
        command=command, python=platform.python_version(), platform=platform.platform(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), affinity=sorted(os.sched_getaffinity(0)))
    Path(args.name+'.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    raise SystemExit(code if before == after else 2)


if __name__ == '__main__':
    main()
