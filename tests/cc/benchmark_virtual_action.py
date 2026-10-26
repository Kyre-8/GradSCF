"""Matched isolated-process benchmark of explicit and contracted CC dressing.

The action comes from src; the old explicit formula is the frozen test oracle.
This measures one virtual action, without an SCF/CC solve or geometry response.
The ``contracted`` choice calls the production action, including its static
choice of a small transient W when that is cheaper than the factored scratch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from gradscf.cc import _intermediates as imd
from test_virtual_action import _explicit_contract


def fingerprint(array):
    array = np.ascontiguousarray(array)
    metadata = dict(shape=array.shape, dtype=array.dtype.str, nbytes=array.nbytes)
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    digest.update(array.tobytes())
    return dict(**metadata, sha256=digest.hexdigest())


def memory():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(peak_kib=peak/1024 if sys.platform == "darwin" else peak,
        current_kib=next(int(x.split()[1]) for x in Path("/proc/self/status").read_text().splitlines()
                         if x.startswith("VmRSS:")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("explicit", "contracted"), required=True)
    parser.add_argument("--nocc", type=int, default=8)
    parser.add_argument("--nvir", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2314)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.nocc, args.nvir, args.repeats) < 1:
        parser.error("Orbital counts and repeats must be positive")
    cpu = next(line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    files = (Path(__file__), Path(imd.__file__), Path(__file__).with_name("test_virtual_action.py"))
    report = dict(complete=False, scope="WF10a_single_virtual_action", command=sys.argv,
        implementation=args.implementation, nocc=args.nocc, nvir=args.nvir, seed=args.seed,
        input_recipe="independent normal float64 t1,ovvv,vvvv,X, scaled by .03,.05,.05,.04",
        jax=jax.__version__, python=platform.python_version(), cpu=cpu,
        affinity=sorted(os.sched_getaffinity(0)), memory_before=memory(),
        environment={name: os.environ.get(name) for name in ("JAX_PLATFORMS", "JAX_ENABLE_X64",
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(phase):
        report.update(phase=phase, memory_latest=memory())
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(phase, report["memory_latest"], flush=True)

    rng = np.random.default_rng(args.seed)
    no, nv = args.nocc, args.nvir
    arrays = []
    for shape, scale in (
        ((no, nv), .03), ((no, nv, nv, nv), .05), ((nv,)*4, .05),
        ((no, no, nv, nv), .04)):
        array = rng.normal(size=shape)
        array *= scale
        arrays.append(array)
    report["inputs"] = [fingerprint(a) for a in arrays]
    arrays = tuple(jnp.asarray(a) for a in arrays)
    jax.block_until_ready(arrays)
    report.update(backend=jax.default_backend(), device=str(arrays[0].device),
                  device_kind=arrays[0].device.device_kind, x64_enabled=jax.config.x64_enabled,
                  memory_after_inputs=memory())
    action = _explicit_contract if args.implementation == "explicit" else imd.contract_Wvvvv

    def function(t1, ovvv, vvvv, x):
        eris = SimpleNamespace(vvvv=vvvv, get_ovvv=lambda: ovvv)
        return action(t1, eris, x)

    save("compile")
    started = perf_counter()
    executable = jax.jit(function).lower(*arrays).compile()
    report["compile_seconds"] = perf_counter()-started
    analysis = executable.memory_analysis()
    report["xla_memory_bytes"] = {name: int(getattr(analysis, name)) for name in
        ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")}
    report["memory_after_compile"] = memory()
    save("first")
    started = perf_counter()
    result = jax.block_until_ready(executable(*arrays))
    report["first_seconds"] = perf_counter()-started
    report["memory_after_first"] = memory()
    durations = []
    for _ in range(args.repeats):
        started = perf_counter()
        result = jax.block_until_ready(executable(*arrays))
        durations.append(perf_counter()-started)
    result = np.asarray(result)
    np.savez(args.output.with_suffix(".npz"), action=result)
    report.update(complete=bool(np.isfinite(result).all()), warm_seconds=durations,
                  warm_median_seconds=float(np.median(durations)), output=fingerprint(result))
    save("complete")
    if not report["complete"]:
        raise RuntimeError("Nonfinite virtual action")


if __name__ == "__main__":
    main()
