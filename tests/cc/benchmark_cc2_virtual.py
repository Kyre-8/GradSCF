"""Measure the complete CC2 virtual term, including formation of T1 outer T1.

Both implementations receive only t1, ovvv and bare vvvv. The independent old
inline formula contracts W with the two singles factors directly. Shared mode
forms their outer product before the source action; factored mode lets that
action defer materialization. Every mode times the complete term.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from gradscf.cc import _intermediates as imd
import benchmark_virtual_action as capture
import test_virtual_action as oracle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("inline", "shared", "factored"), required=True)
    parser.add_argument("--nocc", type=int, required=True)
    parser.add_argument("--nvir", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2314)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.nocc, args.nvir, args.repeats) < 1:
        parser.error("Orbital counts and repeats must be positive")
    cc = Path(imd.__file__).parent
    files = (Path(__file__), Path(imd.__file__), cc/"_equations.py",
             cc/"_qci_equations.py", Path(oracle.__file__), Path(capture.__file__))
    report = dict(complete=False, scope="WF10a_CC2_complete_virtual_term", command=sys.argv,
        implementation=args.implementation, nocc=args.nocc, nvir=args.nvir, seed=args.seed,
        input_recipe="independent normal float64 t1,ovvv,vvvv; scales .03,.05,.05",
        pair_policy={"inline": "inline rank-one contraction", "shared": "outer product before action",
                     "factored": "tuple; action chooses materialization"}[args.implementation],
        jax=jax.__version__, numpy=np.__version__, python=platform.python_version(),
        cpu=next(line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                 if line.startswith("model name")), affinity=sorted(os.sched_getaffinity(0)),
        environment={name: os.environ.get(name) for name in ("JAX_PLATFORMS", "JAX_ENABLE_X64",
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        memory_before=capture.memory())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(phase):
        report.update(phase=phase, memory_latest=capture.memory())
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(phase, report["memory_latest"], flush=True)

    rng = np.random.default_rng(args.seed)
    no, nv = args.nocc, args.nvir
    arrays = []
    for shape, scale in (((no, nv), .03), ((no, nv, nv, nv), .05), ((nv,)*4, .05)):
        array = rng.normal(size=shape)
        array *= scale
        arrays.append(array)
    report["inputs"] = [capture.fingerprint(a) for a in arrays]
    arrays = tuple(jnp.asarray(a) for a in arrays)
    jax.block_until_ready(arrays)
    report.update(backend=jax.default_backend(), device=str(arrays[0].device),
                  x64_enabled=jax.config.x64_enabled, memory_after_inputs=capture.memory())

    def term(t1, ovvv, vvvv):
        eris = SimpleNamespace(vvvv=vvvv, get_ovvv=lambda: ovvv)
        if args.implementation == "inline":
            return oracle._explicit_cc2(t1, eris)
        if args.implementation == "factored":
            return imd.contract_Wvvvv(t1, eris, (t1, t1))
        pair = jnp.einsum("ic,jd->ijcd", t1, t1, precision="highest")
        return imd.contract_Wvvvv(t1, eris, pair)

    save("compile")
    started = perf_counter()
    executable = jax.jit(term).lower(*arrays).compile()
    report["compile_seconds"] = perf_counter()-started
    analysis = executable.memory_analysis()
    report["xla_memory_bytes"] = {name: int(getattr(analysis, name)) for name in
        ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")}
    report["memory_after_compile"] = capture.memory()
    save("first")
    started = perf_counter()
    result = jax.block_until_ready(executable(*arrays))
    report.update(first_seconds=perf_counter()-started, memory_after_first=capture.memory())
    samples = []
    for _ in range(args.repeats):
        started = perf_counter()
        result = jax.block_until_ready(executable(*arrays))
        samples.append(perf_counter()-started)
    result = np.asarray(result)
    np.savez(args.output.with_suffix(".npz"), term=result)
    report.update(complete=bool(np.isfinite(result).all()), warm_seconds=samples,
                  warm_median_seconds=float(np.median(samples)), output=capture.fingerprint(result))
    save("complete")
    if not report["complete"]:
        raise RuntimeError("Nonfinite CC2 virtual term")


if __name__ == "__main__":
    main()
