"""WF21 FCI E E action benchmark; run old/new sources in separate processes.

Example: JAX_PLATFORMS=cpu PYTHONPATH=src python tests/fci/benchmark_action_memory.py
    --norb 10 --output /tmp/fci-n10.json --output-npz /tmp/fci-n10.npz
This measures contract_2e only, without geometry derivatives or an SCF/FCI solve.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

# Accelerator selection, when requested, belongs to the launching process.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from gradscf import fci
from gradscf.fci import hamiltonian as ham


def fingerprint(value):
    value = np.ascontiguousarray(value)
    metadata = dict(dtype=value.dtype.str, shape=list(value.shape), nbytes=value.nbytes)
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    digest.update(value.tobytes())
    return dict(**metadata, sha256=digest.hexdigest())


def rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current = None
    if Path("/proc/self/status").exists():
        current = next(int(line.split()[1]) for line in
                       Path("/proc/self/status").read_text().splitlines()
                       if line.startswith("VmRSS:"))
    return dict(current_kib=current, peak_kib=peak / 1024 if sys.platform == "darwin" else peak)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norb", type=int, default=8)
    parser.add_argument("--nalpha", type=int)
    parser.add_argument("--nbeta", type=int)
    parser.add_argument("--seed", type=int, default=2104)
    parser.add_argument("--max-memory", type=float, default=64., help="Decimal MB; new API only")
    parser.add_argument("--nrepeat", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, help="Save one sigma array for cross-source comparison")
    args = parser.parse_args()
    if args.nrepeat < 1:
        parser.error("--nrepeat must be positive")
    nelec = tuple(args.norb // 2 if n is None else n for n in (args.nalpha, args.nbeta))
    space = fci.make_fci_space(args.norb, nelec)
    keywords = dict(max_workspace_elements=max(1, args.norb**2 * space.size, args.norb**4))
    supports_memory = "max_memory" in inspect.signature(ham.contract_2e).parameters
    if supports_memory:
        keywords["max_memory"] = args.max_memory
    files = [Path(__file__), Path(ham.__file__), Path(fci.__file__).with_name("cistring.py")]
    cpuinfo = Path("/proc/cpuinfo")
    cpu = next((line.partition(":")[2].strip() for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")), platform.processor()) if cpuinfo.exists() else platform.processor()
    report = dict(complete=False, scope="WF21_contract_2e_action_only", command=sys.argv,
        norb=args.norb, nelec=nelec, space_shape=space.shape, determinants=space.size,
        seed=args.seed, input_recipe="symmetric normal(norb+1,norb,norb)*0.1 factors; Gram ERI; normalized normal CI",
        requested_max_memory_mb=args.max_memory, supports_max_memory=supports_memory,
        action_keywords=keywords, jax=jax.__version__, numpy=np.__version__,
        python=platform.python_version(), platform=platform.platform(), cpu=cpu,
        rss_peak_source="resource.getrusage(RUSAGE_SELF).ru_maxrss; converted to KiB on Darwin",
        rss_current_source="/proc/self/status VmRSS (KiB); null when unavailable",
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        environment={key: os.environ.get(key) for key in
            ("JAX_PLATFORMS", "JAX_ENABLE_X64", "XLA_FLAGS", "CUDA_VISIBLE_DEVICES",
             "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        source_sha256={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(phase):
        report.update(phase=phase, rss_latest=rss())
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(phase, report["rss_latest"], flush=True)

    report["rss_before_inputs"] = rss()
    save("inputs")
    rng = np.random.default_rng(args.seed)
    factors = rng.normal(size=(args.norb + 1, args.norb, args.norb)) * .1
    factors = (factors + factors.transpose(0, 2, 1)) * .5
    eri = np.einsum("Lpq,Lrs->pqrs", factors, factors)
    coefficients = rng.normal(size=space.shape)
    coefficients /= np.linalg.norm(coefficients)
    report["inputs"] = dict(eri=fingerprint(eri), coefficients=fingerprint(coefficients))
    eri, coefficients = jnp.asarray(eri), jnp.asarray(coefficients)
    jax.block_until_ready((eri, coefficients))
    report.update(backend=jax.default_backend(), device=str(coefficients.device),
                  device_kind=coefficients.device.device_kind, x64_enabled=jax.config.x64_enabled)
    report["rss_before_lower"] = rss()
    save("lower")
    started = perf_counter()
    lowered = jax.jit(lambda g, c: ham.contract_2e(g, c, space, **keywords)).lower(eri, coefficients)
    report["lower_seconds"] = perf_counter() - started
    save("compile")
    started = perf_counter()
    compiled = lowered.compile()
    report["compile_seconds"] = perf_counter() - started
    memory = compiled.memory_analysis()
    report["compiled_memory_bytes"] = None if memory is None else {name: int(getattr(memory, name))
        for name in ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes",
                     "temp_size_in_bytes", "host_argument_size_in_bytes", "host_output_size_in_bytes",
                     "host_alias_size_in_bytes", "host_temp_size_in_bytes") if hasattr(memory, name)}
    report["rss_before_action"] = rss()
    save("first_action")
    started = perf_counter()
    sigma = jax.block_until_ready(compiled(eri, coefficients))
    report["first_action_seconds"] = perf_counter() - started
    report["rss_after_first_action"] = rss()
    save("warm_action")
    samples = []
    for _ in range(args.nrepeat):
        started = perf_counter()
        sigma = jax.block_until_ready(compiled(eri, coefficients))
        samples.append(perf_counter() - started)
    report.update(warm_action_seconds=samples, warm_action_median_seconds=float(np.median(samples)))
    report["rss_after_warm_action"] = rss()
    sigma = np.asarray(sigma)
    report.update(finite=bool(np.isfinite(sigma).all()), output=fingerprint(sigma),
                  checksum_sum=float(np.sum(sigma)), checksum_norm=float(np.linalg.norm(sigma)))
    if args.output_npz:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output_npz, sigma=sigma)
        report["output_npz"] = str(args.output_npz.resolve())
    report["complete"] = report["finite"]
    save("complete" if report["finite"] else "nonfinite")
    if not report["finite"]:
        raise RuntimeError("FCI action produced a nonfinite output")


if __name__ == "__main__":
    main()
