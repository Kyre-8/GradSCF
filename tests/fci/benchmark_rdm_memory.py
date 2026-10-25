"""Matched old/new public RDM timing and reverse-AD storage diagnostics.

Run one source, orbital count and mode per process. The runner only calls
make_rdm12s/trans_rdm12s; it contains no density-matrix contraction formulas.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import jax
from jax.ad_checkpoint import print_saved_residuals
import jax.numpy as jnp
import numpy as np

from gradscf import fci
from gradscf.fci import hamiltonian, rdm
import benchmark_action_memory as common
import benchmark_action_ad_memory as diagnostics

jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norb", type=int, default=8)
    parser.add_argument("--mode", choices=("same", "transition"), default="same")
    parser.add_argument("--seed", type=int, default=2201)
    parser.add_argument("--max-memory", type=float, default=60000.)
    parser.add_argument("--tile-elements", type=int, default=8_000_000)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--expected-rdm-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()
    source_hash = hashlib.sha256(Path(rdm.__file__).read_bytes()).hexdigest()
    if source_hash != args.expected_rdm_sha:
        raise RuntimeError("RDM source differs from the frozen benchmark target")
    space = fci.make_fci_space(args.norb, (args.norb//2,)*2)
    supports_memory = "max_memory" in inspect.signature(rdm.trans_rdm12s).parameters
    cap = args.tile_elements if supports_memory else max(
        args.tile_elements, args.norb**2*space.size, args.norb**4)
    keywords = dict(max_workspace_elements=cap)
    if supports_memory:
        keywords["max_memory"] = args.max_memory
    files = (__file__, rdm.__file__, hamiltonian.__file__, common.__file__, diagnostics.__file__,
             str(Path(fci.__file__).with_name("cistring.py")))
    report = dict(complete=False, scope="public_RDM_only_no_solver_or_geometry",
        command=sys.argv, mode=args.mode, reverse=args.reverse, norb=args.norb,
        nelec=space.nelec, space_shape=space.shape, determinants=space.size,
        seed=args.seed, input_recipe="independent normal(seed=2201) CI arrays, each normalized",
        supports_max_memory=supports_memory, action_keywords=keywords,
        output_policy="all leaves of the unchanged public return structure",
        jax=jax.__version__, numpy=np.__version__, python=platform.python_version(),
        affinity=sorted(os.sched_getaffinity(0)),
        environment={key: os.environ.get(key) for key in
            ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "JAX_ENABLE_X64", "OMP_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")},
        source_sha256={str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                       for path in files})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(phase):
        report.update(phase=phase, rss_latest=common.rss())
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(phase, report["rss_latest"], flush=True)

    save("inputs")
    rng = np.random.default_rng(args.seed)
    inputs = []
    for _ in range(1 if args.mode == "same" else 2):
        value = rng.normal(size=space.shape)
        value /= np.linalg.norm(value)
        inputs.append(value)
    report["inputs"] = [common.fingerprint(value) for value in inputs]
    point = tuple(jnp.asarray(value) for value in inputs)
    jax.block_until_ready(point)
    report.update(backend=jax.default_backend(), device=str(point[0].device),
                  device_kind=point[0].device.device_kind, x64_enabled=jax.config.x64_enabled)
    public = rdm.make_rdm12s if args.mode == "same" else rdm.trans_rdm12s
    action = lambda *values: public(*values, space, **keywords)
    if args.reverse:
        shape = jax.eval_shape(action, *point)
        probe_rng = np.random.default_rng(args.seed+1)
        probes = jax.tree.map(lambda value: jnp.asarray(
            probe_rng.normal(size=value.shape), dtype=value.dtype), shape)
        report["probe_inputs"] = [common.fingerprint(value) for value in jax.tree.leaves(probes)]

        def loss(*values):
            return sum(jnp.vdot(value, probe) for value, probe in
                       zip(jax.tree.leaves(action(*values)), jax.tree.leaves(probes)))

        residuals = io.StringIO()
        with redirect_stdout(residuals):
            print_saved_residuals(loss, *point)
        report["saved_residuals"] = residuals.getvalue()
        operation = jax.grad(loss, argnums=0 if len(point) == 1 else tuple(range(len(point))))
        scans = diagnostics.scan_shapes(jax.make_jaxpr(operation)(*point))
        report["gradient_scans"] = scans
        report["scan_outputs_matching_per_iteration_full_CI"] = [scan for scan in scans
            if any(tuple(shape) in ((scan["length"], *space.shape), (scan["length"], space.size))
                   for shape in scan["output_shapes"])]
    else:
        operation = action
    save("lower")
    start = perf_counter()
    lowered = jax.jit(operation).lower(*point)
    report["lower_seconds"] = perf_counter()-start
    save("compile")
    start = perf_counter()
    compiled = lowered.compile()
    report["compile_seconds"] = perf_counter()-start
    memory = compiled.memory_analysis()
    report["compiled_memory_bytes"] = None if memory is None else {
        name: int(getattr(memory, name)) for name in
        ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")}
    save("first_action")
    samples = []
    for index in range(4):
        start = perf_counter()
        output = jax.block_until_ready(compiled(*point))
        samples.append(perf_counter()-start)
        if not index:
            save("warm_actions")
    leaves = [np.asarray(value) for value in jax.tree.leaves(output)]
    report.update(first_seconds=samples[0], warm_seconds=samples[1:],
        warm_median_seconds=float(np.median(samples[1:])),
        output=[common.fingerprint(value) for value in leaves],
        output_tree=str(jax.tree.structure(output)),
        finite=all(bool(np.isfinite(value).all()) for value in leaves))
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **{f"leaf_{index}": value for index, value in enumerate(leaves)})
    report["complete"] = report["finite"]
    save("complete" if report["finite"] else "nonfinite")
    if not report["finite"]:
        raise RuntimeError("Public RDM operation produced nonfinite outputs")


if __name__ == "__main__":
    main()
