"""One fixed MP4 value/gradient workload; run each policy in a fresh process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import time
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import mp
from gradscf.mp import _series_memory, series
from gradscf.mp.integrals import _reference_data
from gradscf.solvers.nonlinear import taylor
from test_series_checkpoint import checkpoint_case
from test_series_staging import model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=("off", "on"), required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    jax.config.update("jax_enable_x64", True)
    checkpoint = args.checkpoint == "on"
    h, g, nocc, config, budgets, plans = checkpoint_case(True)
    maximum = budgets[checkpoint]
    observed = []
    original = _series_memory.plan_series

    def record(*values, **kwargs):
        plan = original(*values, **kwargs)
        observed.append(plan)
        return plan

    def loss(parameter, hh, gg):
        _, _, focks, _ = _reference_data(hh, gg, nocc)
        adjusted = hh - (parameter - 1.) * (focks[0] - hh)
        result = mp.run_mp(adjusted, parameter * gg, nocc=nocc,
                           config=config, max_memory=maximum)
        return result.correlation_energy, result

    parameter = jnp.asarray(1., dtype=h.dtype)
    evaluate = jax.jit(jax.value_and_grad(loss, has_aux=True))
    with patch.object(_series_memory, "plan_series", record):
        start = time.perf_counter()
        lowered = evaluate.lower(parameter, h, g)
        lower_seconds = time.perf_counter() - start
    assert observed and all(not p.packed and p.checkpoint == checkpoint for p in observed)
    assert all(p.resident_bytes == plans[checkpoint].resident_bytes for p in observed)
    start = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - start
    memory = executable.memory_analysis()
    start = time.perf_counter()
    (_, result), gradient = jax.block_until_ready(executable(parameter, h, g))
    first_seconds = time.perf_counter() - start
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        (_, result), gradient = jax.block_until_ready(executable(parameter, h, g))
        samples.append(time.perf_counter() - start)
    assert bool(result.valid)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves((result, gradient)))
    sources = [Path(__file__), Path(checkpoint_case.__code__.co_filename),
               Path(model.__code__.co_filename), Path(series.__file__),
               Path(_series_memory.__file__), Path(taylor.__file__),
               Path(series.__file__).with_name("_tensor.py")]
    data = dict(checkpoint=checkpoint, packed=False, max_memory_mb=maximum,
        planner=plans[checkpoint]._asdict(), order=config.order, nmo=h.shape[0], nocc=nocc,
        seed=2318, dtype=str(h.dtype), config=repr(config), backend=jax.default_backend(),
        devices=list(map(str, jax.devices())), jax=jax.__version__,
        affinity=sorted(os.sched_getaffinity(0)),
        input_sha256=[hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in (h, g)],
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        valid=bool(result.valid), corrections=np.asarray(result.corrections).tolist(),
        wavefunction_coefficients=np.asarray(result.wavefunction_coefficients).tolist(),
        t2=np.asarray(result.t2).tolist(), correlation_energy=float(result.correlation_energy),
        parameter_gradient=float(gradient), lower_seconds=lower_seconds,
        compile_seconds=compile_seconds, first_seconds=first_seconds, warm_seconds=samples,
        warm_median_seconds=statistics.median(samples),
        xla_memory_bytes={name: int(getattr(memory, name)) for name in
            ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
             "alias_size_in_bytes")},
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2), flush=True)


if __name__ == "__main__":
    main()
