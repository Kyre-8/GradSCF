"""Warm CPU three-center VJP benchmark; no SCF, full ERI, or MACE model.

Compare thread counts using the current native backend and identical inputs.
Example: PYTHONPATH=src python tools/benchmark_native_basis_vjp.py \
    --threads 1 8 16 --output artifacts/native-vjp-timings.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import integrals
from gradscf.integrals._native import register_integrals
from gradscf.model.nnao.basis import prepare_direct_basis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--xyz", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or min(args.threads) < 1:
        parser.error("repeats and threads must be positive")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")
    if args.xyz:
        atom = ";".join(args.xyz.read_text().splitlines()[2:])
    else:
        # Benzene, Angstrom; identical geometry for all timing runs.
        atoms = []
        for symbol, radius in (("C", 1.397), ("H", 2.479)):
            for angle in np.arange(6)*np.pi/3:
                atoms.append(f"{symbol} {radius*np.cos(angle):.12f} {radius*np.sin(angle):.12f} 0")
        atom = ";".join(atoms)
    direct = prepare_direct_basis(atom, basis_family="szp663_direct", cart=False)
    top, params = direct.topology, direct.parameters
    at, ap = integrals.prepare_basis(atom, "def2-universal-jkfit", cart=False)
    ri = integrals.make_auxiliary_plan(top, at)
    atm, bas, env = ri._pack(params, ap)
    split = len(top.angular_momenta)
    coeff_indices = np.concatenate([np.arange(row[6], row[6]+row[2]*row[3]) for row in bas[:split]])
    ncoeff = len(coeff_indices)
    nexp = sum(int(row[2]) for row in bas[:split])
    shape = (at.nao, top.nao*(top.nao+1)//2)
    cot = jnp.asarray(np.random.default_rng(20261011).normal(size=shape))
    packed = tuple(jnp.asarray(x) for x in (atm, bas, env))
    coeff = packed[2][coeff_indices]
    report = dict(host=platform.node(), platform=platform.platform(), jax=jax.__version__,
                  geometry=atom, basis="szp663_direct", auxiliary="def2-universal-jkfit",
                  dtype="float64", seed=20261011, nao=top.nao, naux=at.nao,
                  ncoeff=ncoeff, nexponent=nexp, rows=[])
    library = Path(register_integrals()._name)
    native_hash = hashlib.sha256(library.read_bytes()).hexdigest()
    # The reference is the current implementation with one worker, not an old library.
    thread_counts = list(dict.fromkeys([1, *args.threads]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for kind, size in (("coefficient", ncoeff), ("exponent", nexp)):
        call = jax.ffi.ffi_call(f"gradscf_{kind}_vjp_cpu_v1",
                               jax.ShapeDtypeStruct((size,), jnp.float64))
        inputs = (*packed, coeff, cot) if kind == "coefficient" else (*packed, cot)
        fn = jax.jit(lambda *x: call(*x, operator=np.int32(6), cart=np.int32(0), split=np.int32(split)))
        reference = None
        for threads in thread_counts:
            os.environ["GRADSCF_NATIVE_THREADS"] = str(threads)
            start = time.perf_counter()
            value = np.asarray(fn(*inputs).block_until_ready())
            cold = time.perf_counter()-start
            if reference is None:
                reference = value.copy()
            error = float(np.max(np.abs(value-reference)))
            np.testing.assert_allclose(value, reference, atol=2e-9, rtol=2e-10)
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                fn(*inputs).block_until_ready()
                times.append(time.perf_counter()-start)
            row = dict(library=str(library), sha256=native_hash,
                       kind=kind, threads=threads, cold_seconds=cold, seconds=times,
                       median_seconds=float(np.median(times)), max_abs_error=error)
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
