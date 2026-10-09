"""Feasibility probe: peak GPU memory / time for one training step at a given
basis set + DFT grid level.  Does NOT touch the production cache."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from gradscf import training
from gradscf.workflows.core import run_molecule_from_spec
from gradscf.workflows.types import MoleculeSpecConfig, SimulationConfig

from skalax_integration.adapter import load_skala_functional
from skalax_integration.train_epoch_sc import SkalaTrainFunctional


jax.config.update("jax_enable_x64", True)
GiB = 2 ** 30


def parse_xyz(path: Path, n: int) -> list[str]:
    lines = path.read_text().splitlines()
    specs, i = [], 0
    while i < len(lines) and len(specs) < n:
        if not lines[i].strip():
            i += 1
            continue
        na = int(lines[i].strip())
        specs.append("; ".join(f"{f[0]} {f[1]} {f[2]} {f[3]}"
                               for f in (ln.split() for ln in lines[i + 2:i + 2 + na])))
        i += na + 2
    return specs


def peak() -> float:
    st = jax.local_devices()[0].memory_stats() or {}
    return float(st.get("peak_bytes_in_use", 0)) / GiB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz-file", type=Path, required=True)
    ap.add_argument("--indices", default="2,38")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--grids-level", type=int, default=3)
    ap.add_argument("--max-cycle", type=int, default=60)
    args = ap.parse_args()

    specs = parse_xyz(args.xyz_file, 50)
    model = load_skala_functional(key=jax.random.PRNGKey(0))
    functional = SkalaTrainFunctional(model)
    config = training.MolecularTrainingConfig(
        mode="self_consistent", scf_gradient_mode="impl",
        e0_total_mse_weight=0.0, e0_total_mae_weight=1.0,
        scf_max_cycle=args.max_cycle, scf_damping=0.5, scf_level_shift=0.0,
        scf_eigenvalue_jitter=1e-8, scf_iterate_selection="first_converged",
        scf_conv_tol_energy=1e-9, scf_conv_tol_density=1e-8, scf_conv_tol_grad=3e-6,
        scf_implicit_diff_tolerance=1e-8, scf_implicit_diff_max_iter=20,
        scf_implicit_diff_restart=40,
        scf_convergence_metric="energy_and_residual", scf_require_converged=False,
    )

    cpu = jax.devices("cpu")[0]
    for k in [int(x) for x in args.indices.split(",")]:
        print(f"--- molecule {k} : {args.basis} / grid level {args.grids_level} ---", flush=True)
        t0 = time.time()
        with jax.default_device(cpu):
            sim = SimulationConfig(
                scf_backend="jax_rks", jax_rks_xc_spec="b3lyp",
                jax_rks_max_cycle=300, jax_rks_damping=0.3,
            )
            m = run_molecule_from_spec(
                MoleculeSpecConfig(atom=specs[k], basis=args.basis, xc="b3lyp",
                                   unit="Angstrom", grids_level=args.grids_level),
                simulation=sim,
            ).molecule
        e = float(m.mf_energy)
        print(f"  build {time.time()-t0:.1f}s  ngrid={int(m.ao.shape[0])} nao={int(m.ao.shape[1])} "
              f"E_ref={e:.6f}  ref_conv={getattr(m,'scf_converged',None)}", flush=True)
        datum = training.MolecularTrainingDatum(
            molecule=m, target_e0_total_h=jnp.asarray(e, dtype=jnp.float64))
        try:
            state = training.trainer.create_train_state_from_molecule(
                functional, jax.random.PRNGKey(0), m, optax.adam(1e-5))
            step = training.trainer.make_molecular_train_step(functional, training_config=config)
            t0 = time.time()
            state, metrics = step(state, (datum,))
            dt = time.time() - t0
            print(f"  1 train step OK: {dt:.1f}s  peak={peak():.2f} GiB  "
                  f"loss={float(metrics['total_loss']):.4e} "
                  f"conv={float(metrics['scf_converged_fraction'][0]):.2f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)


if __name__ == "__main__":
    main()
