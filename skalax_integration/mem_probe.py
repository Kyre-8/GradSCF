"""Measure peak GPU memory vs batch size for the Skala SC training step.

Uses the LARGEST molecules in the cache (the memory-driving ones) and runs the
same train_step as the production script, for several batch sizes in ascending
order.  JAX's peak_bytes_in_use is monotonic within a process, so reporting it
after each batch size gives the peak for that batch size.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from gradscf import training

from skalax_integration.adapter import load_skala_functional
from skalax_integration.train_epoch_sc import SkalaTrainFunctional


jax.config.update("jax_enable_x64", True)
GiB = 2 ** 30


def peak_gib() -> float:
    stats = jax.local_devices()[0].memory_stats() or {}
    return float(stats.get("peak_bytes_in_use", 0)) / GiB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--n-molecules", type=int, default=100)
    ap.add_argument("--largest", type=int, default=8)
    ap.add_argument("--batch-sizes", default="2,4,6,8")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--max-cycle", type=int, default=60)
    args = ap.parse_args()

    items = []
    for i in range(args.n_molecules):
        with open(args.cache_dir / f"mol_{i:04d}.pkl", "rb") as f:
            m, e = pickle.load(f)
        ngrid, nao = m.ao.shape
        items.append((ngrid, nao, i, m, e))
    items.sort(reverse=True)
    top = items[: args.largest]
    print("largest molecules (ngrid, nao, idx):",
          [(t[0], t[1], t[2]) for t in top], flush=True)

    dataset = tuple(
        training.MolecularTrainingDatum(
            molecule=t[3], target_e0_total_h=jnp.asarray(t[4], dtype=jnp.float64)
        )
        for t in top
    )

    model = load_skala_functional(key=jax.random.PRNGKey(0))
    functional = SkalaTrainFunctional(model)
    config = training.MolecularTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        e0_total_mse_weight=0.0,
        e0_total_mae_weight=1.0,
        scf_max_cycle=args.max_cycle,
        scf_damping=0.5,
        scf_level_shift=0.0,
        scf_eigenvalue_jitter=1e-8,
        scf_iterate_selection="first_converged",
        scf_conv_tol_energy=1e-9,
        scf_conv_tol_density=1e-8,
        scf_conv_tol_grad=3e-6,
        scf_implicit_diff_tolerance=1e-8,
        scf_implicit_diff_max_iter=20,
        scf_implicit_diff_restart=40,
        scf_convergence_metric="energy_and_residual",
        scf_require_converged=False,
    )
    state = training.trainer.create_train_state_from_molecule(
        functional, jax.random.PRNGKey(0), top[0][3], optax.adam(5e-5)
    )
    train_step = training.trainer.make_molecular_train_step(
        functional, training_config=config
    )

    print(f"baseline peak after init: {peak_gib():.2f} GiB", flush=True)
    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        if bs > len(dataset):
            print(f"batch={bs}: skipped (only {len(dataset)} molecules)", flush=True)
            continue
        try:
            for _ in range(args.steps):
                state, metrics = train_step(state, dataset[:bs])
            loss = float(metrics["total_loss"])
            print(
                f"batch={bs}: OK   peak={peak_gib():8.2f} GiB   "
                f"loss={loss:.4e}   conv={float(metrics['scf_converged_fraction'][0]):.2f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"batch={bs}: FAILED {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            break


if __name__ == "__main__":
    main()
