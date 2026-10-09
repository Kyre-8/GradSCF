"""Per-molecule diagnostic: Skala SCF energy error for every cached molecule."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import training

from skalax_integration.adapter import load_skala_functional
from skalax_integration.train_epoch_sc import SkalaTrainFunctional


jax.config.update("jax_enable_x64", True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--n-molecules", type=int, default=100)
    ap.add_argument("--weights-dir", type=Path, default=None)
    ap.add_argument("--max-cycle", type=int, default=60)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    ds, names = [], []
    for i in range(args.n_molecules):
        with open(args.cache_dir / f"mol_{i:04d}.pkl", "rb") as f:
            m, e = pickle.load(f)
        ds.append(training.MolecularTrainingDatum(
            molecule=m, target_e0_total_h=jnp.asarray(e, dtype=jnp.float64)))
        names.append((i, m.ao.shape[0], m.ao.shape[1], e))

    model = load_skala_functional(args.weights_dir, key=jax.random.PRNGKey(0))
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
    evaluate = training.trainer.make_molecular_eval(functional, training_config=config)
    # The wrapped functional carries the Skala weights as {"params": ...}.
    _, m = evaluate(functional.init_from_molecule(None, None), tuple(ds))

    mae = np.asarray(m["e0_total_mae"]).ravel()
    conv = np.asarray(m["scf_converged"]).ravel()
    cyc = np.asarray(m["scf_cycles"]).ravel()
    pred = np.asarray(m["predicted_e0_total_h"]).ravel()
    ref = np.array([n[3] for n in names])

    order = np.argsort(-mae)
    print(f"n={len(mae)}  converged={conv.mean():.2f}  "
          f"MAE={mae.mean():.4e} Ha  max={mae.max():.4e} Ha", flush=True)
    print(f"{'idx':>4} {'ngrid':>6} {'nao':>4} {'conv':>5} {'cyc':>5} "
          f"{'E_pred':>14} {'E_ref':>14} {'|dE| Ha':>12}", flush=True)
    for k in order[: args.top]:
        i, ng, nao, e = names[k]
        print(f"{i:>4} {ng:>6} {nao:>4} {bool(conv[k])!s:>5} {int(cyc[k]):>5} "
              f"{pred[k]:>14.6f} {e:>14.6f} {mae[k]:>12.4e}", flush=True)
    print("--- best 5 ---", flush=True)
    for k in order[-5:]:
        i, ng, nao, e = names[k]
        print(f"{i:>4} {ng:>6} {nao:>4} {bool(conv[k])!s:>5} {int(cyc[k]):>5} "
              f"{pred[k]:>14.6f} {e:>14.6f} {mae[k]:>12.4e}", flush=True)


if __name__ == "__main__":
    main()
