"""Ten-molecule, 60-epoch SkalaX-TDDFT excitation-only training.

The training path is:
  theta -> E_xc -> V_xc -> implicit self-consistent SCF -> TDA
         -> S1 -> excitation loss -> gradient -> theta.

Each molecule is an independent optimization step. A molecule is skipped for
that step when SCF does not converge or the reverse gradient is nonfinite/zero.
No ground-state energy loss is used.
"""
from __future__ import annotations

import os
import argparse
import csv
import gc
import json
import pickle
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

from gradscf import training as gtraining
from gradscf.workflows.core import run_molecule_from_spec
from gradscf.workflows.types import MoleculeSpecConfig, SimulationConfig


EV = 27.211386245988


def build_molecule(atom: str, basis: str = "6-31g*"):
    spec = MoleculeSpecConfig(
        atom=atom,
        basis=basis,
        xc="b3lyp",
        unit="Angstrom",
        grids_level=int(os.environ.get("SKALA_GRIDS_LEVEL", "2")),
    )
    sim = SimulationConfig(
        scf_backend="jax_rks",
        jax_rks_xc_spec="b3lyp",
        jax_rks_max_cycle=300,
        jax_rks_damping=0.3,
    )
    with jax.default_device(jax.devices("cpu")[0]):
        return run_molecule_from_spec(spec, simulation=sim).molecule


def scalar_metric(metrics: dict, key: str, default=float("nan")) -> float:
    value = metrics.get(key, default)
    arr = jnp.atleast_1d(jnp.asarray(value))
    return float(arr.reshape(-1)[0]) if arr.size else float(default)


def tree_norm(tree) -> float:
    values = jax.tree_util.tree_leaves(tree)
    if not values:
        return 0.0
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in values)))


def save_checkpoint(params, static, optimizer_state, checkpoint_dir: Path):
    from skalax.convert_weights import save_weights_to_npz

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    theta = jax.device_get(params["params"])
    save_weights_to_npz(eqx.combine(theta, static), checkpoint_dir)
    with (checkpoint_dir / "optimizer.pkl").open("wb") as handle:
        pickle.dump(jax.device_get(optimizer_state), handle)


def save_plots(output_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with (output_dir / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    epoch = np.asarray([int(r["epoch"]) for r in rows])
    loss = np.asarray([float(r["loss_ha2"]) for r in rows])
    mae = np.asarray([float(r["mae_ev"]) for r in rows])
    rmse = np.asarray([float(r["rmse_ev"]) for r in rows])
    accepted = np.asarray([int(r["accepted_steps"]) for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=180)
    axes[0, 0].plot(epoch, loss, marker="o")
    axes[0, 0].set(title="Excitation loss", xlabel="Epoch", ylabel=r"Ha$^2$")
    axes[0, 1].plot(epoch, mae, marker="o", label="MAE")
    axes[0, 1].plot(epoch, rmse, marker="s", label="RMSE")
    axes[0, 1].set(title="S1 error", xlabel="Epoch", ylabel="eV")
    axes[0, 1].legend()
    axes[1, 0].plot(epoch, accepted, marker="o")
    axes[1, 0].set(title="Accepted updates", xlabel="Epoch", ylabel="molecules")
    axes[1, 1].axis("off")
    summary = (
        f"Initial MAE: {mae[0]:.4f} eV\n"
        f"Final MAE: {mae[-1]:.4f} eV\n"
        f"Best MAE: {np.nanmin(mae):.4f} eV @ epoch {epoch[np.nanargmin(mae)]}\n"
        f"Accepted final: {accepted[-1]}/10"
    )
    axes[1, 1].text(0.05, 0.5, summary, fontsize=12, va="center")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"loss_curve.{suffix}", bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path.home() / "test_gwbse_10.csv")
    parser.add_argument(
        "--weights-dir", type=Path,
        default=Path.home() / "skala_run_50_def2svp_l2/checkpoints/best",
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n-molecules", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--optimizer-state", type=Path, default=None,
        help="Optional optimizer.pkl to resume Adam/clip state",
    )
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "excited_gwbse10_60")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.output_dir / "checkpoints"

    with args.dataset.open() as handle:
        source_rows = list(csv.DictReader(handle))[: args.n_molecules]
    if not source_rows:
        raise RuntimeError(f"No rows found in {args.dataset}")
    required = {"mol_id", "xyz", "S1_ref_eV", "S1_ref_Ha"}
    missing = required.difference(source_rows[0])
    if missing:
        raise RuntimeError(f"Dataset lacks required fields: {sorted(missing)}")

    print("[SETUP] Building B3LYP references on CPU", flush=True)
    molecules = []
    for row in source_rows:
        molecule = build_molecule(row["xyz"])
        molecules.append(
            {
                "id": str(row["mol_id"]),
                "formula": row.get("formula", ""),
                "mol": molecule,
                "s1_ref_ev": float(row["S1_ref_eV"]),
                "s1_ref_ha": float(row["S1_ref_Ha"]),
            }
        )
        print(
            f"  mol {row['mol_id']} {row.get('formula', ''):8s} "
            f"nao={molecule.ao.shape[1]:3d} ngrid={molecule.ao.shape[0]:5d} "
            f"S1_ref={float(row['S1_ref_eV']):.4f} eV",
            flush=True,
        )

    from skalax_integration.adapter_v2 import load_skala_functional, make_skala_functional_v2

    print("[SETUP] Loading SkalaX checkpoint", flush=True)
    model = load_skala_functional(args.weights_dir)
    theta, static = eqx.partition(model, eqx.is_array)
    functional = make_skala_functional_v2(model, chunk_size=args.chunk_size)
    params = {"params": theta}

    config = gtraining.MolecularTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        e0_total_mse_weight=0.0,
        e0_total_mae_weight=0.0,
        excitation_gap_mse_weight=1.0,
        excitation_gap_mae_weight=0.0,
        excitation_gap_nstates=1,
        excited_state_solver="tda",
        tda_gradient_mode="eigenvalue_only",
        response_two_electron_mode="auto",
        scf_max_cycle=100,
        scf_damping=0.6,
        scf_level_shift=0.0,
        scf_conv_tol_energy=2e-6,
        scf_conv_tol_density=2e-6,
        scf_conv_tol_grad=3e-6,
        scf_iterate_selection="first_converged",
        scf_implicit_diff_tolerance=1e-8,
        scf_implicit_diff_max_iter=20,
        scf_implicit_diff_restart=40,
        scf_convergence_metric="energy",
        scf_require_converged=True,
    )
    dataset = tuple(
        gtraining.MolecularTrainingDatum(
            molecule=item["mol"],
            target_excitation_gaps_h=jnp.asarray([item["s1_ref_ha"]]),
        )
        for item in molecules
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(args.lr),
    )
    optimizer_state = optimizer.init(params)
    if args.optimizer_state is not None:
        if not args.optimizer_state.is_file():
            raise FileNotFoundError(args.optimizer_state)
        with args.optimizer_state.open("rb") as handle:
            optimizer_state = pickle.load(handle)
        print(f"[SETUP] Resumed optimizer state from {args.optimizer_state}", flush=True)
    loss_and_grad = gtraining.trainer.make_molecular_loss_and_grad(
        functional,
        training_config=config,
    )

    theta_before = tree_norm(params["params"])
    print(f"[SETUP] ||theta_before||={theta_before:.10f}", flush=True)
    save_checkpoint(params, static, optimizer_state, checkpoint_root / "initial")

    metrics_fields = [
        "epoch", "loss_ha2", "mae_ev", "rmse_ev", "mean_error_ev",
        "accepted_steps", "skipped_scf", "skipped_nonfinite", "skipped_zero_grad",
        "mean_gradient_norm", "scf_converged_fraction", "param_norm",
        "elapsed_s", "best_mae_ev", "best_epoch",
    ]
    prediction_fields = ["epoch", "mol_id", "S1_ref_eV", "S1_pred_eV", "error_eV"]
    metrics_handle = (args.output_dir / "metrics.csv").open("w", newline="")
    prediction_handle = (args.output_dir / "predictions.csv").open("w", newline="")
    metrics_writer = csv.DictWriter(metrics_handle, fieldnames=metrics_fields)
    prediction_writer = csv.DictWriter(prediction_handle, fieldnames=prediction_fields)
    metrics_writer.writeheader()
    prediction_writer.writeheader()

    log_path = args.output_dir / "train.log"
    best_mae = float("inf")
    best_epoch = 0

    with log_path.open("w") as log_handle:
        def log(message: str):
            print(message, flush=True)
            log_handle.write(message + "\n")
            log_handle.flush()

        log(f"[CONFIG] molecules={len(dataset)} epochs={args.epochs} lr={args.lr} loss=S1_MSE ground_loss=0")
        log("[CONFIG] SCF=self_consistent/implicit TDA=nstates1 convergence=energy")

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            errors_ev = []
            losses = []
            gradient_norms = []
            convergence = []
            accepted = skipped_scf = skipped_nonfinite = skipped_zero = 0

            for item, datum in zip(molecules, dataset):
                try:
                    loss, metrics, grads = loss_and_grad(params, datum)
                    pred_ha = scalar_metric(metrics, "excitation_gap_predicted_h")
                    loss_value = float(loss)
                    mae_value = scalar_metric(metrics, "excitation_gap_mae")
                    grad_norm = scalar_metric(metrics, "grad_norm")
                    nonfinite_fraction = scalar_metric(metrics, "nonfinite_grad_fraction")
                    converged = scalar_metric(metrics, "scf_converged_fraction")
                    pred_ev = pred_ha * EV
                    error_ev = pred_ev - item["s1_ref_ev"]
                    prediction_writer.writerow(
                        {
                            "epoch": epoch,
                            "mol_id": item["id"],
                            "S1_ref_eV": item["s1_ref_ev"],
                            "S1_pred_eV": pred_ev,
                            "error_eV": error_ev,
                        }
                    )

                    if not (np.isfinite(loss_value) and np.isfinite(pred_ha) and converged > 0.5):
                        skipped_scf += 1
                        reason = "SCF/forward invalid"
                    elif nonfinite_fraction > 0.0 or not np.isfinite(grad_norm):
                        skipped_nonfinite += 1
                        reason = f"nonfinite_grad_fraction={nonfinite_fraction:.6f}"
                    elif grad_norm <= 0.0:
                        skipped_zero += 1
                        reason = "zero_gradient"
                    else:
                        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
                        params = optax.apply_updates(params, updates)
                        accepted += 1
                        reason = "update"

                    errors_ev.append(error_ev)
                    losses.append(loss_value)
                    gradient_norms.append(grad_norm)
                    convergence.append(converged)
                    log(
                        f"epoch={epoch:03d} mol={item['id']:>4s} S1={pred_ev:9.4f} "
                        f"ref={item['s1_ref_ev']:9.4f} err={error_ev:+9.4f} "
                        f"loss={loss_value:.6e} grad={grad_norm:.3e} "
                        f"nan={nonfinite_fraction:.3f} conv={converged:.1f} {reason}"
                    )
                except Exception as exc:
                    skipped_nonfinite += 1
                    log(
                        f"epoch={epoch:03d} mol={item['id']:>4s} EXCEPTION "
                        f"{type(exc).__name__}: {str(exc)[:240]}"
                    )
                # Different molecular shapes create different XLA executables.
                # Clear them between molecules so one large/failed compilation
                # cannot poison the GPU allocator for later molecules.
                jax.clear_caches()
                gc.collect()

            valid_errors = np.asarray([x for x in errors_ev if np.isfinite(x)], dtype=float)
            loss_epoch = float(np.mean([x for x in losses if np.isfinite(x)])) if losses else float("nan")
            mae = float(np.mean(np.abs(valid_errors))) if valid_errors.size else float("nan")
            rmse = float(np.sqrt(np.mean(np.square(valid_errors)))) if valid_errors.size else float("nan")
            mean_error = float(np.mean(valid_errors)) if valid_errors.size else float("nan")
            mean_grad = float(np.mean([x for x in gradient_norms if np.isfinite(x)])) if gradient_norms else float("nan")
            mean_conv = float(np.mean([x for x in convergence if np.isfinite(x)])) if convergence else float("nan")
            param_norm = tree_norm(params["params"])
            elapsed = time.time() - epoch_start

            if np.isfinite(mae) and mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                save_checkpoint(params, static, optimizer_state, checkpoint_root / "best")
            if args.checkpoint_every and epoch % args.checkpoint_every == 0:
                save_checkpoint(params, static, optimizer_state, checkpoint_root / f"epoch_{epoch:03d}")
            save_checkpoint(params, static, optimizer_state, checkpoint_root / "latest")

            metrics_writer.writerow(
                {
                    "epoch": epoch,
                    "loss_ha2": loss_epoch,
                    "mae_ev": mae,
                    "rmse_ev": rmse,
                    "mean_error_ev": mean_error,
                    "accepted_steps": accepted,
                    "skipped_scf": skipped_scf,
                    "skipped_nonfinite": skipped_nonfinite,
                    "skipped_zero_grad": skipped_zero,
                    "mean_gradient_norm": mean_grad,
                    "scf_converged_fraction": mean_conv,
                    "param_norm": param_norm,
                    "elapsed_s": elapsed,
                    "best_mae_ev": best_mae,
                    "best_epoch": best_epoch,
                }
            )
            metrics_handle.flush()
            prediction_handle.flush()
            log(
                f"epoch={epoch:03d} SUMMARY loss={loss_epoch:.6e} MAE={mae:.4f}eV "
                f"RMSE={rmse:.4f}eV accepted={accepted}/{len(dataset)} "
                f"skip(scf/nan/zero)=({skipped_scf}/{skipped_nonfinite}/{skipped_zero}) "
                f"best={best_mae:.4f}eV@{best_epoch:03d} time={elapsed:.1f}s"
            )

    save_checkpoint(params, static, optimizer_state, checkpoint_root / "final")
    metrics_handle.close()
    prediction_handle.close()
    save_plots(args.output_dir)
    theta_after = tree_norm(params["params"])
    with (args.output_dir / "parameter_change.txt").open("w") as handle:
        handle.write(f"theta_before = {theta_before:.12f}\n")
        handle.write(f"theta_after  = {theta_after:.12f}\n")
        handle.write(f"delta_norm   = {theta_after - theta_before:.12f}\n")
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                "dataset": str(args.dataset),
                "n_molecules": len(dataset),
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "chunk_size": args.chunk_size,
                "loss": "S1 excitation MSE only",
                "ground_state_loss": "disabled",
                "scf_mode": "self_consistent implicit",
                "scf_convergence_metric": "energy",
                "response": "TDA, S1, SkalaX ao_kernel_action",
            },
            handle,
            indent=2,
        )
    print(f"[DONE] output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
