"""Subprocess-isolated SkalaX-TDDFT excitation training.

The controller runs on CPU.  Every molecular optimization step is executed in
its own Python/GPU subprocess.  Exiting that subprocess releases the JAX/XLA
GPU context completely, which prevents one molecular shape from poisoning the
allocator for later molecules.

Training path in each worker:
  theta -> E_xc -> V_xc -> implicit self-consistent SCF
        -> TDA -> S1 -> loss -> gradient -> theta
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


EV = 27.211386245988


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_plots(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    with (output_dir / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    epoch = np.asarray([int(row["epoch"]) for row in rows])
    loss = np.asarray([float(row["loss_ha2"]) for row in rows])
    mae = np.asarray([float(row["mae_ev"]) for row in rows])
    rmse = np.asarray([float(row["rmse_ev"]) for row in rows])
    accepted = np.asarray([int(row["accepted_steps"]) for row in rows])

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
    axes[1, 1].text(
        0.05,
        0.5,
        f"Initial MAE: {mae[0]:.4f} eV\n"
        f"Final MAE: {mae[-1]:.4f} eV\n"
        f"Best MAE: {np.nanmin(mae):.4f} eV @ epoch {epoch[np.nanargmin(mae)]}\n"
        f"Accepted final: {accepted[-1]}/10",
        fontsize=12,
        va="center",
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"loss_curve.{suffix}", bbox_inches="tight")


class WorkerController:
    def __init__(
        self,
        *,
        dataset: Path,
        weights_dir: Path,
        output_dir: Path,
        chunk_size: int,
        learning_rate: float,
    ):
        self.dataset = dataset
        self.weights_dir = weights_dir
        self.output_dir = output_dir
        self.state_dir = output_dir / "state"
        self.worker_dir = output_dir / "workers"
        self.chunk_size = chunk_size
        self.learning_rate = learning_rate
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.worker_dir.mkdir(parents=True, exist_ok=True)

    def call(
        self,
        arguments: list[str],
        log_path: Path,
        result_path: Path | None = None,
    ) -> dict:
        command = [
            sys.executable,
            "-m",
            "skalax_integration.train_excited_subprocess",
            *map(str, arguments),
        ]
        environment = os.environ.copy()
        environment["JAX_PLATFORM_NAME"] = "gpu"
        environment["PYTHONUNBUFFERED"] = "1"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if result_path is not None:
            result_path.unlink(missing_ok=True)
        with log_path.open("w") as log_handle:
            process = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        if result_path is None:
            return {"return_code": process.returncode}
        if process.returncode != 0 or not result_path.is_file():
            message = f"worker return_code={process.returncode}"
            if log_path.exists():
                tail = log_path.read_text(errors="replace").splitlines()[-20:]
                message += "\n" + "\n".join(tail)
            return {
                "accepted": False,
                "category": "worker_failure" if process.returncode else "no_result",
                "exception": message,
                "return_code": process.returncode,
            }
        result = read_json(result_path)
        result["return_code"] = process.returncode
        return result

    def prepare(self, resume_optimizer: Path | None) -> dict:
        result_path = self.output_dir / "prepare_result.json"
        arguments = [
            "--mode", "prepare",
            "--dataset", self.dataset,
            "--weights-dir", self.weights_dir,
            "--state-dir", self.state_dir,
            "--learning-rate", self.learning_rate,
            "--result-path", result_path,
        ]
        if resume_optimizer is not None:
            arguments.extend(("--resume-optimizer", resume_optimizer))
        result = self.call(
            arguments,
            self.worker_dir / "prepare.log",
            result_path,
        )
        result_path.unlink(missing_ok=True)
        return result

    def validate(self, epoch: int, val_dataset: Path, n_val_rows: int) -> list[dict]:
        results = []
        for row_index in range(n_val_rows):
            result_path = self.state_dir / f"val_result_epoch_{epoch:03d}_mol_{row_index:02d}.json"
            arguments = [
                "--mode", "eval",
                "--dataset", val_dataset,
                "--weights-dir", self.weights_dir,
                "--state-dir", self.state_dir,
                "--chunk-size", self.chunk_size,
                "--learning-rate", self.learning_rate,
                "--row-index", row_index,
                "--result-path", result_path,
            ]
            result = self.call(
                arguments,
                self.worker_dir / f"val_epoch_{epoch:03d}_mol_{row_index:02d}.log",
                result_path,
            )
            result_path.unlink(missing_ok=True)
            results.append(result)
        return results

    def step(self, epoch: int, row_index: int, mol_id: str) -> dict:
        result_path = self.state_dir / f"result_epoch_{epoch:03d}_mol_{row_index:02d}.json"
        arguments = [
            "--mode", "step",
            "--dataset", self.dataset,
            "--weights-dir", self.weights_dir,
            "--state-dir", self.state_dir,
            "--chunk-size", self.chunk_size,
            "--learning-rate", self.learning_rate,
            "--row-index", row_index,
            "--result-path", result_path,
        ]
        result = self.call(
            arguments,
            self.worker_dir / f"epoch_{epoch:03d}_mol_{mol_id}.log",
            result_path,
        )
        result_path.unlink(missing_ok=True)
        return result

    def snapshot(self, checkpoint_dir: Path) -> dict:
        result_path = checkpoint_dir / "snapshot_result.json"
        arguments = [
            "--mode", "snapshot",
            "--dataset", self.dataset,
            "--weights-dir", self.weights_dir,
            "--state-dir", self.state_dir,
            "--checkpoint-dir", checkpoint_dir,
            "--result-path", result_path,
        ]
        result = self.call(
            arguments,
            checkpoint_dir / "snapshot.log",
            result_path,
        )
        result_path.unlink(missing_ok=True)
        return result


def run_controller(args: argparse.Namespace) -> None:
    with args.dataset.open() as handle:
        rows = list(csv.DictReader(handle))[: args.n_molecules]
    if len(rows) != args.n_molecules:
        raise RuntimeError(
            f"Requested {args.n_molecules} molecules, dataset has {len(rows)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    controller = WorkerController(
        dataset=args.dataset,
        weights_dir=args.weights_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        learning_rate=args.lr,
    )
    resume_optimizer = args.resume_optimizer if args.resume_optimizer else None
    prepare_result = controller.prepare(resume_optimizer)
    print(
        f"[SETUP] prepared state; ||theta||={prepare_result.get('param_norm', float('nan')):.10f}",
        flush=True,
    )
    if args.prepare_only:
        return

    metrics_fields = [
        "epoch", "loss_ha2", "mae_ev", "rmse_ev", "mean_error_ev",
        "accepted_steps", "forward_valid", "skipped_scf",
        "skipped_nonfinite", "skipped_zero_grad", "oom", "worker_failure",
        "mean_gradient_norm", "scf_converged_fraction", "param_norm",
        "elapsed_s", "best_mae_ev", "best_epoch",
    ]
    prediction_fields = [
        "epoch", "row_index", "mol_id", "formula", "S1_ref_eV",
        "S1_pred_eV", "error_eV", "gradient_norm",
        "nonfinite_grad_fraction", "scf_converged", "accepted", "category",
    ]
    metrics_path = args.output_dir / "metrics.csv"
    val_metrics_path = args.output_dir / "val_metrics.csv"
    val_predictions_path = args.output_dir / "val_predictions.csv"
    predictions_path = args.output_dir / "predictions.csv"
    train_log_path = args.output_dir / "train.log"
    metrics_path.unlink(missing_ok=True)
    val_metrics_path.unlink(missing_ok=True)
    val_predictions_path.unlink(missing_ok=True)
    predictions_path.unlink(missing_ok=True)
    train_log_path.unlink(missing_ok=True)

    best_mae = float("inf")
    best_epoch = 0
    initial_norm = float(prepare_result.get("param_norm", float("nan")))
    final_norm = initial_norm

    with train_log_path.open("w", buffering=1) as log_handle:
        def log(message: str):
            print(message, flush=True)
            log_handle.write(message + "\n")

        log(
            f"[CONFIG] molecules={len(rows)} epochs={args.epochs} "
            f"lr={args.lr} chunk={args.chunk_size} loss=S1_MSE ground_loss=0"
        )
        log("[CONFIG] one GPU subprocess per molecular optimization step")
        log("[CONFIG] SCF=self_consistent/implicit TDA=nstates1 convergence=energy")
        if args.val_dataset is not None:
            log(f"[CONFIG] validation: {args.val_dataset} ({args.val_n_molecules} molecules, every epoch)")

        for local_epoch in range(1, args.epochs + 1):
            epoch = local_epoch + args.epoch_offset
            epoch_start = time.time()
            errors = []
            losses = []
            gradients = []
            convergence = []
            accepted_count = 0
            forward_valid = 0
            skipped_scf = 0
            skipped_nonfinite = 0
            skipped_zero = 0
            oom_count = 0
            worker_failure = 0

            for row_index, row in enumerate(rows):
                mol_id = str(row["mol_id"])
                result = controller.step(epoch, row_index, mol_id)
                category = str(result.get("category", "unknown"))
                accepted = bool(result.get("accepted", False))
                pred_ev = float(result.get("s1_pred_ev", float("nan")))
                ref_ev = float(result["s1_ref_ev"]) if "s1_ref_ev" in result else float(row["S1_ref_eV"])
                error_ev = pred_ev - ref_ev
                gradient_norm = float(result.get("gradient_norm", float("nan")))
                nonfinite = float(result.get("nonfinite_grad_fraction", float("nan")))
                converged = float(result.get("scf_converged", 0.0))

                if "s1_pred_ev" in result and result.get("s1_pred_ev") is not None:
                    forward_valid += 1
                    errors.append(error_ev)
                if "loss_ha2" in result and result.get("loss_ha2") is not None:
                    losses.append(float(result["loss_ha2"]))
                if "gradient_norm" in result and result.get("gradient_norm") is not None:
                    gradients.append(gradient_norm)
                if "scf_converged" in result and result.get("scf_converged") is not None:
                    convergence.append(converged)
                if accepted:
                    accepted_count += 1
                if category in {"scf_invalid", "forward_invalid"}:
                    skipped_scf += 1
                elif category in {"nonfinite_gradient", "nonfinite_loss"}:
                    skipped_nonfinite += 1
                elif category == "zero_gradient":
                    skipped_zero += 1
                elif category == "oom":
                    oom_count += 1
                elif category in {"worker_failure", "no_result", "unknown"}:
                    worker_failure += 1

                append_csv(
                    predictions_path,
                    prediction_fields,
                    {
                        "epoch": epoch,
                        "row_index": row_index,
                        "mol_id": mol_id,
                        "formula": row.get("formula", ""),
                        "S1_ref_eV": ref_ev,
                        "S1_pred_eV": pred_ev,
                        "error_eV": error_ev,
                        "gradient_norm": gradient_norm,
                        "nonfinite_grad_fraction": nonfinite,
                        "scf_converged": converged,
                        "accepted": int(accepted),
                        "category": category,
                    },
                )
                exception = str(result.get("exception", ""))
                log(
                    f"epoch={epoch:03d} mol={mol_id:>4s} "
                    f"S1={pred_ev:9.4f} ref={ref_ev:9.4f} err={error_ev:+9.4f} "
                    f"loss={float(result.get('loss_ha2', float('nan'))):.6e} "
                    f"grad={gradient_norm:.3e} nan={nonfinite:.3f} "
                    f"conv={converged:.1f} {category}"
                    + (f" :: {exception[-180:]}" if exception else "")
                )

            valid_errors = [value for value in errors if value == value and abs(value) != float("inf")]
            valid_losses = [value for value in losses if value == value and abs(value) != float("inf")]
            valid_gradients = [value for value in gradients if value == value and abs(value) != float("inf")]
            valid_convergence = [value for value in convergence if value == value and abs(value) != float("inf")]
            loss_epoch = sum(valid_losses) / len(valid_losses) if valid_losses else float("nan")
            mae = sum(abs(value) for value in valid_errors) / len(valid_errors) if valid_errors else float("nan")
            rmse = (sum(value * value for value in valid_errors) / len(valid_errors)) ** 0.5 if valid_errors else float("nan")
            mean_error = sum(valid_errors) / len(valid_errors) if valid_errors else float("nan")
            mean_gradient = sum(valid_gradients) / len(valid_gradients) if valid_gradients else float("nan")
            mean_convergence = sum(valid_convergence) / len(valid_convergence) if valid_convergence else float("nan")

            val_mae = val_rmse = val_mean = float("nan")
            val_valid = 0
            if args.val_dataset is not None and args.val_n_molecules > 0:
                val_results = controller.validate(epoch, args.val_dataset, args.val_n_molecules)
                val_errors = []
                for val_index, res in enumerate(val_results):
                    pred_v = res.get("s1_pred_ev")
                    ref_v = res.get("s1_ref_ev")
                    if pred_v is None or ref_v is None:
                        continue
                    err_v = float(pred_v) - float(ref_v)
                    val_errors.append(err_v)
                    append_csv(
                        val_predictions_path,
                        ["epoch", "row_index", "mol_id", "formula", "S1_ref_eV", "S1_pred_eV", "error_eV"],
                        {
                            "epoch": epoch,
                            "row_index": val_index,
                            "mol_id": res.get("mol_id", ""),
                            "formula": res.get("formula", ""),
                            "S1_ref_eV": float(ref_v),
                            "S1_pred_eV": float(pred_v),
                            "error_eV": err_v,
                        },
                    )
                if val_errors:
                    val_valid = len(val_errors)
                    val_mae = sum(abs(v) for v in val_errors) / len(val_errors)
                    val_rmse = (sum(v * v for v in val_errors) / len(val_errors)) ** 0.5
                    val_mean = sum(val_errors) / len(val_errors)
                append_csv(
                    val_metrics_path,
                    ["epoch", "n_valid", "val_mae_ev", "val_rmse_ev", "val_mean_ev"],
                    {
                        "epoch": epoch,
                        "n_valid": val_valid,
                        "val_mae_ev": val_mae,
                        "val_rmse_ev": val_rmse,
                        "val_mean_ev": val_mean,
                    },
                )

            if mae == mae and mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                snapshot = controller.snapshot(args.output_dir / "checkpoints" / "best")
            else:
                snapshot = controller.snapshot(args.output_dir / "checkpoints" / "latest")
            final_norm = float(snapshot.get("param_norm", float("nan")))
            elapsed = time.time() - epoch_start

            append_csv(
                metrics_path,
                metrics_fields,
                {
                    "epoch": epoch,
                    "loss_ha2": loss_epoch,
                    "mae_ev": mae,
                    "rmse_ev": rmse,
                    "mean_error_ev": mean_error,
                    "accepted_steps": accepted_count,
                    "forward_valid": forward_valid,
                    "skipped_scf": skipped_scf,
                    "skipped_nonfinite": skipped_nonfinite,
                    "skipped_zero_grad": skipped_zero,
                    "oom": oom_count,
                    "worker_failure": worker_failure,
                    "mean_gradient_norm": mean_gradient,
                    "scf_converged_fraction": mean_convergence,
                    "param_norm": final_norm,
                    "elapsed_s": elapsed,
                    "best_mae_ev": best_mae,
                    "best_epoch": best_epoch,
                },
            )
            if args.checkpoint_every and epoch % args.checkpoint_every == 0:
                controller.snapshot(args.output_dir / "checkpoints" / f"epoch_{epoch:03d}")
            log(
                f"epoch={epoch:03d} SUMMARY loss={loss_epoch:.6e} MAE={mae:.4f}eV "
                f"RMSE={rmse:.4f}eV accepted={accepted_count}/{len(rows)} "
                f"forward={forward_valid}/{len(rows)} "
                f"oom={oom_count} best={best_mae:.4f}eV@{best_epoch:03d} "
                f"||theta||={final_norm:.10f} time={elapsed:.1f}s"
                + (f" VAL_MAE={val_mae:.4f}eV VAL_RMSE={val_rmse:.4f}eV"
                   if val_valid else "")
            )

    controller.snapshot(args.output_dir / "checkpoints" / "final")
    with (args.output_dir / "parameter_change.txt").open("w") as handle:
        handle.write(f"theta_before = {initial_norm:.12f}\n")
        handle.write(f"theta_after  = {final_norm:.12f}\n")
        handle.write(f"delta_norm   = {final_norm - initial_norm:.12f}\n")
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                "dataset": str(args.dataset),
                "n_molecules": len(rows),
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "chunk_size": args.chunk_size,
                "loss": "S1 excitation MSE only",
                "ground_state_loss": "disabled",
                "scf_mode": "self_consistent implicit",
                "scf_convergence_metric": "energy",
                "response": "TDA, S1, SkalaX ao_kernel_action",
                "process_isolation": "one GPU subprocess per molecule-step",
            },
            handle,
            indent=2,
        )
    save_plots(args.output_dir)
    print(f"[DONE] output={args.output_dir}", flush=True)


def run_worker(args: argparse.Namespace) -> None:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
    import pickle

    jax.config.update("jax_enable_x64", True)

    from gradscf import training as gtraining
    from gradscf.workflows.core import run_molecule_from_spec
    from gradscf.workflows.types import MoleculeSpecConfig, SimulationConfig
    from skalax_integration.adapter_v2 import load_skala_functional, make_skala_functional_v2

    def tree_norm(tree) -> float:
        values = jax.tree_util.tree_leaves(tree)
        if not values:
            return 0.0
        return float(jnp.sqrt(sum(jnp.sum(jnp.square(value)) for value in values)))

    def load_functional():
        model = load_skala_functional(args.weights_dir)
        _, static = eqx.partition(model, eqx.is_array)
        functional = make_skala_functional_v2(model, chunk_size=args.chunk_size)
        return functional, static

    def _theta_finite(theta) -> bool:
        return all(
            bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))
            for leaf in jax.tree_util.tree_leaves(theta)
        )

    def load_state():
        with (args.state_dir / "params.pkl").open("rb") as handle:
            theta = pickle.load(handle)
        repaired = False
        if not _theta_finite(theta):
            print("[WARN] state params.pkl is non-finite; attempting recovery", flush=True)
            candidates = sorted(
                (args.output_dir / "checkpoints").glob("epoch_*"), reverse=True
            )
            for candidate in candidates:
                try:
                    theta_c = eqx.partition(load_skala_functional(candidate), eqx.is_array)[0]
                except Exception as exc:  # incomplete or unreadable checkpoint
                    print(f"[WARN]   skip {candidate.name}: {type(exc).__name__}", flush=True)
                    continue
                if _theta_finite(theta_c):
                    theta = theta_c
                    repaired = True
                    print(f"[WARN]   restored parameters from {candidate}", flush=True)
                    break
            if not repaired:
                theta = eqx.partition(load_skala_functional(args.weights_dir), eqx.is_array)[0]
                repaired = True
                print("[WARN]   no usable checkpoint; re-initialised from --weights-dir", flush=True)
        with (args.state_dir / "optimizer.pkl").open("rb") as handle:
            optimizer_state = pickle.load(handle)
        if repaired:
            optimizer_state = make_optimizer().init({"params": theta})
            save_state({"params": theta}, optimizer_state)
            print("[WARN]   state repaired and persisted", flush=True)
        return {"params": theta}, optimizer_state

    def save_state(params, optimizer_state) -> None:
        theta = jax.device_get(params["params"])
        optimizer_cpu = jax.device_get(optimizer_state)
        params_tmp = args.state_dir / "params.pkl.tmp"
        optimizer_tmp = args.state_dir / "optimizer.pkl.tmp"
        with params_tmp.open("wb") as handle:
            pickle.dump(theta, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with optimizer_tmp.open("wb") as handle:
            pickle.dump(optimizer_cpu, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(params_tmp, args.state_dir / "params.pkl")
        os.replace(optimizer_tmp, args.state_dir / "optimizer.pkl")

    def make_optimizer():
        return optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(args.learning_rate),
        )

    if args.mode == "prepare":
        functional, static = load_functional()
        model = load_skala_functional(args.weights_dir)
        theta, _ = eqx.partition(model, eqx.is_array)
        params = {"params": theta}
        optimizer = make_optimizer()
        optimizer_state = optimizer.init(params)
        if args.resume_optimizer is not None:
            with args.resume_optimizer.open("rb") as handle:
                optimizer_state = pickle.load(handle)
        save_state(params, optimizer_state)
        write_json(
            args.result_path,
            {
                "param_norm": tree_norm(params["params"]),
                "resumed_optimizer": bool(args.resume_optimizer is not None),
            },
        )
        return

    if args.mode == "snapshot":
        from skalax.convert_weights import save_weights_to_npz

        functional, static = load_functional()
        params, _ = load_state()
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_weights_to_npz(eqx.combine(params["params"], static), args.checkpoint_dir)
        shutil.copy2(args.state_dir / "optimizer.pkl", args.checkpoint_dir / "optimizer.pkl")
        write_json(args.result_path, {"param_norm": tree_norm(params["params"])})
        return

    with args.dataset.open() as handle:
        rows = list(csv.DictReader(handle))
    row = rows[int(args.row_index)]
    atom = row["xyz"]
    spec = MoleculeSpecConfig(
        atom=atom,
        basis="6-31g*",
        xc="b3lyp",
        unit="Angstrom",
        grids_level=int(os.environ.get("SKALA_GRIDS_LEVEL", "2")),
    )
    simulation = SimulationConfig(
        scf_backend="jax_rks",
        jax_rks_xc_spec="b3lyp",
        jax_rks_max_cycle=300,
        jax_rks_damping=0.3,
    )
    with jax.default_device(jax.devices("cpu")[0]):
        molecule = run_molecule_from_spec(spec, simulation=simulation).molecule

    functional, static = load_functional()

    if args.mode == "eval":
        params_eval, _ = load_state()
        eval_config = gtraining.MolecularTrainingConfig(
            mode="self_consistent",
            scf_gradient_mode="impl",
            e0_total_mse_weight=0.0,
            e0_total_mae_weight=0.0,
            excitation_gap_mse_weight=1.0,
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
            scf_convergence_metric="energy",
            scf_require_converged=True,
        )
        eval_datum = gtraining.MolecularTrainingDatum(
            molecule=molecule,
            target_excitation_gaps_h=jnp.asarray([float(row["S1_ref_Ha"])]),
        )
        _eval_loss, eval_metrics = gtraining.molecular_loss(
            params_eval, functional, eval_datum, training_config=eval_config
        )
        pred_ev = float(eval_metrics["excitation_gap_predicted_h"][0]) * 27.211386245988
        ref_ev = float(row["S1_ref_eV"])
        write_json(
            args.result_path,
            {
                "mol_id": str(row["mol_id"]),
                "formula": str(row.get("formula", "")),
                "s1_ref_ev": ref_ev,
                "s1_pred_ev": pred_ev,
                "error_ev": pred_ev - ref_ev,
            },
        )
        return

    params, optimizer_state = load_state()
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
    datum = gtraining.MolecularTrainingDatum(
        molecule=molecule,
        target_excitation_gaps_h=jnp.asarray([float(row["S1_ref_Ha"])]),
    )
    loss_and_grad = gtraining.trainer.make_molecular_loss_and_grad(
        functional,
        training_config=config,
    )

    def scalar(metrics: dict, key: str) -> float:
        value = jnp.atleast_1d(jnp.asarray(metrics.get(key, jnp.nan)))
        return float(value.reshape(-1)[0]) if value.size else float("nan")

    result = {
        "row_index": int(args.row_index),
        "mol_id": str(row["mol_id"]),
        "formula": row.get("formula", ""),
        "s1_ref_ev": float(row["S1_ref_eV"]),
        "accepted": False,
        "category": "unknown",
    }
    try:
        loss, metrics, grads = loss_and_grad(params, datum)
        pred_ha = scalar(metrics, "excitation_gap_predicted_h")
        loss_value = float(loss)
        gradient_norm = scalar(metrics, "grad_norm")
        nonfinite_fraction = scalar(metrics, "nonfinite_grad_fraction")
        converged = scalar(metrics, "scf_converged_fraction")
        result.update(
            {
                "s1_pred_ev": pred_ha * EV,
                "loss_ha2": loss_value,
                "gradient_norm": gradient_norm,
                "nonfinite_grad_fraction": nonfinite_fraction,
                "scf_converged": converged,
                "param_norm_before": tree_norm(params["params"]),
            }
        )
        if not (loss_value == loss_value and abs(loss_value) != float("inf") and pred_ha == pred_ha):
            result["category"] = "forward_invalid"
        elif converged <= 0.5:
            result["category"] = "scf_invalid"
        elif nonfinite_fraction > 0.0 or gradient_norm != gradient_norm or abs(gradient_norm) == float("inf"):
            result["category"] = "nonfinite_gradient"
        elif gradient_norm <= 0.0:
            result["category"] = "zero_gradient"
        else:
            optimizer = make_optimizer()
            updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
            candidate = optax.apply_updates(params, updates)
            candidate_norm = tree_norm(candidate["params"])
            finite = candidate_norm == candidate_norm and abs(candidate_norm) != float("inf")
            if not finite:
                # never persist a non-finite parameter tree: it poisons every later molecule
                result["category"] = "nonfinite_update"
                result["param_norm_after"] = candidate_norm
            else:
                params = candidate
                save_state(params, optimizer_state)
                result.update(
                    {
                        "accepted": True,
                        "category": "update",
                        "param_norm_after": candidate_norm,
                    }
                )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        result["exception"] = message
        result["category"] = "oom" if "RESOURCE_EXHAUSTED" in message else "worker_failure"
    write_json(args.result_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("controller", "prepare", "step", "snapshot", "eval"), default="controller")
    parser.add_argument("--dataset", type=Path, default=Path.home() / "test_gwbse_nao60_10.csv")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--resume-optimizer", type=Path)
    parser.add_argument("--row-index", type=int)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-molecules", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--val-dataset", type=Path)
    parser.add_argument("--val-n-molecules", type=int, default=0)
    parser.add_argument(
        "--epoch-offset", type=int, default=0,
        help="Label the first epoch as this global epoch plus one",
    )
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "excited_gwbse10_subprocess")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    # Backwards-compatible alias used by the controller.
    args.lr = args.learning_rate

    if args.mode == "controller":
        run_controller(args)
    else:
        if args.result_path is None:
            parser.error("--result-path is required for worker modes")
        run_worker(args)


if __name__ == "__main__":
    main()
