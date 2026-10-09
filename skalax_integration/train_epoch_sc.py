"""Skala self-consistent training: original adapter/features + training-module driver.

* Feature construction is the ORIGINAL ``skalax_integration.features.build_skala_features``
  (total density matrix -> rho/grad/tau -> ``SkalaFunctional.get_exc``).
* The SCF is driven by ``gradscf.training`` (``make_molecular_train_step`` /
  ``make_molecular_eval``) with ``scf_gradient_mode="impl"``; this keeps the
  compiled differentiation graph small enough for the GPU (running
  ``DifferentiableSCF.run`` eagerly inside ``filter_value_and_grad`` blew up to
  ~30 GB per step).
* Loss = MAE on the total energy.  Metrics are reported per EPOCH over ALL
  training molecules.
* ``eigenvalue_jitter`` stays at its default (1e-8): a large value (1e-3) adds a
  ramped diagonal shift that floors the density residual and breaks SCF
  convergence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from gradscf import training
from gradscf.workflows.core import run_molecule_from_spec
from gradscf.workflows.types import MoleculeSpecConfig, SimulationConfig

from skalax_integration.adapter import load_skala_functional
from skalax_integration.features import build_skala_features


jax.config.update("jax_enable_x64", True)


def gpu_mem_mib() -> int:
    """Device memory currently used by this job's GPU (MiB), or -1 if unknown."""
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", dev, "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1


class SkalaTrainFunctional:
    """Original Skala adapter semantics, wrapped for the ``gradscf.training`` driver."""

    require_converged_scf = False

    def __init__(self, model):
        self.model = model
        self._params, self._static = eqx.partition(model, eqx.is_array)

    def init_from_molecule(self, rng, molecule):
        del rng, molecule
        return {"params": self._params}

    def _rebuild(self, params):
        p = params["params"] if isinstance(params, dict) and "params" in params else params
        return eqx.combine(p, self._static)

    def scf_xc_energy_and_alpha_for_density(self, params, molecule, density):
        exc = self._rebuild(params).get_exc(build_skala_features(molecule, density))
        return exc, jnp.asarray(0.0, dtype=jnp.asarray(density).dtype)

    def energy_from_molecule(self, params, molecule):
        rdm1 = jnp.asarray(molecule.rdm1)
        density = rdm1.sum(axis=0) if rdm1.ndim == 3 else rdm1
        return self.scf_xc_energy_and_alpha_for_density(params, molecule, density)[0]

    def apply(self, params, inputs):
        return self._rebuild(params).get_exc(inputs)


def save_skala_checkpoint(state, static, checkpoint_dir: Path) -> Path:
    from skalax.convert_weights import save_weights_to_npz

    params = jax.device_get(state.params["params"])
    model = eqx.combine(params, static)
    save_weights_to_npz(model, checkpoint_dir)
    return checkpoint_dir


def parse_xyz(path: Path, max_molecules: int | None) -> list[str]:
    lines = path.read_text().splitlines()
    specs: list[str] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        natom = int(lines[i].strip())
        atoms = []
        for ln in lines[i + 2 : i + 2 + natom]:
            f = ln.split()
            atoms.append(f"{f[0]} {f[1]} {f[2]} {f[3]}")
        specs.append("; ".join(atoms))
        i += natom + 2
        if max_molecules is not None and len(specs) >= max_molecules:
            break
    return specs


def build_reference(
    atom: str,
    basis: str,
    xc: str,
    max_cycle: int = 300,
    damping: float = 0.3,
    grids_level: int = 0,
    jk_backend: str = "full",
    df_tol: float = 1e-10,
    df_max_rank: int | None = None,
):
    """Reference (B3LYP) molecule.  The framework defaults (50 cycles, damping
    0.15) fail to converge for the larger molecules, which silently produces a
    garbage target energy; use a longer, more damped reference SCF."""
    spec = MoleculeSpecConfig(
        atom=atom, basis=basis, xc=xc, unit="Angstrom", grids_level=grids_level
    )
    sim = SimulationConfig(
        scf_backend="jax_rks",
        jax_rks_xc_spec=xc,
        jax_rks_max_cycle=max_cycle,
        jax_rks_damping=damping,
        jax_rks_jk_backend=jk_backend,
        jax_rks_df_tol=df_tol,
        jax_rks_df_max_rank=df_max_rank,
    )
    run = run_molecule_from_spec(spec, simulation=sim)
    return run.molecule


def _mean(metrics, key):
    return float(jnp.mean(jnp.asarray(metrics[key])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz-file", type=Path, required=True)
    ap.add_argument("--max-molecules", type=int, default=None)
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--ref-xc", default="b3lyp")
    ap.add_argument("--grids-level", type=int, default=0)
    ap.add_argument("--jk-backend", choices=["full", "df", "direct"], default="full",
                    help="Two-electron J/K backend for the reference build. "
                    "'df' = density fitting / low-rank factorisation.")
    ap.add_argument("--df-tol", type=float, default=1e-10)
    ap.add_argument("--df-max-rank", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-cycle", type=int, default=40)
    ap.add_argument("--damping", type=float, default=0.5)
    ap.add_argument("--level-shift", type=float, default=0.0)
    ap.add_argument("--eigenvalue-jitter", type=float, default=1e-8)
    ap.add_argument("--iterate-selection", default="first_converged")
    ap.add_argument("--conv-tol-energy", type=float, default=1e-9)
    ap.add_argument("--conv-tol-density", type=float, default=1e-8)
    ap.add_argument(
        "--conv-tol-grad",
        type=float,
        default=3e-6,
        help="MO-gradient residual tolerance. Framework default 1e-7 is below "
        "the numerical noise floor of the grid-based Skala potential, so "
        "molecules stall at ~1e-6 and are never marked converged.",
    )
    ap.add_argument("--implicit-diff-tolerance", type=float, default=1e-8,
                    help="Adjoint (backward) linear-solve tolerance.")
    ap.add_argument("--implicit-diff-max-iter", type=int, default=20)
    ap.add_argument("--implicit-diff-restart", type=int, default=40)
    ap.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    ap.add_argument("--ref-cache-dir", type=Path, default=None,
                    help="Reuse an existing ref_cache directory (pre-built molecules).")
    ap.add_argument("--weights-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("skala_epoch_out"))
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument(
        "--clear-caches-every",
        type=int,
        default=8,
        help="Call jax.clear_caches() every N training batches to release the "
        "per-molecule-shape compiled executables (they are what exhausts the "
        "GPU). 0 disables it.",
    )
    ap.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=None,
        help="Persistent on-disk JAX compilation cache, so re-compiling a shape "
        "after clear_caches() is a fast cache hit instead of a full compile.",
    )
    ap.add_argument("--log-append", action="store_true",
                    help="Append to an existing log.csv instead of overwriting it.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.compile_cache_dir is not None:
        args.compile_cache_dir.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(args.compile_cache_dir))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        print(f"compilation cache dir: {args.compile_cache_dir}", flush=True)

    # The GPUs on this cluster are shared between users, so report how much of
    # this job's card is already taken before we start allocating.
    try:
        _dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
        _q = subprocess.run(
            ["nvidia-smi", "-i", _dev, "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        print(f"GPU {_dev} before start: used/total = {_q} MiB", flush=True)
    except Exception:  # noqa: BLE001
        pass

    specs = parse_xyz(args.xyz_file, args.max_molecules)
    print(f"molecules: {len(specs)}", flush=True)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.ref_cache_dir if args.ref_cache_dir is not None else out / "ref_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"ref_cache_dir: {cache_dir}", flush=True)
    # Guard against silently reusing a cache built with different settings.
    settings = {
        "basis": args.basis, "ref_xc": args.ref_xc, "grids_level": args.grids_level,
        "jk_backend": args.jk_backend, "df_tol": args.df_tol,
        "df_max_rank": args.df_max_rank,
    }
    cache_meta = cache_dir / "settings.json"
    if cache_meta.exists():
        _prev = json.loads(cache_meta.read_text())
        if _prev != settings:
            raise SystemExit(
                f"\nref_cache {cache_dir} was built with:\n  {_prev}\n"
                f"but you asked for:\n  {settings}\n"
                "Use a different --ref-cache-dir (the cache key is only the molecule index)."
            )
    else:
        cache_meta.write_text(json.dumps(settings, indent=2))
    print(f"reference settings: {settings}", flush=True)

    # Reference molecules (B3LYP) are built on CPU (native integral backend).
    cpu = jax.devices("cpu")[0]
    molecules, ref_energies = [], []
    n_skipped = 0
    with jax.default_device(cpu):
        for idx, atom in enumerate(specs):
            cache_file = cache_dir / f"mol_{idx:04d}.pkl"
            m = None
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    m, e = pickle.load(f)
                if not bool(getattr(m, "scf_converged", True)):
                    print(f"mol{idx}: cached reference NOT converged -> rebuilding", flush=True)
                    m = None
            if m is None:
                m = build_reference(
                    atom, args.basis, args.ref_xc,
                    grids_level=args.grids_level,
                    jk_backend=args.jk_backend,
                    df_tol=args.df_tol,
                    df_max_rank=args.df_max_rank,
                )
                e = float(m.mf_energy)
                if bool(getattr(m, "scf_converged", False)):
                    with open(cache_file, "wb") as f:
                        pickle.dump((m, e), f)
            if not bool(getattr(m, "scf_converged", False)):
                n_skipped += 1
                print(
                    f"mol{idx}: reference SCF did not converge "
                    f"({getattr(m, 'scf_cycles', '?')} cycles) -> SKIPPED (bad target)",
                    flush=True,
                )
                continue
            molecules.append(m)
            ref_energies.append(e)
            print(f"mol{idx}: E_ref = {e:.8f} Ha", flush=True)
    print(f"usable molecules: {len(molecules)}  (skipped {n_skipped} with unconverged reference)",
          flush=True)

    model = load_skala_functional(args.weights_dir, key=jax.random.PRNGKey(args.seed))
    functional = SkalaTrainFunctional(model)
    _, static = eqx.partition(model, eqx.is_array)

    config = training.MolecularTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        e0_total_mse_weight=0.0,
        e0_total_mae_weight=1.0,
        scf_max_cycle=args.max_cycle,
        scf_damping=args.damping,
        scf_level_shift=args.level_shift,
        scf_eigenvalue_jitter=args.eigenvalue_jitter,
        scf_iterate_selection=args.iterate_selection,
        scf_conv_tol_energy=args.conv_tol_energy,
        scf_conv_tol_density=args.conv_tol_density,
        scf_conv_tol_grad=args.conv_tol_grad,
        scf_implicit_diff_tolerance=args.implicit_diff_tolerance,
        scf_implicit_diff_max_iter=args.implicit_diff_max_iter,
        scf_implicit_diff_restart=args.implicit_diff_restart,
        scf_convergence_metric="energy_and_residual",
        scf_require_converged=False,
    )
    dataset = tuple(
        training.MolecularTrainingDatum(
            molecule=m,
            target_e0_total_h=jnp.asarray(e, dtype=jnp.float64),
        )
        for m, e in zip(molecules, ref_energies, strict=True)
    )

    rng = np.random.default_rng(args.seed)
    n = len(dataset)
    n_val = int(round(n * args.val_fraction))
    n_train = n - n_val
    perm = rng.permutation(n)
    val_dataset = tuple(dataset[int(i)] for i in perm[:n_val])
    train_dataset = tuple(dataset[int(i)] for i in perm[n_val:])
    # Grid size per training sample, used to pair a large molecule with a small
    # one so that no batch ever holds two large molecules (that is what OOMs).
    train_grid = np.array(
        [int(dataset[int(perm[n_val + j])].molecule.ao.shape[0]) for j in range(n_train)]
    )
    print(f"train={n_train}  val={n_val}", flush=True)

    batch_size = max(1, args.batch_size)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    print(f"epochs={args.epochs}  steps_per_epoch={steps_per_epoch}", flush=True)
    print(
        f"train grid points: min={train_grid.min()} median={int(np.median(train_grid))} "
        f"max={train_grid.max()}",
        flush=True,
    )

    total_steps = max(1, args.epochs * steps_per_epoch)
    if args.lr_schedule == "cosine":
        schedule = optax.cosine_decay_schedule(args.lr, total_steps)
    else:
        schedule = optax.constant_schedule(args.lr)
    print(
        f"lr={args.lr:g}  schedule={args.lr_schedule}  "
        f"batch_size={batch_size}  total_steps={total_steps}",
        flush=True,
    )

    state = training.trainer.create_train_state_from_molecule(
        functional, jax.random.PRNGKey(args.seed), molecules[0], optax.adam(schedule)
    )
    train_step = training.trainer.make_molecular_train_step(functional, training_config=config)
    evaluate = training.trainer.make_molecular_eval(functional, training_config=config)

    log_path = out / "log.csv"
    best_val = float("inf")
    best_epoch = -1
    log_exists = args.log_append and log_path.exists()
    with open(log_path, "a" if log_exists else "w", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow([
                "epoch", "last_batch_loss", "train_energy_mae", "train_energy_mse",
                "train_energy_rmse", "train_scf_converged", "train_scf_cycles",
                "val_energy_mae", "val_energy_mse", "val_energy_rmse",
                "val_scf_converged", "val_scf_cycles", "gpu_mem_peak_mib",
                "best_val_mae", "is_best",
            ])

        for epoch in range(1, args.epochs + 1):
            order = rng.permutation(n_train)
            # Largest-with-smallest pairing: every batch has at most one large
            # molecule, capping the peak GPU memory (2 large ones exceed 32 GB).
            by_size = sorted(order.tolist(), key=lambda j: -train_grid[j])
            paired: list[int] = []
            lo, hi = 0, len(by_size) - 1
            while lo < hi:
                paired.append(by_size[lo])
                paired.append(by_size[hi])
                lo += 1
                hi -= 1
            if lo == hi:
                paired.append(by_size[lo])
            batches = [
                paired[k * batch_size:(k + 1) * batch_size]
                for k in range(steps_per_epoch)
            ]
            rng.shuffle(batches)
            last_batch_loss = float("nan")
            mem_peak = 0
            for i, idx in enumerate(batches):
                idx = np.asarray(idx)
                batch = tuple(train_dataset[int(j)] for j in idx)
                state, metrics = train_step(state, batch)
                last_batch_loss = float(metrics["total_loss"])
                mem = gpu_mem_mib()
                mem_peak = max(mem_peak, mem)
                print(
                    f"epoch {epoch:03d} batch {i:03d}  batch_loss={last_batch_loss:.6e}  "
                    f"conv={float(metrics['scf_converged_fraction'][0]):.2f}  "
                    f"nonfinite_grad={float(metrics['nonfinite_grad_fraction'][0]):.2f}  "
                    f"update={bool(metrics['update_accepted'])}  gpu_mem={mem}MiB",
                    flush=True,
                )
                if args.clear_caches_every and (i + 1) % args.clear_caches_every == 0:
                    before = gpu_mem_mib()
                    jax.clear_caches()
                    print(
                        f"epoch {epoch:03d} batch {i:03d}  cleared JIT cache "
                        f"(gpu_mem {before} -> {gpu_mem_mib()} MiB)",
                        flush=True,
                    )

            # ---- epoch-level metrics over ALL training molecules ----
            tr_loss, tr_m = evaluate(state.params, train_dataset)
            train_mae = _mean(tr_m, "e0_total_mae")
            train_mse = _mean(tr_m, "e0_total_mse")
            train_rmse = float(np.sqrt(train_mse))
            train_conv = _mean(tr_m, "scf_converged")
            train_cycles = _mean(tr_m, "scf_cycles")
            if n_val:
                val_loss, val_m = evaluate(state.params, val_dataset)
                val_mae = _mean(val_m, "e0_total_mae")
                val_mse = _mean(val_m, "e0_total_mse")
                val_rmse = float(np.sqrt(val_mse))
                val_conv = _mean(val_m, "scf_converged")
                val_cycles = _mean(val_m, "scf_cycles")
            else:
                val_mae = val_mse = val_rmse = val_conv = val_cycles = float("nan")

            is_best = 0
            if n_val and np.isfinite(val_mae) and val_mae < best_val:
                best_val = float(val_mae)
                best_epoch = epoch
                is_best = 1
                save_skala_checkpoint(state, static, out / "checkpoints" / "best")
            print(
                f"epoch {epoch:03d}  : train_energy_mae={train_mae:.6e}  "
                f"train_energy_mse={train_mse:.6e}  train_energy_rmse={train_rmse:.6e}  "
                f"train_scf_converged={train_conv:.2f} cyc={train_cycles:.1f}  |  "
                f"val_energy_mae={val_mae:.6e}  val_energy_mse={val_mse:.6e}  "
                f"val_scf_converged={val_conv:.2f} cyc={val_cycles:.1f}  |  "
                f"gpu_peak={mem_peak}MiB  |  "
                f"best_val_mae={best_val:.6e} @epoch {best_epoch}{'  <- best' if is_best else ''}",
                flush=True,
            )
            writer.writerow([
                epoch, last_batch_loss, train_mae, train_mse, train_rmse,
                train_conv, train_cycles,
                val_mae, val_mse, val_rmse, val_conv, val_cycles,
                mem_peak, best_val, is_best,
            ])
            f.flush()

            if args.checkpoint_every and epoch % args.checkpoint_every == 0:
                save_skala_checkpoint(state, static, out / "checkpoints" / f"epoch_{epoch:03d}")
            save_skala_checkpoint(state, static, out / "checkpoints" / "latest")

    save_skala_checkpoint(state, static, out / "checkpoints" / "final")
    print(f"log -> {log_path}", flush=True)


if __name__ == "__main__":
    main()
