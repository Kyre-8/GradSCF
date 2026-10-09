"""Self-consistent training of the Skala XC functional.

Data: a multi-molecule XYZ file (``--xyz-file``), one sample per molecule.
Only the coordinates are read; the comment line's numeric column in the QM8
files is NOT a total energy and is ignored.  Reference energies/densities
come from the GradSCF molecule builder (``--ref-xc``, default ``b3lyp``;
traditional XC functionals need the optional ``jax_xc`` package).
Molecules are built lazily and cached on first use.

Loss per sample (batch means over converged samples):

    L_E = (E_scf - E_ref)^2                                 [Ha^2]
    L_P = ||P_scf - P_ref||_F^2 / ||P_ref||_F^2             [dimensionless]
    L   = mean(L_E) + density_weight * mean(L_P)

Total energy (restricted KS):

    E_scf = Tr(P h1e) + 1/2 Tr(P J) + E_xc + E_nuc,   P = sum_spin rdm1

E_xc is the pure Skala XC energy: the adapter reports exact-exchange
fraction alpha = 0 (Skala is a pure density functional), so there is no
double-counting correction to subtract.  P_scf / P_ref are the full
spin-resolved density matrices -- never real-space traces.

Gradients flow through the SCF fixed point (implicit differentiation) back
into the Skala parameters.  Each step trains on a random minibatch
(``--batch-size``).  Samples are split train/val/test (default 70/15/15);
validation loss drives best-checkpoint selection and early stopping.

Example:
    python skalax_integration/train_self_consistent.py \
        --xyz-file qm8_text/XYZ_B3LYP_631G2dfp_mol576_1075.xyz \
        --basis "6-31g(2df,p)" --ref-xc b3lyp --batch-size 32 --steps 5000
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from jax.tree_util import GetAttrKey

jax.config.update("jax_enable_x64", True)

from gradscf.integrals import build_jk_from_eri_pair_matrix
from gradscf.scf import (
    DifferentiableSCF,
    DifferentiableSCFConfig,
    SCFDifferentiationConfig,
    restricted_molecule_from_spec_with_jax_rks,
)

from skalax_integration.adapter import SkalaFunctionalAdapter, load_skala_functional


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class MoleculeSpecEntry:
    """One training sample before the (expensive) molecule build."""

    name: str
    atom: str


@dataclass(frozen=True)
class TrainingConfig:
    """One molecular configuration with its reference data."""

    name: str
    molecule: object
    e_ref: float
    rho_ref: jnp.ndarray  # full spin-resolved reference 1-RDM, shape (2, nao, nao)
    rho_ref_norm_sq: float  # ||P_ref||_F^2, normalizer of the relative density loss
    n_elec: float


class MoleculeFactory:
    """Lazy builder: molecules are constructed (and cached) on first use."""

    def __init__(self, specs: list[MoleculeSpecEntry], args):
        self.specs = specs
        self.args = args
        self._cache: dict[int, TrainingConfig] = {}

    def __len__(self) -> int:
        return len(self.specs)

    def get(self, index: int) -> TrainingConfig:
        if index in self._cache:
            return self._cache[index]
        spec = self.specs[index]
        # The native integral backend is CPU-only; build the reference molecule
        # under a CPU device context so this also runs cleanly on GPU nodes.
        with jax.default_device(jax.devices("cpu")[0]):
            molecule = restricted_molecule_from_spec_with_jax_rks(
                atom=spec.atom,
                basis=self.args.basis,
                xc_spec=self.args.ref_xc,
                unit="Angstrom",
                charge=self.args.charge,
                spin=self.args.spin,
                integral_backend="cpu",
            )
        if molecule.mf_energy is None:
            raise RuntimeError(f"{spec.name}: reference energy unavailable")
        e_ref = float(molecule.mf_energy)
        rdm1_ref = jnp.asarray(molecule.rdm1)
        cfg = TrainingConfig(
            name=spec.name,
            molecule=molecule,
            e_ref=e_ref,
            rho_ref=rdm1_ref,
            rho_ref_norm_sq=float(jnp.sum(rdm1_ref**2)),
            n_elec=float(np.asarray(molecule.mo_occ).sum()),
        )
        print(f"built {spec.name}: E_ref = {e_ref:.8f} Ha", flush=True)
        self._cache[index] = cfg
        return cfg


def parse_xyz_specs(path: Path, max_molecules: int | None) -> list[MoleculeSpecEntry]:
    """Parse a multi-molecule XYZ file into specs.

    Comment-line format: ``Molecule <id> <value> <smiles>``; only the id is
    used.  The numeric column in the QM8 xyz files is NOT the total energy, so
    it is ignored -- reference energies always come from the molecule builder
    (``--ref-xc``).  Atom lines: ``Symbol x y z``.
    """
    specs: list[MoleculeSpecEntry] = []
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        natom = int(lines[i].strip())
        comment = lines[i + 1].strip()
        tokens = comment.split()
        name = f"mol{tokens[1]}" if len(tokens) >= 2 and tokens[0] == "Molecule" else f"idx{len(specs)}"
        atoms = []
        for line in lines[i + 2 : i + 2 + natom]:
            fields = line.split()
            atoms.append(f"{fields[0]} {fields[1]} {fields[2]} {fields[3]}")
        specs.append(MoleculeSpecEntry(name=name, atom="; ".join(atoms)))
        i += natom + 2
        if max_molecules is not None and len(specs) >= max_molecules:
            break
    return specs


def _zero_w3j_grads(grads):
    """Zero out gradients of the Wigner-3j buffers (physical constants)."""

    def zero(path, leaf):
        if any(isinstance(k, GetAttrKey) and k.name == "w3j" for k in path):
            return jnp.zeros_like(leaf)
        return leaf

    return jax.tree_util.tree_map_with_path(zero, grads)


def save_weights_to_npz(model, weights_dir):
    """Save weights/buffers in the exact NPZ format ``load_weights_from_npz`` reads."""
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    weights = {
        "input_model.0.weight": model.input_linear1.weight,
        "input_model.0.bias": model.input_linear1.bias,
        "input_model.2.weight": model.input_linear2.weight,
        "input_model.2.bias": model.input_linear2.bias,
        "output_model.0.weight": model.output_linear1.weight,
        "output_model.0.bias": model.output_linear1.bias,
        "output_model.2.weight": model.output_linear2.weight,
        "output_model.2.bias": model.output_linear2.bias,
        "output_model.4.weight": model.output_linear3.weight,
        "output_model.4.bias": model.output_linear3.bias,
        "output_model.6.weight": model.output_linear4.weight,
        "output_model.6.bias": model.output_linear4.bias,
    }
    buffers = {}
    if model.non_local and model.non_local_model is not None:
        nl = model.non_local_model
        weights["non_local_model.pre_down_layer.0.weight"] = nl.pre_down_linear.weight
        weights["non_local_model.pre_down_layer.0.bias"] = nl.pre_down_linear.bias
        weights["non_local_model.post_up_layer.0.weight"] = nl.post_up_linear.weight
        weights["non_local_model.post_up_layer.0.bias"] = nl.post_up_linear.bias
        for tp_name in ("tp_down", "tp_up"):
            tp = getattr(nl, tp_name)
            for key, value in tp.weights.items():
                weights[f"non_local_model.{tp_name}.{key}"] = value
            for key, value in tp.w3j.items():
                buffers[f"non_local_model.{tp_name}.{key}"] = value

    np.savez(
        weights_dir / "skala_weights.npz",
        **{k: np.asarray(v) for k, v in weights.items()},
    )
    np.savez(
        weights_dir / "skala_buffers.npz",
        **{k: np.asarray(v) for k, v in buffers.items()},
    )


def make_solver(args) -> DifferentiableSCF:
    return DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=args.max_cycle,
            damping=0.5,
            conv_tol_energy=1e-8,
            conv_tol_density=1e-8,
            conv_tol_grad=args.conv_tol_grad,
            differentiation=SCFDifferentiationConfig(mode="implicit"),
        )
    )


def scf_energy_and_density_only(model, molecule, solver):
    """Run self-consistent SCF with the Skala functional (lightweight).

    Returns ``(e_tot, rdm1, converged)`` with ``rdm1`` the full spin-resolved
    density matrix, shape ``(2, nao, nao)``.  MO energies/occupations are not
    extracted; used by the training loop (``loss_fn``) and ``evaluate``.

    Total energy (restricted KS):

        E = Tr(P h1e) + 1/2 Tr(P J) + E_xc + E_nuc,   P = sum_spin rdm1

    The Skala adapter reports exact-exchange fraction alpha = 0, so E_xc is
    the entire XC contribution and there is no double-counting correction.
    """
    adapter = SkalaFunctionalAdapter(model)
    result, info = solver.run(molecule, adapter, xc_params={})
    rdm1 = jnp.asarray(result.rdm1)
    # Spin-summed total density matrix for the restricted energy expression.
    density = rdm1.sum(axis=0) if rdm1.ndim == 3 else rdm1
    h1e = jnp.asarray(molecule.h1e)
    coulomb, _ = build_jk_from_eri_pair_matrix(
        jnp.asarray(molecule.eri_pair_matrix), density
    )
    xc = adapter.energy_from_density({}, molecule, density)
    e_tot = (
        jnp.sum(density * h1e)
        + 0.5 * jnp.sum(density * coulomb)
        + xc
        + jnp.asarray(molecule.nuclear_repulsion)
    )
    converged = jax.lax.stop_gradient(jnp.asarray(info.converged))
    return e_tot, rdm1, converged


def scf_energy_and_density(model, molecule, solver):
    """Full version of :func:`scf_energy_and_density_only`.

    Additionally returns ``mo_energy`` / ``mo_occ`` for frontier-orbital
    (HOMO/LUMO) analysis.  Only called by ``evaluate_detailed`` for the final
    test-set evaluation.
    """
    adapter = SkalaFunctionalAdapter(model)
    result, info = solver.run(molecule, adapter, xc_params={})
    rdm1 = jnp.asarray(result.rdm1)
    # Spin-summed total density matrix for the restricted energy expression.
    density = rdm1.sum(axis=0) if rdm1.ndim == 3 else rdm1
    h1e = jnp.asarray(molecule.h1e)
    coulomb, _ = build_jk_from_eri_pair_matrix(
        jnp.asarray(molecule.eri_pair_matrix), density
    )
    xc = adapter.energy_from_density({}, molecule, density)
    e_tot = (
        jnp.sum(density * h1e)
        + 0.5 * jnp.sum(density * coulomb)
        + xc
        + jnp.asarray(molecule.nuclear_repulsion)
    )
    converged = jax.lax.stop_gradient(jnp.asarray(info.converged))
    return e_tot, rdm1, result.mo_energy, result.mo_occ, converged


def density_rel_error(rdm1, cfg: TrainingConfig):
    """Relative density-matrix error ``||P - P_ref||_F^2 / ||P_ref||_F^2``."""
    return jnp.sum((rdm1 - cfg.rho_ref) ** 2) / cfg.rho_ref_norm_sq


def evaluate(model, solver, configs, density_weight):
    """No-grad validation metrics: per-sample errors plus global means.

    Only converged SCF samples are included; non-converged ones are dropped.
    Returns ``(metrics, per_sample)``.  ``metrics`` holds the global means
    (energy MAE/RMSE, mean relative density-matrix error) and ``val_loss`` --
    the validation analogue of the training loss, used for best-checkpoint
    selection and early stopping.  ``per_sample`` lists each converged
    molecule's individual errors.  If no sample converged, all metrics are
    NaN and ``per_sample`` is empty.
    """
    per_sample = []
    n_skipped = 0
    for cfg in configs:
        e, rdm1, converged = scf_energy_and_density_only(model, cfg.molecule, solver)
        if not bool(converged):
            n_skipped += 1
            continue
        e_err = e - cfg.e_ref
        per_sample.append(
            {
                "name": cfg.name,
                "energy": float(e),
                "energy_ref": cfg.e_ref,
                "energy_abs_err": float(jnp.abs(e_err)),
                "energy_sq_err": float(e_err**2),
                "density_rel_err": float(density_rel_error(rdm1, cfg)),
            }
        )
    if n_skipped:
        print(
            f"evaluate: skipped {n_skipped}/{len(configs)} non-converged samples",
            flush=True,
        )
    if not per_sample:
        metrics = {
            "energy_mae": float("nan"),
            "energy_rmse": float("nan"),
            "density_rel": float("nan"),
            "val_loss": float("nan"),
        }
        return metrics, per_sample
    energy_mse = float(np.mean([s["energy_sq_err"] for s in per_sample]))
    density_rel = float(np.mean([s["density_rel_err"] for s in per_sample]))
    metrics = {
        "energy_mae": float(np.mean([s["energy_abs_err"] for s in per_sample])),
        "energy_rmse": float(np.sqrt(energy_mse)),
        "density_rel": density_rel,
        "val_loss": energy_mse + density_weight * density_rel,
    }
    return metrics, per_sample


def homo_lumo_gap(mo_energy, mo_occ):
    """HOMO/LUMO energies and gap from ``(spin, nmo)`` MO arrays."""
    mo_energy = np.asarray(mo_energy).ravel()
    occ = np.asarray(mo_occ).ravel() > 1e-8
    homo = float(mo_energy[occ].max())
    lumo = float(mo_energy[~occ].min())
    return homo, lumo, lumo - homo


def evaluate_detailed(model, solver, configs):
    """Final per-sample physical metrics: energy, density matrix, SCF
    convergence, and frontier orbitals (HOMO/LUMO/gap, pred vs reference)."""
    rows = []
    for cfg in configs:
        e, rdm1, mo_energy, mo_occ, converged = scf_energy_and_density(
            model, cfg.molecule, solver
        )
        homo, lumo, gap = homo_lumo_gap(mo_energy, mo_occ)
        homo_ref, lumo_ref, gap_ref = homo_lumo_gap(
            cfg.molecule.mo_energy, cfg.molecule.mo_occ
        )
        rows.append(
            {
                "name": cfg.name,
                "converged": bool(converged),
                "energy": float(e),
                "energy_ref": cfg.e_ref,
                "energy_err": float(e - cfg.e_ref),
                "density_rel_err": float(density_rel_error(rdm1, cfg)),
                "homo": homo,
                "lumo": lumo,
                "gap": gap,
                "homo_ref": homo_ref,
                "lumo_ref": lumo_ref,
                "gap_ref": gap_ref,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xyz-file",
        type=Path,
        required=True,
        help="Multi-molecule XYZ file; one sample per molecule.",
    )
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument(
        "--ref-xc",
        default="b3lyp",
        help="XC used for the builder's reference run and reference density "
        "(default b3lyp, matching the QM8 B3LYP xyz files; traditional XC "
        "needs the optional jax_xc package).",
    )
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="If set, train for this many epochs (each epoch iterates the "
        "whole training split once). Loss/MAE/MSE are then reported per "
        "epoch over ALL training molecules. Overrides --steps.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr-schedule", choices=("constant", "cosine"), default="constant"
    )
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Minibatch size per step; 0 = full training set.")
    parser.add_argument("--density-weight", type=float, default=1e-2,
                        help="Weight lambda of the relative density-matrix loss.")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--val-every", type=int, default=50)
    # NOTE: --val-max is kept for local debugging only; formal experiments
    # always evaluate the full validation set (no slicing is applied).
    parser.add_argument("--val-max", type=int, default=8,
                        help="[local debug only, currently unused] Max number of "
                        "validation samples evaluated each time.")
    parser.add_argument("--early-stop-patience", type=int, default=20,
                        help="Stop after this many validation checks without "
                        "val_loss improvement; 0 disables early stopping.")
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--max-cycle", type=int, default=40)
    parser.add_argument(
        "--conv-tol-grad",
        type=float,
        default=1e-5,
        help="MO gradient residual tolerance of the SCF convergence check. "
        "The framework default (1e-7) is tighter than the numerical noise "
        "floor of the grid-based Skala potential on larger molecules "
        "(e.g. benzene stalls at ~1e-6 and is never marked converged, so "
        "its samples get masked out of the loss).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint weights dir (with skala_weights.npz) to resume from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        REPO_ROOT / "runs" / f"self_consistent_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = parse_xyz_specs(args.xyz_file, args.max_molecules)
    if len(specs) < 2:
        raise ValueError("need at least 2 samples")

    factory = MoleculeFactory(specs, args)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(specs))
    n_test = n_val = 0
    if len(specs) > 2:
        if args.test_frac > 0.0:
            n_test = max(1, int(round(len(specs) * args.test_frac)))
        if args.val_frac > 0.0:
            n_val = max(1, int(round(len(specs) * args.val_frac)))
    n_test = min(n_test, len(specs) - 2)
    n_val = min(n_val, len(specs) - n_test - 1)
    test_idx = list(order[:n_test])
    val_idx = list(order[n_test : n_test + n_val])
    train_idx = list(order[n_test + n_val :])
    print(
        f"samples: {len(specs)} total, {len(train_idx)} train, "
        f"{len(val_idx)} val, {len(test_idx)} test",
        flush=True,
    )

    solver = make_solver(args)

    if args.resume is not None:
        model = load_skala_functional(args.resume)
        print(f"resumed weights from {args.resume}", flush=True)
    else:
        model = load_skala_functional(key=jax.random.PRNGKey(args.seed))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    w3j_ref = model.non_local_model.tp_down.w3j["w3j_0_0_0"]

    batch_size = args.batch_size if args.batch_size > 0 else len(train_idx)
    batch_size = min(batch_size, len(train_idx))
    steps_per_epoch = max(1, math.ceil(len(train_idx) / batch_size))
    if args.epochs is not None:
        total_steps = int(args.epochs) * steps_per_epoch
    else:
        total_steps = int(args.steps)
    print(
        f"epochs={args.epochs}  train={len(train_idx)}  batch_size={batch_size}  "
        f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}",
        flush=True,
    )

    if args.lr_schedule == "cosine":
        schedule = optax.cosine_decay_schedule(args.lr, max(total_steps, 1))
    else:
        schedule = optax.constant_schedule(args.lr)
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = optim.init(params)

    def loss_fn(params, batch_cfgs):
        m = eqx.combine(params, static)
        e_errs, d_errs, masks = [], [], []
        for cfg in batch_cfgs:
            e, rdm1, converged = scf_energy_and_density_only(m, cfg.molecule, solver)
            e_errs.append(e - cfg.e_ref)
            d_errs.append(density_rel_error(rdm1, cfg))
            masks.append(converged.astype(e.dtype))
        e_errs = jnp.stack(e_errs)
        d_errs = jnp.stack(d_errs)
        mask = jnp.stack(masks)
        n_valid = jnp.maximum(mask.sum(), 1.0)
        energy_loss = jnp.sum(mask * e_errs**2) / n_valid  # L_E
        density_loss = jnp.sum(mask * d_errs) / n_valid  # L_P
        loss = energy_loss + args.density_weight * density_loss
        return loss, (e_errs, d_errs, mask)

    log_path = output_dir / "log.csv"
    log_fields = [
        "epoch",
        "last_batch_loss",
        "train_energy_mae",
        "train_energy_mse",
        "train_energy_rmse",
        "train_density_rel",
        "n_skipped",
        "grad_norm",
        "lr",
        "val_energy_mae",
        "val_energy_rmse",
        "val_density_rel",
        "val_loss",
    ]
    train_sample_path = output_dir / "train_per_sample.csv"
    train_sample_fields = [
        "step",
        "name",
        "converged",
        "energy",
        "energy_ref",
        "energy_err",
        "density_rel_err",
    ]
    val_sample_path = output_dir / "val_per_sample.csv"
    val_sample_fields = [
        "step",
        "name",
        "energy",
        "energy_ref",
        "energy_abs_err",
        "energy_sq_err",
        "density_rel_err",
    ]

    best_val_loss = math.inf
    best_params = None
    best_step = -1
    val_checks_no_improve = 0

    with open(log_path, "w", newline="") as log_file, open(
        train_sample_path, "w", newline=""
    ) as train_sample_file, open(
        val_sample_path, "w", newline=""
    ) as val_sample_file:
        writer = csv.writer(log_file)
        writer.writerow(log_fields)
        train_sample_writer = csv.writer(train_sample_file)
        train_sample_writer.writerow(train_sample_fields)
        val_sample_writer = csv.writer(val_sample_file)
        val_sample_writer.writerow(val_sample_fields)

        epoch_order = np.empty(0, dtype=np.int64)
        for i in range(total_steps):
            epoch = i // steps_per_epoch
            pos = i % steps_per_epoch
            if pos == 0:
                # New epoch: shuffle the whole training split.
                epoch_order = rng.permutation(train_idx)
            batch_idx = epoch_order[pos * batch_size : (pos + 1) * batch_size]
            batch_cfgs = [factory.get(int(j)) for j in batch_idx]

            (loss_value, (e_errs, d_errs, mask)), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True
            )(params, batch_cfgs)
            grads = _zero_w3j_grads(grads)
            grad_norm = jnp.sqrt(
                sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads))
            )
            lr = float(schedule(i))
            n_skipped = int(mask.size - mask.sum())

            if not (math.isfinite(float(loss_value)) and math.isfinite(float(grad_norm))):
                print(
                    f"epoch {epoch:03d} batch {pos:03d}  : non-finite loss/grad, update skipped",
                    flush=True,
                )
                continue

            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            print(
                f"epoch {epoch:03d} batch {pos:03d}  : batch_loss={float(loss_value):.6e}  "
                f"skipped={n_skipped}  grad_norm={float(grad_norm):.6e}  lr={lr:.2e}",
                flush=True,
            )

            val_energy_mae = val_energy_rmse = val_density_rel = val_loss = ""
            step_model = eqx.combine(params, static)
            stop_early = False
            epoch_end = (pos == steps_per_epoch - 1) or (i == total_steps - 1)
            if epoch_end:
                # -------- epoch-level metrics over the WHOLE training split --------
                train_cfgs = [factory.get(int(j)) for j in train_idx]
                tr_metrics, tr_per_sample = evaluate(
                    step_model, solver, train_cfgs, args.density_weight
                )
                epoch_train_mae = tr_metrics["energy_mae"]
                epoch_train_rmse = tr_metrics["energy_rmse"]
                epoch_train_mse = epoch_train_rmse * epoch_train_rmse
                epoch_train_drel = tr_metrics["density_rel"]
                epoch_train_skipped = len(train_cfgs) - len(tr_per_sample)
                print(
                    f"epoch {epoch:03d}  : train_energy_mae={epoch_train_mae:.6e}  "
                    f"train_energy_mse={epoch_train_mse:.6e}  "
                    f"train_energy_rmse={epoch_train_rmse:.6e}  "
                    f"train_density_rel={epoch_train_drel:.6e}  "
                    f"skipped={epoch_train_skipped}/{len(train_cfgs)}",
                    flush=True,
                )
                for row in tr_per_sample:
                    train_sample_writer.writerow(
                        [
                            epoch,
                            row["name"],
                            True,
                            row["energy"],
                            row["energy_ref"],
                            row["energy"] - row["energy_ref"],
                            row["density_rel_err"],
                        ]
                    )
                train_sample_file.flush()
            if val_idx and epoch_end:
                val_cfgs = [factory.get(int(j)) for j in val_idx]
                val_metrics, val_per_sample = evaluate(
                    step_model, solver, val_cfgs, args.density_weight
                )
                val_energy_mae = val_metrics["energy_mae"]
                val_energy_rmse = val_metrics["energy_rmse"]
                val_density_rel = val_metrics["density_rel"]
                val_loss = val_metrics["val_loss"]
                print(
                    f"epoch {epoch:03d}  : val_energy_mae={val_energy_mae:.6e}  "
                    f"val_energy_rmse={val_energy_rmse:.6e}  "
                    f"val_density_rel={val_density_rel:.6e}  "
                    f"val_loss={val_loss:.6e}",
                    flush=True,
                )
                for row in val_per_sample:
                    val_sample_writer.writerow(
                        [
                            epoch,
                            row["name"],
                            row["energy"],
                            row["energy_ref"],
                            row["energy_abs_err"],
                            row["energy_sq_err"],
                            row["density_rel_err"],
                        ]
                    )
                val_sample_file.flush()

                if not math.isfinite(val_loss):
                    print(
                        f"epoch {epoch:03d}  : no converged val sample, "
                        "checkpoint/early-stop check skipped",
                        flush=True,
                    )
                elif val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = params
                    best_step = i
                    val_checks_no_improve = 0
                    best_dir = output_dir / "weights_best"
                    save_weights_to_npz(step_model, best_dir)
                    print(
                        f"epoch {epoch:03d}  : new best val_loss={best_val_loss:.6e} "
                        f"-> {best_dir}",
                        flush=True,
                    )
                else:
                    val_checks_no_improve += 1
                    if (
                        args.early_stop_patience > 0
                        and val_checks_no_improve >= args.early_stop_patience
                    ):
                        print(
                            f"epoch {epoch:03d}  : early stopping "
                            f"(no val_loss improvement for {val_checks_no_improve} "
                            f"checks; best at global step {best_step})",
                            flush=True,
                        )
                        stop_early = True

            if epoch_end:
                writer.writerow(
                    [
                        epoch,
                        float(loss_value),
                        epoch_train_mae,
                        epoch_train_mse,
                        epoch_train_rmse,
                        epoch_train_drel,
                        epoch_train_skipped,
                        float(grad_norm),
                        lr,
                        val_energy_mae,
                        val_energy_rmse,
                        val_density_rel,
                        val_loss,
                    ]
                )
                log_file.flush()

            if stop_early:
                break

            if epoch_end:
                ckpt_dir = output_dir / f"weights_epoch{epoch + 1:03d}"
                save_weights_to_npz(step_model, ckpt_dir)
                print(f"checkpoint -> {ckpt_dir}", flush=True)

    m_final = eqx.combine(params, static)
    # w3j sanity print disabled; the buffers stay frozen via _zero_w3j_grads.
    # w3j_changed = float(
    #     jnp.max(jnp.abs(m_final.non_local_model.tp_down.w3j["w3j_0_0_0"] - w3j_ref))
    # )
    # print(f"w3j max |change| = {w3j_changed:.3e}  (frozen: {w3j_changed == 0.0})", flush=True)

    if best_params is not None:
        eval_model = eqx.combine(best_params, static)
        print(
            f"final evaluation uses best weights from step {best_step} "
            f"(val_loss={best_val_loss:.6e})",
            flush=True,
        )
    else:
        eval_model = m_final

    eval_idx = test_idx or val_idx or train_idx
    eval_split = "test" if test_idx else ("val" if val_idx else "train")
    eval_cfgs = [factory.get(int(j)) for j in eval_idx]
    rows = evaluate_detailed(eval_model, solver, eval_cfgs)

    eval_path = output_dir / f"{eval_split}_per_sample.csv"
    eval_fields = [
        "name",
        "converged",
        "energy",
        "energy_ref",
        "energy_err",
        "density_rel_err",
        "homo",
        "lumo",
        "gap",
        "homo_ref",
        "lumo_ref",
        "gap_ref",
    ]
    with open(eval_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=eval_fields)
        writer.writeheader()
        writer.writerows(rows)

    e_err = np.array([r["energy_err"] for r in rows])
    d_rel = np.array([r["density_rel_err"] for r in rows])
    conv = np.array([r["converged"] for r in rows], dtype=bool)
    gap = np.array([r["gap"] for r in rows])
    gap_ref = np.array([r["gap_ref"] for r in rows])
    print(f"final {eval_split} metrics ({len(rows)} samples):", flush=True)
    print(f"  energy MAE             = {np.mean(np.abs(e_err)):.6e} Ha", flush=True)
    print(f"  energy RMSE            = {np.sqrt(np.mean(e_err**2)):.6e} Ha", flush=True)
    print(f"  density matrix rel err = {np.mean(d_rel):.6e}", flush=True)
    print(f"  SCF convergence rate   = {conv.mean():.1%}", flush=True)
    print(f"  HOMO-LUMO gap MAE      = {np.mean(np.abs(gap - gap_ref)):.6e} Ha", flush=True)
    print(
        f"  mean gap (pred / ref)  = {gap.mean():.6e} / {gap_ref.mean():.6e} Ha",
        flush=True,
    )

    print(f"log -> {log_path}", flush=True)
    print(f"per-sample train energies -> {train_sample_path}", flush=True)
    print(f"per-sample val errors -> {val_sample_path}", flush=True)
    print(f"per-sample {eval_split} metrics -> {eval_path}", flush=True)


if __name__ == "__main__":
    main()
