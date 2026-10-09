"""Diagnose whether SCF non-convergence is the RKS solver or the Skala functional.

Runs the same SCF settings (damping=0.6, level_shift=0.2, max_cycle=100,
conv_tol=2e-6) on the first few molecules with:

  * a standard hybrid functional (PBE0 / B3LYP) via the built-in RKS path;
  * the external Skala functional via the implicit DifferentiableSCF path.

If PBE0/B3LYP converge but Skala does not, the problem is the Skala network
(its vxc potential), not the RKS solver.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from gradscf import training
from gradscf.scf.builders import restricted_molecule_from_spec_with_jax_rks
from gradscf.scf.rks import RKSConfig
from gradscf.workflows.core import run_molecule_from_spec
from gradscf.workflows.types import MoleculeSpecConfig, SimulationConfig

from adapter import make_skala_functional, load_skala_functional


jax.config.update("jax_enable_x64", True)


def parse_xyz(path: Path, nmol: int) -> list[str]:
    lines = path.read_text().splitlines()
    specs: list[str] = []
    i = 0
    while i < len(lines) and len(specs) < nmol:
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
    return specs


def run_builtin_xc(atom: str, xc: str) -> tuple[bool, int, float]:
    cfg = RKSConfig(
        xc_spec=xc,
        max_cycle=100,
        conv_tol=2e-6,
        conv_tol_density=2e-6,
        damping=0.6,
        level_shift=0.2,
    )
    with jax.default_device(jax.devices("cpu")[0]):
        m = restricted_molecule_from_spec_with_jax_rks(
            atom=atom,
            basis="sto-3g",
            xc_spec=xc,
            unit="Angstrom",
            rks_config=cfg,
        )
    return bool(m.scf_converged), int(m.scf_cycles), float(m.mf_energy)


def build_b3lyp_reference(atom: str):
    spec = MoleculeSpecConfig(atom=atom, basis="sto-3g", xc="b3lyp", unit="Angstrom")
    sim = SimulationConfig(scf_backend="jax_rks", jax_rks_xc_spec="b3lyp")
    with jax.default_device(jax.devices("cpu")[0]):
        return run_molecule_from_spec(spec, simulation=sim).molecule


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz-file", type=Path, required=True)
    ap.add_argument("--nmol", type=int, default=3)
    ap.add_argument("--weights-dir", type=Path, default=None)
    args = ap.parse_args()

    specs = parse_xyz(args.xyz_file, args.nmol)
    print(f"testing {len(specs)} molecules", flush=True)

    print("\n=== built-in RKS (damping=0.6, level_shift=0.2, max_cycle=100, tol=2e-6) ===", flush=True)
    for xc in ("hf",):
        for idx, atom in enumerate(specs):
            conv, cyc, e = run_builtin_xc(atom, xc)
            print(f"{xc:6s} mol{idx}: converged={conv}  cycles={cyc}  E={e:.8f} Ha", flush=True)

    print("\n=== Skala external functional (implicit, same SCF settings) ===", flush=True)
    model = load_skala_functional(args.weights_dir, key=jax.random.PRNGKey(0))
    functional = make_skala_functional(model)
    config = training.MolecularTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        e0_total_mse_weight=1.0,
        e0_total_mae_weight=1.0,
        scf_max_cycle=100,
        scf_damping=0.6,
        scf_level_shift=0.2,
        scf_eigenvalue_jitter=1e-3,
        scf_iterate_selection="best_rms",
        scf_conv_tol_energy=2e-6,
        scf_conv_tol_density=2e-6,
        scf_convergence_metric="energy_and_residual",
        scf_require_converged=False,
    )
    for idx, atom in enumerate(specs):
        m = build_b3lyp_reference(atom)
        e_ref = float(m.mf_energy)
        datum = training.MolecularTrainingDatum(
            molecule=m,
            target_e0_total_h=jnp.asarray(e_ref, dtype=jnp.float64),
        )
        state = training.trainer.create_train_state_from_molecule(
            functional, jax.random.PRNGKey(0), m, optax.adam(1e-4)
        )
        train_step = training.trainer.make_molecular_train_step(functional, training_config=config)
        _, metrics = train_step(state, (datum,))
        conv = float(metrics["scf_converged"][0])
        cyc = float(metrics["scf_cycles"][0])
        rms = float(metrics["scf_final_rms_density"][0])
        e_pred = float(metrics["predicted_e0_total_h"][0])
        print(
            f"skala mol{idx}: converged={conv:.2f}  cycles={cyc:.1f}  rms={rms:.3e}  "
            f"E_pred={e_pred:.8f}  E_ref={e_ref:.8f} Ha",
            flush=True,
        )


if __name__ == "__main__":
    main()
