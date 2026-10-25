"""Forward-only held-out evaluation of a trained SkalaX checkpoint.

No gradients, so the implicit-adjoint buffers are never allocated; this fits
molecules whose *training* step OOMs on the same GPU.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import training as gtraining
from gradscf.workflows.core import run_molecule_from_spec
from gradscf.workflows.types import MoleculeSpecConfig, SimulationConfig

from skalax_integration.adapter_v2 import load_skala_functional, make_skala_functional_v2

EV = 27.211386245988


def build_molecule(atom: str, basis: str = "6-31g*"):
    spec = MoleculeSpecConfig(atom=atom, basis=basis, xc="b3lyp",
                              unit="Angstrom", grids_level=2)
    sim = SimulationConfig(scf_backend="jax_rks", jax_rks_xc_spec="b3lyp",
                           jax_rks_max_cycle=300, jax_rks_damping=0.3)
    with jax.default_device(jax.devices("cpu")[0]):
        return run_molecule_from_spec(spec, simulation=sim).molecule


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--n-molecules", type=int, default=99)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--basis", default="6-31g*")
    args = p.parse_args()

    model = load_skala_functional(args.weights_dir)
    functional = make_skala_functional_v2(model, chunk_size=args.chunk_size)
    params = functional.init_fn(None, None)

    config = gtraining.MolecularTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        excitation_gap_mse_weight=1.0,
        excitation_gap_nstates=1,
        excited_state_solver="tda",
        tda_gradient_mode="eigenvalue_only",
        scf_max_cycle=100,
        scf_damping=0.6,
        scf_conv_tol_energy=2e-6,
        scf_conv_tol_density=2e-6,
        scf_conv_tol_grad=3e-6,
        scf_iterate_selection="first_converged",
        scf_convergence_metric="energy",
        scf_require_converged=True,
    )

    rows = list(csv.DictReader(open(args.dataset)))[: args.n_molecules]
    print(f'{"mol":>6} {"formula":<10} {"nat":>4} {"ref_eV":>9} {"pred_eV":>9} '
          f'{"err_meV":>10} {"time_s":>7}', flush=True)
    errs = []
    for row in rows:
        t0 = time.perf_counter()
        try:
            mol = build_molecule(row["xyz"], args.basis)
            datum = gtraining.MolecularTrainingDatum(
                molecule=mol,
                target_excitation_gaps_h=jnp.asarray([float(row["S1_ref_Ha"])]),
            )
            _loss, metrics = gtraining.molecular_loss(
                params, functional, datum, training_config=config
            )
            pred = float(metrics["excitation_gap_predicted_h"][0]) * EV
            ref = float(row["S1_ref_eV"])
            err = (pred - ref) * 1000.0
            errs.append(abs(err))
            print(f'{row["mol_id"]:>6} {row["formula"]:<10} {row["natom"]:>4} '
                  f'{ref:>9.4f} {pred:>9.4f} {err:>+10.1f} '
                  f'{time.perf_counter() - t0:>7.1f}', flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f'{row["mol_id"]:>6} {row["formula"]:<10} {row["natom"]:>4} '
                  f'FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:70]}',
                  flush=True)
    if errs:
        print(f"\nholdout MAE = {np.mean(errs):.1f} meV   (n={len(errs)})", flush=True)


if __name__ == "__main__":
    main()
