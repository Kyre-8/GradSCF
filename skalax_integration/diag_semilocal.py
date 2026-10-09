"""Compare Skala SCF convergence with vs without the non-local branch gradient.

Hypothesis: the unstable Vxc comes from backpropagating through the non-local
message passing.  This builds a "semilocal" variant that freezes the non-local
feature ``h_nl`` with ``stop_gradient``, and compares SCF convergence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from gradscf import training
from gradscf.dft.functional import Functional
from gradscf.workflows.core import run_molecule_from_spec
from gradscf.workflows.types import MoleculeSpecConfig, SimulationConfig
from skalax.functional.base import enhancement_density_inner_product
from skalax.functional.functions import prepare_features

from adapter import make_skala_functional, make_skala_input, load_skala_functional


jax.config.update("jax_enable_x64", True)


def make_semilocal_functional(model) -> Functional:
    params, static = eqx.partition(model, eqx.is_array)

    def input_fn(di):
        return make_skala_input(di)

    def energy_fn(params, mol):
        p = params["params"] if isinstance(params, dict) and "params" in params else params
        m = eqx.combine(p, static)
        grid_coords = mol["grid_coords"]
        grid_weights = mol["grid_weights"]
        coarse_coords = mol["coarse_0_atomic_coords"]
        features_ab, features_ba = prepare_features(mol)
        spin_feats = jnp.concatenate([features_ab, features_ba], axis=0)
        spin_feats = spin_feats.astype(m.dtype)
        spin_feats = m._input_model(spin_feats)
        ab, ba = jnp.split(spin_feats, 2, axis=0)
        features = (ab + ba) / 2
        if m.non_local and m.non_local_model is not None:
            h_nl = m.non_local_model(features, grid_coords, coarse_coords, grid_weights)
            density_sum = mol["density"].sum(0).reshape(-1, 1)
            h_nl = h_nl * jnp.exp(-density_sum).astype(m.dtype)
            h_nl = jax.lax.stop_gradient(h_nl)
            features = jnp.concatenate([features, h_nl], axis=-1)
        enhancement_factor = m._output_model(features)
        exc_density = enhancement_density_inner_product(enhancement_factor, mol["density"])
        return (exc_density * mol["grid_weights"]).sum()

    def init_fn(key, mol):
        del key, mol
        return {"params": params}

    return Functional(input_fn=input_fn, energy_fn=energy_fn, init_fn=init_fn, name="skala_semilocal")


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


def build_b3lyp_reference(atom: str):
    spec = MoleculeSpecConfig(atom=atom, basis="sto-3g", xc="b3lyp", unit="Angstrom")
    sim = SimulationConfig(scf_backend="jax_rks", jax_rks_xc_spec="b3lyp")
    with jax.default_device(jax.devices("cpu")[0]):
        return run_molecule_from_spec(spec, simulation=sim).molecule


def config() -> training.MolecularTrainingConfig:
    return training.MolecularTrainingConfig(
        mode="self_consistent",
        scf_gradient_mode="impl",
        e0_total_mse_weight=0.0,
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


def run_one(functional, molecule, e_ref, label, seed=0):
    datum = training.MolecularTrainingDatum(
        molecule=molecule,
        target_e0_total_h=jnp.asarray(e_ref, dtype=jnp.float64),
    )
    state = training.trainer.create_train_state_from_molecule(
        functional, jax.random.PRNGKey(seed), molecule, optax.adam(1e-4)
    )
    train_step = training.trainer.make_molecular_train_step(functional, training_config=config())
    _, metrics = train_step(state, (datum,))
    conv = float(metrics["scf_converged"][0])
    cyc = float(metrics["scf_cycles"][0])
    rms = float(metrics["scf_final_rms_density"][0])
    e_pred = float(metrics["predicted_e0_total_h"][0])
    nonfinite = float(metrics["nonfinite_grad_fraction"][0])
    print(
        f"{label:12s}: conv={conv:.2f}  cyc={cyc:.1f}  rms={rms:.3e}  "
        f"E_pred={e_pred:.8f}  E_ref={e_ref:.8f}  nonfinite={nonfinite:.2f}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz-file", type=Path, required=True)
    ap.add_argument("--nmol", type=int, default=3)
    ap.add_argument("--weights-dir", type=Path, default=None)
    args = ap.parse_args()

    specs = parse_xyz(args.xyz_file, args.nmol)
    model = load_skala_functional(args.weights_dir, key=jax.random.PRNGKey(0))
    full = make_skala_functional(model)
    semi = make_semilocal_functional(model)

    for idx, atom in enumerate(specs):
        m = build_b3lyp_reference(atom)
        e_ref = float(m.mf_energy)
        print(f"--- mol{idx} ---", flush=True)
        run_one(full, m, e_ref, "full", seed=idx)
        run_one(semi, m, e_ref, "semilocal", seed=idx)


if __name__ == "__main__":
    main()
