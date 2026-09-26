"""Self-consistent training of the Skala XC functional (H2 demo).

Loss = (E_total - E_HF)^2, where E_total comes from a self-consistent SCF using
the Skala functional and E_HF is the HF reference.  The gradient flows through
the SCF fixed point (implicit differentiation) back into the Skala parameters.
"""

from __future__ import annotations

import argparse
import shutil
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


SAVE_DIR = (
    Path(__file__).resolve().parent.parent
    / "tests" / "skalax" / "trained_weights" / "weights"
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom", default="H 0 0 0; H 0 0 0.74")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    molecule = restricted_molecule_from_spec_with_jax_rks(
        atom=args.atom,
        basis=args.basis,
        xc_spec="hf",
        unit="Angstrom",
        integral_backend="cpu",
    )
    e_ref = float(molecule.mf_energy)

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=40,
            damping=0.5,
            conv_tol_energy=1e-8,
            conv_tol_density=1e-8,
            differentiation=SCFDifferentiationConfig(mode="implicit"),
        )
    )

    def total_energy(model, molecule):
        adapter = SkalaFunctionalAdapter(model)
        result, _ = solver.run(molecule, adapter, xc_params={})
        density = result.rdm1.sum(axis=0)
        h1e = jnp.asarray(molecule.h1e)
        coulomb, _ = build_jk_from_eri_pair_matrix(
            jnp.asarray(molecule.eri_pair_matrix), density
        )
        xc = adapter.energy_from_molecule({}, result)
        return (
            jnp.sum(density * h1e)
            + 0.5 * jnp.sum(density * coulomb)
            + xc
            + jnp.asarray(molecule.nuclear_repulsion)
        )

    model = load_skala_functional()
    params, static = eqx.partition(model, eqx.is_inexact_array)
    w3j_ref = model.non_local_model.tp_down.w3j["w3j_0_0_0"]

    def loss_fn(params):
        m = eqx.combine(params, static)
        return (total_energy(m, molecule) - e_ref) ** 2

    optim = optax.adam(args.lr)
    opt_state = optim.init(params)

    def make_step(params, opt_state):
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(params)
        grads = _zero_w3j_grads(grads)
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads))
        )
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss_value, grad_norm, params, opt_state

    e0 = float(total_energy(model, molecule))
    print(f"E_ref (HF) = {e_ref:.8f} Ha", flush=True)
    print(f"step init : E_total = {e0:.8f} Ha", flush=True)

    for i in range(args.steps):
        loss_value, grad_norm, params, opt_state = make_step(params, opt_state)
        m = eqx.combine(params, static)
        e_total = float(total_energy(m, molecule))
        print(
            f"step {i:02d}  : loss={float(loss_value):.6e}  "
            f"E_total={e_total:.8f}  grad_norm={float(grad_norm):.6e}",
            flush=True,
        )

    m_final = eqx.combine(params, static)
    w3j_changed = float(
        jnp.max(jnp.abs(m_final.non_local_model.tp_down.w3j["w3j_0_0_0"] - w3j_ref))
    )
    print(f"w3j max |change| = {w3j_changed:.3e}  (frozen: {w3j_changed == 0.0})", flush=True)

    backup_dir = SAVE_DIR.parent / "weights_prev"
    if SAVE_DIR.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(SAVE_DIR, backup_dir)
    save_weights_to_npz(m_final, SAVE_DIR)
    print(f"saved trained weights -> {SAVE_DIR}", flush=True)
    print(f"previous weights backed up -> {backup_dir}", flush=True)


if __name__ == "__main__":
    main()
