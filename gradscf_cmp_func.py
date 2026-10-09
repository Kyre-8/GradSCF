"""Run the cluster's Functional-based adapter directly (bypassing the broken
package __init__) and compare against the inline original adapter on H2."""
import importlib.util
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from gradscf.scf import (
    DifferentiableSCF, DifferentiableSCFConfig, SCFDifferentiationConfig,
    restricted_molecule_from_spec_with_jax_rks,
)
from skalax import SkalaFunctional, get_default_weights_dir, load_config, load_weights_from_npz


def load_model():
    wd = get_default_weights_dir()
    cfg = load_config(wd)
    m = SkalaFunctional(
        lmax=cfg["lmax"], non_local=cfg["non_local"],
        non_local_hidden_nf=cfg["non_local_hidden_nf"],
        radius_cutoff=cfg["radius_cutoff"], key=jax.random.PRNGKey(0),
    )
    return load_weights_from_npz(m, wd)


def load_adapter_module(path):
    spec = importlib.util.spec_from_file_location("skala_adapter_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CFG_ORIG = dict(max_cycle=40, damping=0.5, conv_tol_energy=1e-8, conv_tol_density=1e-8)
CFG_CLUSTER = dict(max_cycle=100, damping=0.6, level_shift=0.2, conv_tol_energy=2e-6, conv_tol_density=2e-6)


def run(adapter, cfg, label):
    solver = DifferentiableSCF(DifferentiableSCFConfig(
        mode="self_consistent",
        differentiation=SCFDifferentiationConfig(mode="implicit"),
        **cfg,
    ))
    for r in [0.74, 1.0, 1.5]:
        mol = restricted_molecule_from_spec_with_jax_rks(
            atom=f"H 0 0 0; H 0 0 {r}", basis="sto-3g", xc_spec="hf",
            unit="Angstrom", charge=0, spin=0, integral_backend="cpu",
        )
        result, info = solver.run(mol, adapter, xc_params={})
        print(f"{label:12s} R={r:<4} conv={bool(info.converged)!s:5} cycles={int(info.cycles):3d} "
              f"rms={float(info.final_rms_density):.3e}", flush=True)


model = load_model()
mod = load_adapter_module("skalax_integration/adapter.py")
print("adapter module attrs:", [a for a in dir(mod) if not a.startswith("_")], flush=True)
func = mod.make_skala_functional(model)
print("type(func) =", type(func), flush=True)
print("has scf_xc_energy_and_alpha_for_density:",
      hasattr(func, "scf_xc_energy_and_alpha_for_density"), flush=True)
run(func, CFG_ORIG, "func/solver")
run(func, CFG_CLUSTER, "func/clust")
