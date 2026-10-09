"""Compare the original SkalaFunctionalAdapter vs the new Functional-based
adapter, and two solver configs, on H2."""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from gradscf.scf import (
    DifferentiableSCF, DifferentiableSCFConfig, SCFDifferentiationConfig,
    restricted_molecule_from_spec_with_jax_rks,
)
from skalax import SkalaFunctional, get_default_weights_dir, load_config, load_weights_from_npz


def build_skala_features(molecule, density_total):
    ao = jnp.asarray(molecule.ao)
    ao_deriv1 = jnp.asarray(molecule.ao_deriv1)
    density = jnp.asarray(density_total)
    rho = jnp.einsum("pq,gp,gq->g", density, ao, ao)
    grad = 2.0 * jnp.einsum("xrp,pq,rq->xr", ao_deriv1[1:4], density, ao)
    tau = 0.5 * jnp.einsum("xrp,pq,xrq->r", ao_deriv1[1:4], density, ao_deriv1[1:4])
    rho_a = rho_b = 0.5 * rho
    grad_a = grad_b = 0.5 * grad
    tau_a = tau_b = 0.5 * tau
    return {
        "density": jnp.stack([rho_a, rho_b]),
        "grad": jnp.stack([grad_a, grad_b]),
        "kin": jnp.stack([tau_a, tau_b]),
        "grid_coords": jnp.asarray(molecule.grid.coords),
        "grid_weights": jnp.asarray(molecule.grid.weights),
        "coarse_0_atomic_coords": jnp.asarray(molecule.atom_coords),
    }


class OrigAdapter:
    def __init__(self, model):
        self.model = model

    def energy_from_density(self, params, molecule, density_total):
        return self.model.get_exc(build_skala_features(molecule, density_total))

    def scf_xc_energy_and_alpha_for_density(self, params, molecule, density):
        exc = self.energy_from_density(params, molecule, density)
        return exc, jnp.asarray(0.0, dtype=jnp.asarray(density).dtype)

    def energy_from_molecule(self, params, molecule):
        rdm1 = jnp.asarray(molecule.rdm1)
        density = rdm1.sum(axis=0) if rdm1.ndim == 3 else rdm1
        return self.energy_from_density(params, molecule, density)


def load_model():
    wd = get_default_weights_dir()
    cfg = load_config(wd)
    m = SkalaFunctional(
        lmax=cfg["lmax"], non_local=cfg["non_local"],
        non_local_hidden_nf=cfg["non_local_hidden_nf"],
        radius_cutoff=cfg["radius_cutoff"], key=jax.random.PRNGKey(0),
    )
    return load_weights_from_npz(m, wd)


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
        print(f"{label:10s} R={r:<4} conv={bool(info.converged)!s:5} cycles={int(info.cycles):3d} "
              f"rms={float(info.final_rms_density):.3e}", flush=True)


model = load_model()
run(OrigAdapter(model), CFG_ORIG, "orig/solver")
run(OrigAdapter(model), CFG_CLUSTER, "orig/clust")

try:
    from skalax_integration.adapter import make_skala_functional
    func = make_skala_functional(model)
    run(func, CFG_ORIG, "func/solver")
    run(func, CFG_CLUSTER, "func/clust")
except Exception as exc:  # noqa: BLE001
    print("Functional-based adapter unavailable:", repr(exc), flush=True)
