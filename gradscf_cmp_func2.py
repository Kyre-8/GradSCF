"""Same molecule + same solver config; only the functional differs:
inline original adapter vs the cluster's Functional-based adapter."""
import importlib.util
import jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_enable_x64", True)

from gradscf.scf import (
    DifferentiableSCF, DifferentiableSCFConfig, SCFDifferentiationConfig,
    restricted_molecule_from_spec_with_jax_rks,
)
from skalax import SkalaFunctional, get_default_weights_dir, load_config, load_weights_from_npz


def build_skala_features(molecule, density_total):
    ao = jnp.asarray(molecule.ao); ao_deriv1 = jnp.asarray(molecule.ao_deriv1)
    density = jnp.asarray(density_total)
    rho = jnp.einsum("pq,gp,gq->g", density, ao, ao)
    grad = 2.0 * jnp.einsum("xrp,pq,rq->xr", ao_deriv1[1:4], density, ao)
    tau = 0.5 * jnp.einsum("xrp,pq,xrq->r", ao_deriv1[1:4], density, ao_deriv1[1:4])
    return {
        "density": jnp.stack([0.5 * rho, 0.5 * rho]),
        "grad": jnp.stack([0.5 * grad, 0.5 * grad]),
        "kin": jnp.stack([0.5 * tau, 0.5 * tau]),
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
        return self.energy_from_density(params, molecule, density), jnp.asarray(0.0, dtype=jnp.asarray(density).dtype)

    def energy_from_molecule(self, params, molecule):
        rdm1 = jnp.asarray(molecule.rdm1)
        density = rdm1.sum(axis=0) if rdm1.ndim == 3 else rdm1
        return self.energy_from_density(params, molecule, density)


def load_model():
    wd = get_default_weights_dir(); c = load_config(wd)
    return load_weights_from_npz(SkalaFunctional(
        lmax=c["lmax"], non_local=c["non_local"], non_local_hidden_nf=c["non_local_hidden_nf"],
        radius_cutoff=c["radius_cutoff"], key=jax.random.PRNGKey(0)), wd)


def load_adapter_module(path):
    spec = importlib.util.spec_from_file_location("skala_adapter_mod", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


CFG = dict(max_cycle=100, damping=0.6, level_shift=0.2, conv_tol_energy=2e-6, conv_tol_density=2e-6)


def run(adapter, xc_params, label, atoms):
    solver = DifferentiableSCF(DifferentiableSCFConfig(
        mode="self_consistent", differentiation=SCFDifferentiationConfig(mode="implicit"), **CFG))
    for k, atom in enumerate(atoms):
        mol = restricted_molecule_from_spec_with_jax_rks(
            atom=atom, basis="sto-3g", xc_spec="b3lyp", unit="Angstrom",
            charge=0, spin=0, integral_backend="cpu")
        result, info = solver.run(mol, adapter, xc_params=xc_params)
        print(f"{label:12s} mol{k} conv={bool(info.converged)!s:5} cycles={int(info.cycles):3d} "
              f"rms={float(info.final_rms_density):.3e}", flush=True)


ATOMS = ["C 0 0 0; C 0 0 1.39; H 0 0 -1.09; H 0 0 2.48",
         "O 0 0 0; H 0 0 0.96; H 0.93 0 0"]

model = load_model()
arrs, _ = eqx.partition(model, eqx.is_array)

run(OrigAdapter(model), {}, "orig", ATOMS)
mod = load_adapter_module("skalax_integration/adapter.py")
func = mod.make_skala_functional(model)
run(func, {"params": arrs}, "functional", ATOMS)
