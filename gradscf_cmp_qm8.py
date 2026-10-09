"""Run the ORIGINAL Skala adapter on real QM8 molecules (b3lyp ref, sto-3g)
with the cluster's solver config, to confirm convergence."""
import sys
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
        exc = self.energy_from_density(params, molecule, density)
        return exc, jnp.asarray(0.0, dtype=jnp.asarray(density).dtype)

    def energy_from_molecule(self, params, molecule):
        rdm1 = jnp.asarray(molecule.rdm1)
        density = rdm1.sum(axis=0) if rdm1.ndim == 3 else rdm1
        return self.energy_from_density(params, molecule, density)


def parse_first(path, n):
    lines = open(path).read().splitlines()
    specs, i = [], 0
    while i < len(lines) and len(specs) < n:
        if not lines[i].strip():
            i += 1
            continue
        nat = int(lines[i].strip())
        atoms = [f"{f[0]} {f[1]} {f[2]} {f[3]}"
                 for f in (ln.split() for ln in lines[i + 2:i + 2 + nat])]
        specs.append("; ".join(atoms))
        i += nat + 2
    return specs


wd = get_default_weights_dir()
c = load_config(wd)
model = load_weights_from_npz(SkalaFunctional(
    lmax=c["lmax"], non_local=c["non_local"], non_local_hidden_nf=c["non_local_hidden_nf"],
    radius_cutoff=c["radius_cutoff"], key=jax.random.PRNGKey(0)), wd)
adapter = OrigAdapter(model)

xyz = sys.argv[1] if len(sys.argv) > 1 else "XYZ_B3LYP_631G2dfp.xyz"
solver = DifferentiableSCF(DifferentiableSCFConfig(
    mode="self_consistent", max_cycle=100, damping=0.6, level_shift=0.2,
    conv_tol_energy=2e-6, conv_tol_density=2e-6,
    differentiation=SCFDifferentiationConfig(mode="implicit")))

for k, atom in enumerate(parse_first(xyz, 5)):
    mol = restricted_molecule_from_spec_with_jax_rks(
        atom=atom, basis="sto-3g", xc_spec="b3lyp", unit="Angstrom",
        charge=0, spin=0, integral_backend="cpu")
    result, info = solver.run(mol, adapter, xc_params={})
    print(f"qm8[{k}] natoms={atom.count(';')+1:3d} conv={bool(info.converged)!s:5} "
          f"cycles={int(info.cycles):3d} rms={float(info.final_rms_density):.3e} "
          f"E_hf_per_atom={float(mol.mf_energy)/(atom.count(';')+1):.4f}", flush=True)
