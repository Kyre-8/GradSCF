"""Isolate which SCF option breaks convergence: eval on real QM8 molecules with
the original adapter, sweeping eigenvalue_jitter / iterate_selection."""
import sys
import jax, jax.numpy as jnp
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


class Adapter:
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


def parse_first(path, n):
    lines = open(path).read().splitlines()
    specs, i = [], 0
    while i < len(lines) and len(specs) < n:
        if not lines[i].strip():
            i += 1; continue
        nat = int(lines[i].strip())
        specs.append("; ".join(f"{f[0]} {f[1]} {f[2]} {f[3]}"
                               for f in (ln.split() for ln in lines[i+2:i+2+nat])))
        i += nat + 2
    return specs


wd = get_default_weights_dir(); c = load_config(wd)
model = load_weights_from_npz(SkalaFunctional(
    lmax=c["lmax"], non_local=c["non_local"], non_local_hidden_nf=c["non_local_hidden_nf"],
    radius_cutoff=c["radius_cutoff"], key=jax.random.PRNGKey(0)), wd)
adapter = Adapter(model)

mols = [restricted_molecule_from_spec_with_jax_rks(
    atom=a, basis="sto-3g", xc_spec="b3lyp", unit="Angstrom",
    charge=0, spin=0, integral_backend="cpu")
    for a in parse_first(sys.argv[1] if len(sys.argv) > 1 else "XYZ_B3LYP_631G2dfp.xyz", 5)]

BASE = dict(max_cycle=100, damping=0.6, level_shift=0.2,
            conv_tol_energy=2e-6, conv_tol_density=2e-6)
VARIANTS = [
    ("base", {}),
    ("jitter1e-3", dict(eigenvalue_jitter=1e-3)),
    ("best_rms", dict(iterate_selection="best_rms")),
    ("train(cfg)", dict(eigenvalue_jitter=1e-3, iterate_selection="best_rms")),
]

for label, extra in VARIANTS:
    solver = DifferentiableSCF(DifferentiableSCFConfig(
        mode="self_consistent", differentiation=SCFDifferentiationConfig(mode="implicit"),
        **BASE, **extra))
    out = []
    for k, mol in enumerate(mols):
        _, info = solver.run(mol, adapter, xc_params={})
        out.append(f"m{k}:{'T' if bool(info.converged) else 'F'}({int(info.cycles)},{float(info.final_rms_density):.1e})")
    print(f"{label:12s} " + "  ".join(out), flush=True)
