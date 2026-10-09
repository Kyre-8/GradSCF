"""Smoke test: self-consistent SCF convergence of the Skala functional on benzene.

Same solver configuration as skalax_integration/train_self_consistent.py
(max_cycle=40, damping=0.5, tol 1e-8), STO-3G, HF reference.
Regular hexagon geometry (C-C 1.39 A, C-H 1.09 A, in the xy plane);
also tests uniformly scaled variants to probe robustness.
"""

import math

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gradscf.integrals import build_jk_from_eri_pair_matrix
from gradscf.scf import (
    DifferentiableSCF,
    DifferentiableSCFConfig,
    SCFDifferentiationConfig,
    restricted_molecule_from_spec_with_jax_rks,
)
from skalax_integration.adapter import SkalaFunctionalAdapter, load_skala_functional

R_CC = 1.39
R_CH = 1.09


def benzene_atom(scale: float = 1.0) -> str:
    lines = []
    for i in range(6):
        theta = math.radians(60 * i)
        c = R_CC * scale
        h = (R_CC + R_CH) * scale
        lines.append(f"C {c * math.cos(theta):.8f} {c * math.sin(theta):.8f} 0.0")
    for i in range(6):
        theta = math.radians(60 * i)
        h = (R_CC + R_CH) * scale
        lines.append(f"H {h * math.cos(theta):.8f} {h * math.sin(theta):.8f} 0.0")
    return "; ".join(lines)


def total_energy(adapter, molecule, rdm1):
    density = rdm1.sum(axis=0) if rdm1.ndim == 3 else rdm1
    h1e = jnp.asarray(molecule.h1e)
    coulomb, _ = build_jk_from_eri_pair_matrix(
        jnp.asarray(molecule.eri_pair_matrix), density
    )
    xc = adapter.energy_from_density({}, molecule, density)
    return (
        jnp.sum(density * h1e)
        + 0.5 * jnp.sum(density * coulomb)
        + xc
        + jnp.asarray(molecule.nuclear_repulsion)
    )


def main():
    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=40,
            damping=0.5,
            conv_tol_energy=1e-8,
            conv_tol_density=1e-8,
            conv_tol_grad=1e-5,
            differentiation=SCFDifferentiationConfig(mode="implicit"),
        )
    )
    model = load_skala_functional(key=jax.random.PRNGKey(0))
    adapter = SkalaFunctionalAdapter(model)

    print(f"{'scale':>5}  {'converged':>9}  {'cycles':>6}  {'rms_density':>11}  "
          f"{'E_skala (Ha)':>14}  {'E_hf (Ha)':>13}  {'dE (Ha)':>9}", flush=True)
    n_ok = 0
    scales = [0.9, 1.0, 1.1]
    for s in scales:
        mol = restricted_molecule_from_spec_with_jax_rks(
            atom=benzene_atom(s), basis="sto-3g", xc_spec="hf",
            unit="Angstrom", charge=0, spin=0, integral_backend="cpu",
        )
        result, info = solver.run(mol, adapter, xc_params={})
        converged = bool(info.converged)
        n_ok += converged
        e_hf = float(mol.mf_energy)
        rdm1 = jnp.asarray(result.rdm1)
        e_sk = float(total_energy(adapter, mol, rdm1))
        print(f"{s:5.2f}  {str(converged):>9}  {int(info.cycles):6d}  "
              f"{float(info.final_rms_density):11.3e}  "
              f"{e_sk:14.8f}  {e_hf:13.8f}  {e_sk - e_hf:9.4f}", flush=True)

    print(f"\nconverged {n_ok}/{len(scales)}")


if __name__ == "__main__":
    main()
