"""Smoke test: self-consistent SCF convergence of the Skala functional on H2.

Runs the same solver configuration as skalax_integration/train_self_consistent.py
(max_cycle=40, damping=0.5, tol 1e-8) at several bond lengths and reports
convergence, cycles, final density RMS, and energy vs the HF reference.
"""

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

BOND_LENGTHS = [0.5, 0.7, 0.74, 1.0, 1.5, 2.0]


def main():
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
    model = load_skala_functional(key=jax.random.PRNGKey(0))
    adapter = SkalaFunctionalAdapter(model)

    print(f"{'R (A)':>6}  {'converged':>9}  {'cycles':>6}  {'rms_density':>11}  "
          f"{'E_skala (Ha)':>14}  {'E_hf (Ha)':>12}  {'dE (mHa)':>9}")
    n_ok = 0
    for r in BOND_LENGTHS:
        atom = f"H 0 0 0; H 0 0 {r}"
        mol = restricted_molecule_from_spec_with_jax_rks(
            atom=atom, basis="sto-3g", xc_spec="hf",
            unit="Angstrom", charge=0, spin=0, integral_backend="cpu",
        )
        result, info = solver.run(mol, adapter, xc_params={})
        converged = bool(info.converged)
        n_ok += converged
        e_hf = float(mol.mf_energy)
        rdm1 = jnp.asarray(result.rdm1)
        e_sk = float(total_energy(adapter, mol, rdm1))
        print(f"{r:6.2f}  {str(converged):>9}  {int(info.cycles):6d}  "
              f"{float(info.final_rms_density):11.3e}  "
              f"{e_sk:14.8f}  {e_hf:12.8f}  {1e3 * (e_sk - e_hf):9.3f}")

    print(f"\nconverged {n_ok}/{len(BOND_LENGTHS)}")


if __name__ == "__main__":
    main()
