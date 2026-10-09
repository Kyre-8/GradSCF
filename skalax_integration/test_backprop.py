"""Minimal check that the SCF total energy is differentiable in the Skala params.

Runs a tiny restricted KS-DFT (H2) with the Skala functional and asks JAX for
the gradient of the total energy with respect to the model parameters.  Two
backward modes are exercised: ``unrolled`` (chain rule through the SCF
iterations) and ``implicit`` (adjoint through the converged fixed point).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)

from gradscf.integrals import build_jk_from_eri_pair_matrix
from gradscf.scf import (
    DifferentiableSCF,
    DifferentiableSCFConfig,
    SCFDifferentiationConfig,
    restricted_molecule_from_spec_with_jax_rks,
)

from skalax_integration.adapter import SkalaFunctionalAdapter, load_skala_functional


def make_molecule(atom: str, basis: str = "sto-3g"):
    return restricted_molecule_from_spec_with_jax_rks(
        atom=atom,
        basis=basis,
        xc_spec="hf",
        unit="Angstrom",
        integral_backend="jax",
    )


def make_solver(mode: str, max_cycle: int = 30) -> DifferentiableSCF:
    return DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=max_cycle,
            damping=0.3,
            conv_tol_energy=1e-8,
            conv_tol_density=1e-8,
            differentiation=SCFDifferentiationConfig(mode=mode),
        )
    )


def make_energy_fn(molecule, solver):
    def total_energy(model):
        adapter = SkalaFunctionalAdapter(model)
        result, _info = solver.run(molecule, adapter, xc_params={})
        density = result.rdm1.sum(axis=0)
        h1e = jnp.asarray(molecule.h1e)
        coulomb, _exchange = build_jk_from_eri_pair_matrix(
            jnp.asarray(molecule.eri_pair_matrix), density
        )
        xc = adapter.energy_from_molecule({}, result)
        return (
            jnp.sum(density * h1e)
            + 0.5 * jnp.sum(density * coulomb)
            + xc
            + jnp.asarray(molecule.nuclear_repulsion)
        )

    return total_energy


def grad_stats(grads) -> tuple[float, float, int]:
    leaves = [
        l
        for l in jax.tree_util.tree_leaves(grads)
        if isinstance(l, jax.Array) and jnp.issubdtype(l.dtype, jnp.floating)
    ]
    norm = float(jnp.sqrt(sum(jnp.sum(l * l) for l in leaves)))
    n_params = sum(int(l.size) for l in leaves)
    return norm, norm, n_params


def main() -> None:
    molecule = make_molecule("H 0 0 0; H 0 0 0.74")
    model = load_skala_functional()

    for mode in ("unrolled", "implicit"):
        print(f"--- mode={mode} ---", flush=True)
        try:
            solver = make_solver(mode=mode, max_cycle=30)
            energy_fn = make_energy_fn(molecule, solver)
            value, grads = eqx.filter_value_and_grad(energy_fn)(model)
            norm, _acc, n_params = grad_stats(grads)
            print(f"  energy      = {float(value):.8f} Ha")
            print(f"  grad norm   = {norm:.6e}")
            print(f"  n trainable = {n_params}")
            print(f"  finite      = {bool(jnp.isfinite(jnp.asarray(norm)))}")
            print(f"  non-zero    = {norm > 0.0}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
