"""Reporting helpers for Skala SCF examples."""

from __future__ import annotations

import jax.numpy as jnp


def print_final_density_and_energy(result, total_energy) -> None:
    """Print the final total AO density matrix and total energy."""
    rdm1 = jnp.asarray(result.rdm1)
    density = rdm1.sum(axis=0) if rdm1.ndim == 3 else rdm1

    print("final density (total AO density matrix):")
    print(density)
    print(f"final total energy: {float(total_energy):.12f} Ha")
