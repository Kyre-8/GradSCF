"""End-to-end demo: a GradSCF self-consistent DFT run using the Skala XC.

This script builds a GradSCF restricted molecule (closed shell), swaps the
traditional XC for the neural Skala functional through
:class:`SkalaFunctionalAdapter`, runs the differentiable SCF, and prints the
HF reference energy alongside the Skala-DFT total energy.

Run from the repository root with the native integral backend already built:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
        .venv/bin/python -m skalax_integration.run_skala_scf --molecule h2
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from gradscf.integrals import build_jk_from_eri_pair_matrix
from gradscf.scf import DifferentiableSCF, DifferentiableSCFConfig
from gradscf.scf.builders import restricted_molecule_from_spec_with_jax_rks

from .adapter import SkalaFunctionalAdapter, load_skala_functional
from .report import print_final_density_and_energy


MOLECULES = {
    "h2": "H 0 0 0; H 0 0 0.74",
    "h2o": "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",
    "nh3": "N 0 0 0; H 0 0 1.0; H 0.94 0 -0.33; H -0.47 0.81 -0.33",
    "ch4": "C 0 0 0; H 0.63 0.63 0.63; H -0.63 -0.63 0.63; "
    "H -0.63 0.63 -0.63; H 0.63 -0.63 -0.63",
    "ben": "C 0.000000 1.402720 0.000000; "
    "C 0.698396 1.209657 0.000000; "
    "C -0.698396 1.209657 0.000000; "
    "C -1.396792 0.000000 0.000000; "
    "C -0.698396 -1.209657 0.000000; "
    "C 0.698396 -1.209657 0.000000; "
    "H 2.481527 0.000000 0.000000; "
    "H 1.240763 2.148986 0.000000; "
    "H -1.240763 2.148986 0.000000; "
    "H -2.481527 0.000000 0.000000; "
    "H -1.240763 -2.148986 0.000000; "
    "H 1.240763 -2.148986 0.000000",
    "naphthalene": "C 0.000000 1.402720 0.000000; "
    "C 1.214790 0.701360 0.000000; "
    "C 1.214790 -0.701360 0.000000; "
    "C 0.000000 -1.402720 0.000000; "
    "C -1.214790 -0.701360 0.000000; "
    "C -1.214790 0.701360 0.000000; "
    "C 2.456610 1.424910 0.000000; "
    "C 2.456610 -1.424910 0.000000; "
    "C -2.456610 -1.424910 0.000000; "
    "C -2.456610 1.424910 0.000000; "
    "H 0.000000 2.490290 0.000000; "
    "H 2.156550 2.507250 0.000000; "
    "H 2.156550 -2.507250 0.000000; "
    "H 0.000000 -2.490290 0.000000; "
    "H -2.156550 -2.507250 0.000000; "
    "H -2.156550 2.507250 0.000000; "
    "H 3.391100 0.000000 0.000000; "
    "H -3.391100 0.000000 0.000000"
}


def total_energy(molecule, result, adapter):
    """Restricted closed-shell total energy for a pure density functional."""
    density = result.rdm1.sum(axis=0)
    h1e = jnp.asarray(molecule.h1e)
    coulomb, exchange = build_jk_from_eri_pair_matrix(
        jnp.asarray(molecule.eri_pair_matrix), density
    )
    xc = adapter.energy_from_molecule({}, result)
    return (
        jnp.sum(density * h1e)
        + 0.5 * jnp.sum(density * coulomb)
        + xc
        + molecule.nuclear_repulsion
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", choices=sorted(MOLECULES), default="ben")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--max-cycle", type=int, default=60)
    parser.add_argument("--damping", type=float, default=0.3)
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    atom = MOLECULES[args.molecule]

    model = load_skala_functional()
    adapter = SkalaFunctionalAdapter(model)

    # HF reference provides the integrals, quadrature grid and an initial guess.
    started = time.perf_counter()
    molecule = restricted_molecule_from_spec_with_jax_rks(
        atom=atom,
        basis=args.basis,
        xc_spec="hf",
        unit="Angstrom",
        integral_backend="cpu",
    )

    solver = DifferentiableSCF(
        DifferentiableSCFConfig(
            mode="self_consistent",
            max_cycle=args.max_cycle,
            damping=args.damping,
            conv_tol_energy=1e-7,
            conv_tol_density=1e-7,
            conv_tol_grad=2e-6,
        )
    )
    result, info = solver.run(molecule, adapter, xc_params={})
    energy = total_energy(molecule, result, adapter)
    # print_final_density_and_energy(result, energy)
    elapsed = time.perf_counter() - started

    print(f"molecule      : {args.molecule} ({args.basis})")
    print(f"grid points   : {molecule.ao.shape[0]}")
    print(f"converged     : {bool(info.converged)} ({info.cycles} cycles)")
    print(f"E_HF          : {float(molecule.mf_energy):.8f} Ha")
    print(f"E_total(Skala): {float(energy):.8f} Ha")
    print(f"wall time     : {elapsed:.1f} s")


if __name__ == "__main__":
    main()
