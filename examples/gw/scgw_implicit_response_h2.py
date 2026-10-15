"""Check dE/ds for DF vertices B -> s*B in finite-temperature H2 scGW.

Run from the repository root with PYTHONPATH=src. This checks the derivative
of the finite-grid self-consistent model, not a nuclear force or a basis/
grid-converged physical response. beta and nw remain static under AD.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import gto, scf
from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors
from gradscf.gw import scgw_matsubara_restricted
from gradscf.scf import SCFDifferentiationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--nw", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", cart=True)
    mf = scf.RHF(mol).run()
    if not mf.converged:
        raise ArithmeticError("HF reference did not converge.")
    ref = mf.scf_result
    factors = getattr(mf._scf_inputs, "df_factors", None)
    if factors is None:
        factors = eri_pair_matrix_to_df_factors(mf._scf_inputs.eri_pair_matrix, nao=ref.mo_coeff.shape[0], tol=1e-12)
    inputs = dict(mo_energy=ref.mo_energy, mo_coeff=ref.mo_coeff, nocc=1,
                  hcore_matrix=ref.hcore_matrix, nuclear_repulsion=ref.nuclear_repulsion,
                  beta=args.beta, nw=args.nw, max_iter=200, tol=1e-10, particle_tol=1e-12)
    policy = SCFDifferentiationConfig(tolerance=1e-9, max_iter=20, restart=30)
    def loss(scale, differentiation):
        return scgw_matsubara_restricted(**inputs, df_factors=factors * scale,
                                        differentiation=differentiation).total_energy
    started = time.perf_counter()
    value, derivative = jax.jit(jax.value_and_grad(lambda s: loss(s, policy)))(jnp.array(1.0))
    derivative = float(derivative)
    response_seconds = time.perf_counter() - started
    if not np.isfinite(derivative):
        raise ArithmeticError("Implicit response was rejected: check adjoint convergence and charge conditioning.")
    step = 1e-4
    finite_difference = float((loss(1 + step, None) - loss(1 - step, None)) / (2 * step))
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-5, atol=2e-7)
    record = dict(
        system="H2, 0.74 Angstrom, Cartesian STO-3G, HF start",
        beta=args.beta, nw=args.nw, primal_tol_Ha=1e-10, particle_tol=1e-12,
        response_tol=1e-9, response_restart=30, response_max_iter=20,
        charge_response_tol_electrons_per_Ha=1e-10, finite_difference_step=step,
        total_energy_Ha=float(value), dE_d_df_scale_Ha=derivative,
        finite_difference_Ha=finite_difference, absolute_error_Ha=abs(derivative - finite_difference),
        elapsed_response_seconds_including_compile=response_seconds,
        jax_version=jax.__version__, architecture=platform.machine(),
        devices=[str(d) for d in jax.devices()], dtype="float64/complex128",
    )
    print(json.dumps(record, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
