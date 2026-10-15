"""Reproduce the finite-temperature H2 scGW grid sweep (Ha, beta in Ha^-1).

Example:
    PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python examples/gw/scgw_matsubara_h2.py \
        --beta 80 --nw 40 80 160 320 640 --output /tmp/scgw_h2_beta80.json

For a low-temperature check at fixed time resolution, compare
(beta,nw)=(40,160),(80,320),(160,640). These are small-basis implementation
checks, not a basis-converged physical benchmark.
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
import numpy as np

from gradscf import gto, scf
from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors
from gradscf.gw import scgw_matsubara_restricted
from gradscf.gw.g0w0 import _mo_factors
from gradscf.gw.matsubara import gw_matsubara_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=80.0)
    parser.add_argument("--nw", type=int, nargs="+", default=[40, 80, 160, 320, 640])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", cart=True)
    mf = scf.RHF(mol).run()
    if not mf.converged:
        raise ArithmeticError("HF reference did not converge.")
    ref = mf.scf_result
    factors = eri_pair_matrix_to_df_factors(mf._scf_inputs.eri_pair_matrix, nao=ref.mo_coeff.shape[0], tol=1e-12)
    b_mo = _mo_factors(factors, ref.mo_coeff)
    report = dict(
        system="H2, 0.74 Angstrom, STO-3G (Cartesian), HF start",
        jax_version=jax.__version__, architecture=platform.machine(),
        devices=[str(device) for device in jax.devices()],
        dtype="float64/complex128", beta=args.beta, tol_Ha=1e-7,
        particle_tol=1e-12, mixing=0.5, max_iter=400,
        hf_energy_Ha=float(ref.total_energy), results=[],
    )
    for nw in args.nw:
        started = time.perf_counter()
        out = scgw_matsubara_restricted(
            mo_energy=ref.mo_energy, mo_coeff=ref.mo_coeff, nocc=1,
            df_factors=factors, hcore_matrix=ref.hcore_matrix,
            nuclear_repulsion=float(ref.nuclear_repulsion), beta=args.beta, nw=nw,
            tol=1e-7, particle_tol=1e-12, max_iter=400,
        )
        mapped = gw_matsubara_step(out.green_iw, out.fock_mo, out.chemical_potential, b_mo, out.grid, out.sigma_moment)
        dielectric = np.eye(b_mo.shape[0])[None] - np.asarray(mapped["polarizability_inu"])
        record = dict(
            nw=nw, iterations=int(out.n_iter), total_energy_Ha=float(out.total_energy),
            gm_dynamic_energy_Ha=float(out.correlation_energy), mu_Ha=float(out.chemical_potential),
            density_eigenvalues=np.linalg.eigvalsh(out.density_mo).tolist(),
            minimum_dielectric_eigenvalue=float(np.linalg.eigvalsh(dielectric).min()),
            fixed_point_residual_Ha=float(out.fixed_point_residual),
            electron_number_error=float(out.particle_number_error),
            elapsed_seconds=time.perf_counter() - started,
        )
        report["results"].append(record)
        print(json.dumps(record), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
