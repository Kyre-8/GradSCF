"""RHF H4 ground-state CC hierarchy and a fixed-orbital response demonstration."""

import json
import platform
from time import perf_counter
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf, cc
from gradscf.cc.integrals import prepare_integrals
from gradscf.scf.reference import reference_from_source


def main():
    atom = "H 0 0 0; H 0 0 .8; H 0 0 1.9; H 0 0 3.1"
    mf = scf.RHF(
        gto.M(atom=atom, basis="sto-3g", unit="Angstrom"), conv_tol=1e-12
    ).run()
    report = {
        "atom": atom,
        "basis": "sto-3g",
        "geometry_unit": "Angstrom",
        "energy_unit": "Hartree",
        "machine": platform.machine(),
        "jax": jax.__version__,
        "dtype": "float64",
        "devices": [str(d) for d in jax.devices()],
        "hf_energy": float(mf.e_tot),
        "methods": {},
    }
    for method in ["CCS", "CCD", "LCCD", "LCCSD", "CC2", "CCSD"]:
        start = perf_counter()
        obj = getattr(cc, method)(mf).run()
        if not obj.converged:
            raise RuntimeError(f"{method} failed to converge")
        row = {
            "energy": float(obj.e_tot),
            "correlation_energy": float(obj.e_corr),
            "residual_norm": float(obj.result.residual_norm),
            "cycles": int(obj.result.iterations),
            "seconds_including_compile": perf_counter() - start,
        }
        report["methods"][method] = row
        if method == "CCSD":
            obj.solve_lambda()
            if not obj.converged_lambda:
                raise RuntimeError("Lambda failed to converge")
            row["lambda_residual"] = float(obj.lambda_result.residual_norm)
            triples = obj.ccsd_t()
            row["triples_correction"] = float(triples)
            row["ccsd_t_total"] = float(obj.e_tot + triples)
            urban = obj.triples(variant="ccsd+t(ccsd)")
            conventional = obj.triples()
            row["urban_triples_correction"] = float(urban.energy)
            row["triples_singles_component"] = float(conventional.singles_component)
            row["triples_min_abs_denominator"] = float(conventional.min_abs_denominator)
            density = obj.make_rdm1()
            row["rdm1_trace"] = float(jnp.trace(density))
            row["rdm1_symmetry_error"] = float(jnp.max(jnp.abs(density - density.T)))
            # The fixed-integral response below explicitly needs a small full-MO
            # reference; the CC object's selected-block source does not own it.
            reference = reference_from_source(mf, max_memory=mf.max_memory)
    h, g = reference.h1, reference.eri
    potential = prepare_integrals(h, g, nocc=reference.nocc).fock - h
    cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)

    def energy(t):
        ht, gt = h - t * potential, g * (1 + t)
        result = cc.run_cc(
            ht,
            gt,
            nocc=reference.nocc,
            nuclear_repulsion=reference.nuclear_repulsion,
            config=cfg,
        )
        return result.total_energy + cc.triples_correction(
            ht, gt, result, nocc=reference.nocc
        )

    ad = jax.jit(jax.grad(energy))(0.0)
    fd = (energy(1e-4) - energy(-1e-4)) / 2e-4
    report["response"] = {
        "parameter": "coupling scale at fixed canonical Fock matrix",
        "ad": float(ad),
        "finite_difference": float(fd),
        "step": 1e-4,
        "absolute_error": float(jnp.abs(ad - fd)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
