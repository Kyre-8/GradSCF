"""G02: complete nuclear gradients through HF and correlated wavefunctions.

Compare RHF, MP2 and CCSD in Cartesian 6-31G*. A separate STO-3G calculation
also includes FCI; its forces must not be compared as if they used 6-31G*.
Coordinates move native AO integrals, the AO metric, relaxed HF orbitals and
the correlated amplitudes/eigenstate. No frozen-orbital approximation is used.
"""
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf, fci
from gradscf.ao2mo import transform_integrals
from gradscf.solvers import EigenSolverConfig
from gradscf_tools.correlated_response import CorrelatedEnergy
from gradscf_tools.molecular_response import MolecularEnergy


atom = "O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11"
output = Path("reproducibility/ad_showcase/G02_water_correlated_forces")
output.mkdir(parents=True, exist_ok=True)
records = []
for basis in ("sto-3g", "6-31g*"):
    mol = gto.M(atom=atom, basis=basis, unit="Bohr", cart=True)
    mf = scf.RHF(mol, conv_tol=1e-12, conv_tol_density=1e-11,
                 integral_backend="native").run()
    hf = MolecularEnergy(mf)
    coordinates = hf.coordinates
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ)))
    nmo = mf.mo_coeff.shape[1]
    fci_space = fci.make_fci_space(nmo, (nocc, nocc)) if basis == "sto-3g" else None
    energies = {"hf": hf,
                "mp2": CorrelatedEnergy(mf, "mp2", gradient_tolerance=1e-9),
                "ccsd": CorrelatedEnergy(mf, "ccsd", gradient_tolerance=1e-9)}

    def fci_energy(coords):
        state, inputs = hf.solve_with_inputs(coords)
        h1, eri = transform_integrals(inputs["hcore"], state.mo_coeff, eri=inputs["eri"])
        result = fci.solve_fci(h1, eri, fci_space, ecore=inputs["nuclear_repulsion"],
                               config=EigenSolverConfig(nroots=1, atol=1e-10, max_subspace=50))
        valid = state.stationary & jnp.all(result.converged) & jnp.all(result.response_valid)
        return jnp.where(valid, result.total_energies[0], jnp.nan)

    if fci_space is not None:
        energies["fci"] = fci_energy

    direction = np.random.default_rng(23).normal(size=coordinates.shape)
    direction /= np.linalg.norm(direction)
    for method, energy in energies.items():
        value_and_grad = jax.jit(jax.value_and_grad(energy))
        start = perf_counter()
        value, gradient = value_and_grad(coordinates)
        gradient = np.asarray(gradient)
        cold = perf_counter()-start
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise RuntimeError(f"{basis}/{method} state or derivative did not converge.")
        np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-7, rtol=0)
        np.testing.assert_allclose(np.cross(np.asarray(coordinates), gradient).sum(axis=0), 0., atol=1e-7, rtol=0)
        forward = jax.jit(energy)
        checks = []
        for step in (1e-3, 3e-4, 1e-4):
            numerical = float((forward(coordinates+step*direction)
                               - forward(coordinates-step*direction))/(2*step))
            if not np.isfinite(numerical):
                raise RuntimeError(f"{basis}/{method} displaced state failed.")
            checks.append(dict(step_bohr=step, derivative=numerical,
                               error=abs(numerical-float(np.sum(gradient*direction)))))
        assert checks[-1]["error"] < 2e-6
        records.append(dict(basis=basis, method=method, energy_hartree=float(value),
                             gradient_hartree_per_bohr=gradient.tolist(), cold_seconds=cold,
                             directional_checks=checks))
        print(f"{basis:8s} {method:5s}: E={float(value):.12f}, "
              f"FD error={checks[-1]['error']:.3e} Eh/Bohr", flush=True)
        (output / "gradscf.json").write_text(json.dumps(dict(
            symbols=["O", "H", "H"], coordinates_bohr=np.asarray(coordinates).tolist(),
            cart=True, direction=direction.tolist(), records=records), indent=2) + "\n")

# Measured on jy_gpu_1, CPU float64, 2026-10-15; energies / Eh:
# STO-3G: HF -74.963082931333, MP2 -74.998659586331,
#         CCSD -75.012562094035, FCI -75.012678488615.
# 6-31G*: HF -76.010492818245, MP2 -76.199033408490, CCSD -76.207828747824.
# Maximum analytic-gradient differences from independent PySCF / Eh Bohr^-1:
# STO-3G: [1.414e-13, 2.762e-13, 8.614e-12, 2.979e-8].
# 6-31G*: [4.554e-12, 5.907e-9, 7.666e-8]. FCI is the small-basis reference only.
