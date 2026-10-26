"""Optimize shared 3-21G contractions using native integrals and JAX gradients.

Only coefficients vary. Fixed primitive integrals are cached outside AD.
The converged RHF Lagrangian includes the overlap/Pulay term; no derivative
through native calls or SCF eigendecompositions is assumed.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize

from gradscf import integrals, scf
from gradscf.integrals.basis.normalization import normalized_shell_coefficients, radial_primitive_norm
from gradscf.integrals.molecular.jk import exchange_matrix
from gradscf.scf.core import _diagonalize_fock, _orthogonalizer
from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable

# Same fixed geometry as tests/comparisons/trace_benzene_pbe_tddft_davidson.py (Angstrom).
BENZENE_ATOM = """
C 0.000000 1.396792 0.000000
C -1.209657 0.698396 0.000000
C -1.209657 -0.698396 0.000000
C 0.000000 -1.396792 0.000000
C 1.209657 -0.698396 0.000000
C 1.209657 0.698396 0.000000
H 0.000000 2.484212 0.000000
H -2.151390 1.242106 0.000000
H -2.151390 -1.242106 0.000000
H 0.000000 -2.484212 0.000000
H 2.151390 -1.242106 0.000000
H 2.151390 1.242106 0.000000
"""


class ContractionExperiment:
    def __init__(self, molecule="benzene"):
        self.atom = BENZENE_ATOM if molecule == "benzene" else "H 0 0 0; H 0 0 .74"
        self.top, self.params = integrals.prepare_basis(atom=self.atom, basis="3-21g", unit="Angstrom")
        self.nelectron = sum(self.top.nuclear_charges)
        self.enuc = scf.nuclear_repulsion_energy(self.params.nuclear_coords,
                                                jnp.asarray(self.top.nuclear_charges))
        self.plan = integrals.make_plan(self.top, backend="native")
        self.groups, self.shell_groups = [], []
        counts = [0]*len(self.top.nuclear_charges)
        keys = {}
        offset = 0
        for i, (a, c, center) in enumerate(zip(self.params.exponents, self.params.coefficients,
                                             self.params.centers)):
            atom = int(np.argmin(np.linalg.norm(np.asarray(self.params.nuclear_coords-center), axis=1)))
            key = (self.top.nuclear_charges[atom], counts[atom])
            counts[atom] += 1
            if c.shape[1] != 1:
                raise ValueError("This 3-21G experiment expects one contraction per shell")
            if key not in keys:
                tangent = null_space(np.asarray(c[:, 0])[None, :])
                group = dict(key=key, shell=i, raw=c[:, 0], tangent=jnp.asarray(tangent),
                             start=offset, stop=offset+tangent.shape[1])
                keys[key] = len(self.groups)
                self.groups.append(group)
                offset += tangent.shape[1]
            self.shell_groups.append(keys[key])
        self.x0 = np.zeros(offset)
        self.primitive_top = replace(self.top, contraction_counts=self.top.primitive_counts)
        primitive_params = replace(self.params, coefficients=tuple(jnp.eye(n) for n in self.top.primitive_counts))
        pp = integrals.make_plan(self.primitive_top, backend="native")
        print(f"AO={self.top.nao}, primitive AO={self.primitive_top.nao}, variables={offset}; "
              f"primitive ERI={self.primitive_top.nao**4*8/1e9:.3f} GB", flush=True)
        started = time.perf_counter()
        self.ps = pp.evaluate("overlap", primitive_params)
        self.ph = pp.evaluate("kinetic", primitive_params)+pp.evaluate("nuclear", primitive_params)
        self.peri = pp.evaluate("eri", primitive_params)
        self.peri.block_until_ready()
        self.integral_seconds = time.perf_counter()-started
        print(f"Fixed native primitive integrals ready: {self.integral_seconds:.2f} s", flush=True)
        self.gradient = jax.jit(jax.value_and_grad(self.lagrangian, has_aux=True))
        self.solve = jax.jit(self._solve)

    def coefficients(self, x):
        shared = []
        for g in self.groups:
            raw = g["raw"]
            v = raw+g["tangent"]@x[g["start"]:g["stop"]]
            shared.append(v*jnp.linalg.norm(raw)/jnp.linalg.norm(v))
        return tuple(shared[i][:, None] for i in self.shell_groups)

    def transform(self, x):
        t = jnp.zeros((self.primitive_top.nao, self.top.nao))
        pi = ci = 0
        for l, a, raw in zip(self.top.angular_momenta, self.params.exponents, self.coefficients(x)):
            nangular = (l+1)*(l+2)//2
            c = normalized_shell_coefficients(l, a, raw)[:, 0]/radial_primitive_norm(l, a)
            block = jnp.kron(c[:, None], jnp.eye(nangular))
            t = t.at[pi:pi+len(a)*nangular, ci:ci+nangular].set(block)
            pi += len(a)*nangular
            ci += nangular
        return t

    def lagrangian(self, x, dm, w, ps, ph, peri):
        t = self.transform(x)
        dp = t@dm@t.T
        j = jnp.einsum("pqrs,rs->pq", peri, dp)
        k = exchange_matrix(peri, dp)
        e = jnp.sum(dp*ph)+.5*jnp.sum(dp*j)-.25*jnp.sum(dp*k)+self.enuc
        lagrangian = e-jnp.sum((t@w@t.T)*ps)
        return lagrangian, e

    def _solve(self, s, h, eri):
        n = s.shape[0]
        result = run_rks_from_integrals_traceable(
            overlap=s, hcore=h, eri=eri, nelectron=self.nelectron, nuclear_repulsion=self.enuc,
            ao=jnp.zeros((0,n)), ao_deriv1=jnp.zeros((4,0,n)), grid_weights=jnp.zeros(0),
            config=RKSConfig(xc_spec="hf", max_cycle=100, conv_tol=1e-12, conv_tol_density=1e-10))
        # The solver's final energy-based acceptance can leave a small orbital
        # residual. The stationary Lagrangian needs tighter stationarity than
        # an energy-only comparison, so polish only this experiment's orbitals.
        x = _orthogonalizer(s, 1e-10)
        nocc = self.nelectron//2

        def refine(state):
            count, _, _, _, fock, _ = state
            eps, c = _diagonalize_fock(fock, x)
            dm = (c*result.mo_occ[None, :])@c.T
            j = jnp.einsum("pqrs,rs->pq", eri, dm)
            k = exchange_matrix(eri, dm)
            fock = h+j-.5*k
            residual = jnp.max(jnp.abs(fock@c[:, :nocc]-(s@c[:, :nocc])*eps[None, :nocc]))
            return count+1, dm, c, eps, fock, residual

        count, dm, c, eps, fock, residual = jax.lax.while_loop(
            lambda state: (state[0] < 50) & (state[-1] > 1e-10), refine,
            (jnp.asarray(0), result.density_matrix, result.mo_coeff, result.mo_energy,
             result.fock_matrix, jnp.asarray(jnp.inf)))
        energy = .5*jnp.sum(dm*(h+fock))+self.enuc
        return replace(result, total_energy=energy, electronic_energy=energy-self.enuc,
                       density_matrix=dm, mo_coeff=c, mo_energy=eps, fock_matrix=fock,
                       cycles=result.cycles+count, converged=result.converged & (residual <= 1e-10))

    def evaluate(self, x, *, gradient=True):
        started = time.perf_counter()
        params = replace(self.params, coefficients=self.coefficients(jnp.asarray(x)))
        s = self.plan.evaluate("overlap", params)
        h = self.plan.evaluate("kinetic", params)+self.plan.evaluate("nuclear", params)
        eri = self.plan.evaluate("eri", params)
        result = self.solve(s, h, eri)
        if not bool(result.converged):
            raise RuntimeError("RHF failed to converge")
        energy = float(result.total_energy)
        row = dict(energy_hartree=energy, scf_cycles=int(result.cycles))
        if gradient:
            w = (result.mo_coeff*(result.mo_occ*result.mo_energy)[None, :])@result.mo_coeff.T
            (_, reconstructed), grad = self.gradient(jnp.asarray(x), result.density_matrix, w,
                                                      self.ps, self.ph, self.peri)
            error = abs(float(reconstructed)-energy)
            if error > 1e-8:
                raise RuntimeError(f"Primitive/contracted energy mismatch: {error} Ha")
            grad = np.asarray(grad)
            if not np.all(np.isfinite(grad)):
                raise RuntimeError("Nonfinite gradient")
            row.update(gradient=grad.tolist(), max_gradient=float(np.max(np.abs(grad))),
                       primitive_energy_error=error)
        row["elapsed_seconds"] = time.perf_counter()-started
        return row

    def basis_dict(self, x):
        cs = self.coefficients(jnp.asarray(x))
        basis = {}
        for group in self.groups:
            z, _ = group["key"]
            symbol = {1:"H", 6:"C"}[z]
            i = group["shell"]
            rows = [[float(a), float(c)] for a, c in zip(self.params.exponents[i], cs[i][:, 0])]
            basis.setdefault(symbol, []).append([self.top.angular_momenta[i], *rows])
        return basis


def check_gradient(experiment, x, row, step=1e-4):
    # Check every independent parameter using independently reconverged SCFs.
    fd = []
    for i in range(len(x)):
        delta = np.zeros_like(x)
        delta[i] = step
        plus = experiment.evaluate(x+delta, gradient=False)["energy_hartree"]
        minus = experiment.evaluate(x-delta, gradient=False)["energy_hartree"]
        plus2 = experiment.evaluate(x+2*delta, gradient=False)["energy_hartree"]
        minus2 = experiment.evaluate(x-2*delta, gradient=False)["energy_hartree"]
        fd.append((8*(plus-minus)-plus2+minus2)/(12*step))
    error = float(np.max(np.abs(np.asarray(fd)-row["gradient"])))
    np.testing.assert_allclose(row["gradient"], fd, atol=2e-7, rtol=2e-5)
    return dict(step=step, stencil="four-point central", finite_difference=fd, max_absolute_error=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", choices=("benzene", "h2"), default="benzene")
    parser.add_argument("--maxiter", type=int, default=15)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benzene-321g-contractions"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    started = time.perf_counter()
    experiment = ContractionExperiment(args.molecule)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cache, history = {}, []

    def evaluate(x):
        key = tuple(x)
        if key not in cache:
            cache[key] = experiment.evaluate(x)
            print(f"eval={len(cache)} E={cache[key]['energy_hartree']:.12f} "
                  f"max|g|={cache[key]['max_gradient']:.3e} "
                  f"time={cache[key]['elapsed_seconds']:.2f}s", flush=True)
        return cache[key]

    def record(x):
        row = dict(iteration=len(history), parameters=list(x), **evaluate(x))
        history.append(row)
        (output/"history.json").write_text(json.dumps(history, indent=2)+"\n")

    record(experiment.x0)
    if args.preflight:
        (output/"preflight.json").write_text(json.dumps(dict(
            initial=history[0], integral_seconds=experiment.integral_seconds,
            elapsed_seconds=time.perf_counter()-started), indent=2)+"\n")
        return
    initial_fd = check_gradient(experiment, experiment.x0, history[0])
    print("Initial gradient FD check:", initial_fd, flush=True)

    def objective(x):
        row = evaluate(x)
        return row["energy_hartree"], np.asarray(row["gradient"])

    result = minimize(objective, experiment.x0, jac=True, method="BFGS", callback=record,
                      options={"gtol":1e-6, "maxiter":args.maxiter})
    if not np.array_equal(history[-1]["parameters"], result.x):
        record(result.x)
    final_fd = check_gradient(experiment, result.x, evaluate(result.x))
    summary = dict(molecule=args.molecule, atom_angstrom=experiment.atom, method="RHF", basis="3-21g",
                   integral_backend="native", derivative="JAX converged RHF Lagrangian including Pulay overlap term",
                   dtype="float64", jax_version=jax.__version__, devices=[str(d) for d in jax.devices()],
                   optimizer="BFGS", gtol=1e-6, optimizer_success=bool(result.success), message=str(result.message),
                   iterations=int(result.nit), evaluations=len(cache), initial=history[0], final=history[-1],
                   energy_decrease_hartree=history[0]["energy_hartree"]-history[-1]["energy_hartree"],
                   initial_basis=experiment.basis_dict(experiment.x0), optimized_basis=experiment.basis_dict(result.x),
                   initial_fd=initial_fd, final_fd=final_fd,
                   elapsed_seconds=time.perf_counter()-started)
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (output/"optimized_basis.json").write_text(json.dumps(summary["optimized_basis"], indent=2)+"\n")
    with (output/"optimization.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "energy_hartree", "max_gradient", "scf_cycles"])
        writer.writerows([r[k] for k in ("iteration", "energy_hartree", "max_gradient", "scf_cycles")] for r in history)
    print(f"{result.message}; decrease={summary['energy_decrease_hartree']:.12g} Ha; "
          f"final max|g|={history[-1]['max_gradient']:.3e}; elapsed={summary['elapsed_seconds']:.2f}s", flush=True)
    print("Final FD:", final_fd, flush=True)


if __name__ == "__main__":
    main()
