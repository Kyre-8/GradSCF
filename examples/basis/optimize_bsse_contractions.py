#!/usr/bin/env python
"""Reduce water-dimer BSSE by differentiating shared STO-3G contractions.

Fixed S22 geometry/exponents/O 1s core; optimize O 2s, O 2p and H 1s.
All own/ghost calculations use the same updated contractions.
Run from the GradSCF root:
    PYTHONPATH=src python examples/basis/optimize_bsse_contractions.py
"""
import json
import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from gradscf import gto, integrals, scf
from gradscf.data.molecule import parse_molecule_spec
from gradscf.integrals.basis.contraction import primitive_basis, contraction_matrix, contract_integrals

KCAL = 627.5094740631
ENERGY_BUDGET = 1e-4  # Ha: maximum increase in each own-basis fragment energy.


def prepare_energy(atom, ghosts=()):
    """Cache primitives once and use the existing public implicit HF solver."""
    spec = parse_molecule_spec(atom)
    charges = np.asarray(spec.charges).copy()
    charges[list(ghosts)] = 0
    spec = replace(spec, charges=charges)
    mf = scf.RHF(gto.M(atom=spec, basis="sto-3g", cart=True),
                 conv_tol=1e-12, conv_tol_density=1e-10,
                 conv_tol_grad=1e-9, max_cycle=200).run()
    if not mf.converged:
        raise RuntimeError("Initial RHF did not converge.")
    top, params = integrals.prepare_basis(spec, "sto-3g", cart=True)
    pt, pp = primitive_basis(top, params)
    plan = integrals.make_plan(pt)
    primitive = dict(overlap=plan.evaluate("overlap", pp),
                     hcore=plan.evaluate("kinetic", pp)+plan.evaluate("nuclear", pp),
                     eri=plan.evaluate("eri", pp))
    occupations = np.stack([np.asarray(mf.mo_occ)/2] * 2)
    # Each OHH fragment has shells: O 1s (fixed), O 2s, O 2p, H 1s, H 1s.
    slots = (None, 0, 1, 2, 2) * (len(params.coefficients)//5)

    def energy(x):
        shared = jnp.concatenate([reference[:, :1], x.reshape(3, 2)], axis=1)
        coeffs = tuple(c if i is None else shared[i, :, None]
                       for c,i in zip(params.coefficients, slots))
        t = contraction_matrix(top, replace(params, coefficients=coeffs))
        data = contract_integrals(primitive, t)
        n = top.nao
        result = scf.minimize_roks_from_integrals(
            overlap=data["overlap"], hcore=data["hcore"], eri=data["eri"],
            nuclear_repulsion=spec.nuclear_repulsion,
            ao=jnp.zeros((0,n)), ao_deriv1=jnp.zeros((4,0,n)), grid_weights=jnp.zeros(0),
            mo_coeff=mf.mo_coeff, mo_occ=occupations, xc_spec="hf",
            orthonormalize_initial=True, max_iterations=300, gradient_tolerance=1e-9,
            differentiation=scf.SCFDifferentiationConfig(max_iter=100, restart=40),
        )
        # Shared spatial orbitals and equal 0/1 spin occupations give closed-shell RHF.
        return jnp.where(result.stationary, result.total_energy, jnp.nan)

    return energy, params.coefficients


system = json.loads(Path(__file__).with_name("s22_hbond_geometries.json").read_text()
                    )["systems"]["Water_dimer"]
atom = list(zip(system["symbols"], system["positions_angstrom"]))
a, b = system["fragments"]
prepared = [prepare_energy(atoms, ghosts) for atoms,ghosts in (
    (atom, ()), ([atom[i] for i in a], ()), ([atom[i] for i in b], ()),
    (atom, b), (atom, a))]
reference = jnp.stack([prepared[1][1][i][:,0] for i in (1,2,3)])
reference /= jnp.linalg.norm(reference, axis=1, keepdims=True)
# Anchor the first raw coefficient in each channel to remove scale freedom.
x0 = np.asarray(reference[:,1:]).reshape(-1)


def energies(x):
    return jnp.stack([entry[0](x) for entry in prepared])


def loss(x):
    e = energies(x)
    return (e[1]+e[2]-e[3]-e[4])*KCAL, e


value_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))
own_jacobian = jax.jit(jax.jacrev(lambda x: energies(x)[1:3]))


@lru_cache(maxsize=1)
def evaluate(x_tuple):
    x = jnp.asarray(x_tuple)
    (value, e), grad = value_grad(x)
    jac = own_jacobian(x)
    if not all(np.isfinite(np.asarray(y)).all() for y in (value,e,grad,jac)):
        raise RuntimeError("SCF stationarity or implicit-gradient check failed.")
    return float(value), np.asarray(grad), np.asarray(e), np.asarray(jac)


if __name__ == "__main__":
    before = evaluate(tuple(x0))[2]
    constraints = dict(type="ineq",
        fun=lambda x: before[1:3]+ENERGY_BUDGET-evaluate(tuple(x))[2][1:3],
        jac=lambda x: -evaluate(tuple(x))[3])
    fit = minimize(lambda x: evaluate(tuple(x))[:2], x0, jac=True, method="SLSQP",
        bounds=[(x-.15,x+.15) for x in x0], constraints=constraints,
        options=dict(maxiter=80, ftol=1e-8))
    after = evaluate(tuple(fit.x))[2]
    if (not fit.success or np.max(after[1:3]-before[1:3]) > ENERGY_BUDGET+1e-8
            or np.min(after[1:3]-after[3:5]) < -1e-8):
        raise RuntimeError(f"Constrained optimization failed: {fit.message}")
    print(f"Converged in {fit.nit} iterations.")
    print("                            Initial/Ha         Optimized/Ha")
    for name,e0,e1 in zip(("AB","A own","B own","A ghost B","B ghost A"), before, after):
        print(f"{name:22s} {e0:18.12f} {e1:18.12f}")
    for name,weights in (("Raw interaction",[1,-1,-1,0,0]),
                         ("CP interaction",[1,0,0,-1,-1]), ("BSSE",[0,1,1,-1,-1])):
        print(f"{name:22s} {np.asarray(weights)@before*KCAL:12.8f}"
              f" {np.asarray(weights)@after*KCAL:12.8f} kcal/mol")
    optimized = np.column_stack([np.asarray(reference[:,0]), fit.x.reshape(3,2)])
    optimized /= np.linalg.norm(optimized, axis=1, keepdims=True)
    print("Unit-scale raw coefficients (physical AO normalization is applied separately):")
    for name,c0,c1 in zip(("O 2s","O 2p","H 1s"), np.asarray(reference), optimized):
        print(f"{name}: {c0.tolist()} -> {c1.tolist()}")


# Reference output (2026-10-08), CPU float64, fixed S22 water geometry.
# Contractions: O 2s, O 2p and H 1s, shared between both fragments.
# Geometry/exponents/O 1s core coefficients are fixed; all electrons included.
# Six free coefficients bounded to reference +/-0.15 in the fixed scale gauge.
# Own-fragment energy increase budget: 1e-4 Ha per fragment.
# SLSQP converged in 2 iterations. The solution reaches coefficient bounds.
#
# Energy component        Initial/Ha          Optimized/Ha
# AB                    -149.935375926426  -150.061446791554
# A own                  -74.963402136324   -75.023208200585
# B own                  -74.963160069924   -75.023035809071
# A ghost B              -74.963543825991   -75.023354621529
# B ghost A              -74.969599187153   -75.026211694200
#
# Quantity                 Initial             Optimized (kcal/mol)
# Raw interaction             -5.53069291        -9.53988967
# CP interaction              -1.40117424        -7.45511114
# BSSE                         4.12951867         2.08477854
#
# BSSE decreases by 49.52%; both own-fragment energies decrease.
# The CP interaction changes substantially; BSSE reduction alone does not
# establish improved interaction-energy accuracy. This is a local demonstration.
