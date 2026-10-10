"""Ne and F- / spherical aug-cc-pVDZ, frozen 1s, no virtual truncation.

GradSCF independently computes native HF/MP2/MP3. The complete high-order
reference uses PySCF's Ag/D2h compressed Hamiltonian and NumPy RS recurrence.
This is NOT a GradSCF high-order calculation: its current representations do
not fit this space. Do not raise the tensor capacity bound to bypass that fact.

References: Olsen et al., JCP 112, 9736 (2000), Table III (Ne E2--E25);
Leininger et al., JCP 112, 9213 (2000), Table I and Fig. 7 (Ne/F-).
"""
import json
from math import comb
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from pyscf import gto as pgto, scf as pscf, lib, mp as pmp

jax.config.update("jax_enable_x64", True)

from gradscf import mp
from gradscf.integrals.basis import prepare_basis
from gradscf.integrals import make_plan
from gradscf.integrals.molecular.ao2mo import transform_integrals
from gradscf.scf.rhf import run_rhf_from_integrals, RHFConfig
from _pyscf_reference import symmetry_reference, rs_coefficients

order = 25
folder = Path("artifacts/mp_convergence")
folder.mkdir(parents=True,exist_ok=True)

for symbol,charge in (("Ne",0),("F",-1)):
    start = perf_counter()
    # Native spherical integrals use the integral-level SCF entry point because
    # the grid-bearing molecular facade currently requires Cartesian AOs.
    top,params = prepare_basis(symbol+" 0 0 0","aug-cc-pvdz",charge=charge,cart=False)
    plan = make_plan(top,backend="native")
    s = plan.evaluate("overlap",params)
    h = plan.evaluate("kinetic",params)+plan.evaluate("nuclear",params)
    g = plan.evaluate("eri",params)
    mf = run_rhf_from_integrals(overlap=s,hcore=h,eri=g,nelectron=10,
        nuclear_repulsion=0.,config=RHFConfig(conv_tol=1e-13,conv_tol_grad=1e-10,max_cycle=200))
    if not mf.converged:
        raise RuntimeError("Native RHF did not converge")
    hmo,gmo = transform_integrals(h,mf.mo_coeff,eri=g)
    native = mp.run_mp(hmo,gmo,nocc=5,frozen=1,
                      config=mp.MPConfig(order=3,algorithm="direct",with_t2=False))
    if not native.valid:
        raise RuntimeError("Native MP reference invalid")

    # Independent orbitals, integrals and frozen-core folding.
    mol = pgto.M(atom=symbol+" 0 0 0",basis="aug-cc-pvdz",charge=charge,
                 cart=False,symmetry="D2h",verbose=0)
    ref = pscf.RHF(mol).run(conv_tol=1e-13,conv_tol_grad=1e-10,max_cycle=200)
    assert ref.converged
    np.testing.assert_allclose(mf.total_energy,ref.e_tot,atol=2e-9,rtol=0.)
    np.testing.assert_allclose(native.e2,pmp.MP2(ref,frozen=1).run().e_corr,atol=2e-10,rtol=0.)
    action,phi,gaps,diagonal,ecore = symmetry_reference(ref,frozen=1)
    full_size = comb(top.nao-1,4)**2
    tick = perf_counter()
    first = action(phi)
    sigma_seconds = perf_counter()-tick
    print("%s: %d spherical AOs, full=%d, symmetry-packed=%d, first sigma %.3f s" %
          (symbol,top.nao,full_size,phi.size,sigma_seconds),flush=True)
    print("Native HF %.12f, E2 %.12f, E3 %.12f" %
          (mf.total_energy,native.e2,native.e3),flush=True)
    checkpoint = folder/(symbol.lower()+"_progress.json")
    history = []
    def progress(degree,value):
        history.append(dict(order=degree,correction=value,elapsed=perf_counter()-start))
        checkpoint.write_text(json.dumps(history,indent=2)+"\n")
        print("%s E%02d = % .12e Ha; elapsed %.1f s" %
              (symbol,degree,value,perf_counter()-start),flush=True)
    corrections = rs_coefficients(action,phi,gaps,order,callback=progress)
    np.testing.assert_allclose([native.e2,native.e3],corrections[:2],atol=2e-9,rtol=0.)

    # A separate variational reference in exactly the same compressed space.
    # Delegate iteration/preconditioning to PySCF; do not implement a solver.
    print(symbol+": computing independent FCI energy",flush=True)
    def fci_progress(state):
        if state["icyc"] % 5 == 0:
            print("%s FCI cycle %d, residual %.3e Ha" %
                  (symbol,state["icyc"],max(state["dx_norm"])),flush=True)
    converged,energies,vectors = lib.davidson1(lambda xs:[action(x) for x in xs],
        [phi],diagonal,tol=1e-11,tol_residual=1e-8,max_cycle=100,
        max_space=16,max_memory=4000,lindep=1e-20,verbose=0,callback=fci_progress)
    if not converged[0]:
        raise RuntimeError("External compressed FCI did not converge")
    fci_energy = float(energies[0]+ecore)
    residual = float(np.linalg.norm(action(vectors[0])-energies[0]*vectors[0]))
    report = dict(system=symbol,charge=charge,basis="aug-cc-pvdz",cart=False,
        frozen=1,virtual_truncation=None,order=order,native_hf=float(mf.total_energy),
        native_corrections=[float(native.e2),float(native.e3)],reference_hf=ref.e_tot,
        reference_corrections=corrections.tolist(),fci_energy=fci_energy,
        fci_residual=residual,full_dimension=full_size,compressed_dimension=int(phi.size),
        sigma_seconds=sigma_seconds,elapsed_seconds=perf_counter()-start,
        high_order_backend="external PySCF Ag/D2h action + NumPy RS; no GradSCF AD")
    (folder/(symbol.lower()+"_reference.json")).write_text(json.dumps(report,indent=2)+"\n")
    print("%s FCI=%.12f, residual %.3e Ha, completed %.1f s" %
          (symbol,fci_energy,residual,report["elapsed_seconds"]),flush=True)
    del action,phi,gaps,diagonal,first,vectors
    jax.clear_caches()

# Measured CPU / float64 output, 2026-10-10:
#                   Ne                  F-
# Native HF       -128.496349730541     -99.428282441803
# Native E2         -0.206873508539      -0.237665735177
# Native E3         -0.001547443201       0.009273579005
# External FCI    -128.709475548753     -99.669368843138
# FCI residual       4.272e-9 Ha           5.176e-9 Ha
# |native - external| for E2/E3: <=2.53e-12 Ha.
# Ne E2--E25 vs Olsen Table III: maximum difference 1.009e-10 Ha.
# Ne minimum |E(n)| occurs at n=16; |E25|=5.8854813e-4 Ha.
# F- E15=+0.2757162 Ha; E25=+18.8938827 Ha (alternating divergence).
# Whole-example wall times: Ne 133.9 s, F- 142.7 s (not a paired speed test).
