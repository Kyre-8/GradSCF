"""Independent PySCF RHF/CIS and projector FD for the native benzene example.

Read only model definitions and invariant results from the native report.
SCF orbitals, AO integrals, excitation matrix and eigenvectors are calculated
independently. The exported raw basis definition fixes snapshot precision.
V=(S W+W S)/2 is reconstructed with PySCF's own overlap and AO atom owners.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import platform
import resource
from time import perf_counter

import numpy as np


def run(native_path=Path("reproducibility/ad_showcase/A03_degenerate_vjp/benzene_gradscf.json"), *,
        output=Path("reproducibility/ad_showcase/A03_degenerate_vjp/benzene_pyscf.json"), prepare_only=False):
    """Run this example with ordinary Python parameters; no command-line parser."""
    args = SimpleNamespace(native=Path(native_path), output=Path(output), prepare_only=prepare_only)
    native_bytes = args.native.read_bytes()
    native = json.loads(native_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(complete=False, passed=False, stage="starting", backend="independent PySCF RHF/CIS",
        native_sha256=hashlib.sha256(native_bytes).hexdigest(),
        native_source_python_sha256=native.get("source_python_sha256"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hostname=platform.node(), process_id=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), numpy=np.__version__, dtype="float64",
        units=native.get("units"), scope="Independent fixed-reference SCF/operator/projector FD; no shared MOs or AO integrals.",
        threads={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    started = perf_counter()
    def save():
        divisor = 1024**2 if platform.system() == "Darwin" else 1024
        report.update(elapsed_seconds=perf_counter()-started,
                      process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/divisor)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=True)+"\n")
    save()
    try:
        compare(args, native, report, save)
    except Exception as error:
        report.update(complete=False, passed=False, error=f"{type(error).__name__}: {error}")
        save()
        raise
    print(f"Saved {args.output}: stage={report['stage']}, complete={report['complete']}, passed={report['passed']}", flush=True)


def compare(args, native, report, save):
    import pyscf
    from pyscf import gto, scf, tdscf
    from scipy.linalg import eigh

    if not native.get("preparation_complete") or native.get("integral_backend") != "native":
        raise RuntimeError("Require a successfully prepared native GradSCF report")
    if not args.prepare_only and (not native.get("complete") or not native.get("passed")):
        raise RuntimeError("Full independent response comparison requires a complete passing native report")
    mol = gto.M(atom=native["geometry_angstrom"], basis=native["basis_definition"],
                unit="Angstrom", cart=True, charge=0, spin=0, symmetry=False, verbose=4)
    report.update(pyscf=pyscf.__version__, basis_source="native raw basis definition only; independent integrals/SCF",
                  stage="independent RHF")
    save()
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 200
    mf.kernel()
    report["scf"] = dict(converged=bool(mf.converged), energy_hartree=float(mf.e_tot),
                         cycles=getattr(mf, "cycles", None), stability_test_performed=False)
    save()
    if not mf.converged:
        raise RuntimeError("Independent PySCF RHF did not converge")
    coeff, occupations, overlap = np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ), mf.get_ovlp()
    nocc, nao = int(np.count_nonzero(occupations == 2)), mol.nao_nr()
    co, cv = coeff[:, :nocc], coeff[:, nocc:]
    dim = nocc*cv.shape[1]
    owners, order = [], []
    for shell in range(mol.nbas):
        atom, angular = mol.bas_atom(shell), mol.bas_angular(shell)
        for contraction in range(mol.bas_nctr(shell)):
            for lx in range(angular, -1, -1):
                for ly in range(angular-lx, -1, -1):
                    owners.append(atom)
                    order.append([atom, shell, angular, contraction, lx, ly, angular-lx-ly])
    counts = np.bincount(owners, minlength=mol.natm)
    if nao != native["nao"] or dim != native["dimension"] or counts.tolist() != native["ao_counts"]:
        raise AssertionError("Independent AO ownership/counts or singles dimension differ")
    coordinates = mol.atom_coords()
    phi = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    atom_weights = np.stack((np.cos(2*phi), np.sin(2*phi)))
    if not np.allclose(atom_weights, native["atom_weights"], atol=1e-14, rtol=0):
        raise AssertionError("The independently defined atomic source weights differ")
    w = atom_weights[:, owners]
    potentials = .5*overlap[None]*(w[:, :, None]+w[:, None, :])
    with mol.with_common_orig((0., 0., 0.)):
        dipole = mol.intor_symmetric("int1e_r", comp=3)
    probes = -np.sqrt(2.)*np.einsum("pi,xpq,qa->iax", co, dipole, cv).reshape(dim, 3)
    report.update(nao=nao, nocc=nocc, nvir=cv.shape[1], dimension=dim, ao_owners=owners, ao_order=order,
        ao_counts=counts.tolist(), ao_order_matches_native=order == native["ao_order"],
        ao_owner_sequence_matches_native=owners == native["ao_owners"],
        order_note="Any within-atom AO permutation leaves the atomic scalar W unchanged; only invariants are compared.",
        atom_weights=atom_weights.tolist(), probe_gram=(probes.T@probes).tolist(), stage="independent TDA matrix")
    save()
    tda = tdscf.TDA(mf)
    tda.singlet, tda.frozen = True, None
    a, b = tda.get_ab()
    matrix = np.asarray(a).reshape(dim, dim)
    del b
    report["oracle_hermiticity_error"] = float(np.max(np.abs(matrix-matrix.T)))
    if report["oracle_hermiticity_error"] > 1e-10:
        raise AssertionError("PySCF TDA matrix is not symmetric")
    count, chosen = min(32, dim), None
    while chosen is None:
        energies, vectors = eigh(matrix, subset_by_index=(0, count-1), driver="evr")
        boundaries = np.r_[0, np.flatnonzero(np.diff(energies) > native["settings"]["cluster_tol"])+1, count]
        groups = []
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            transition = vectors[:, lo:hi].T@probes
            tensor = transition.T@transition
            row = dict(start=int(lo), stop=int(hi), energies_hartree=energies[lo:hi].tolist(),
                tensor=tensor.tolist(), splitting_hartree=float(np.ptp(energies[lo:hi])),
                lower_gap_hartree=float(energies[lo]-energies[lo-1]) if lo else None,
                upper_gap_hartree=float(energies[hi]-energies[hi-1]) if hi < count else None)
            groups.append(row)
            if chosen is None and row["upper_gap_hartree"] is not None and np.trace(tensor[:2, :2]) > 1e-8:
                if hi-lo != 2 or np.min(np.linalg.eigvalsh(tensor[:2, :2])) <= 1e-8:
                    raise RuntimeError("Independent lowest in-plane bright cluster is not a doublet")
                chosen = row
        report.update(low_roots_hartree=energies.tolist(), low_root_clusters=groups)
        save()
        if chosen is None:
            if count == dim:
                raise RuntimeError("Independent bright doublet not identified")
            count = min(2*count, dim)
    lo, hi = chosen["start"], chosen["stop"]
    tensor = np.asarray(chosen["tensor"])
    native_tensor = np.asarray(native.get("tensor", native["cluster"]["tensor"]))
    errors = dict(scf_energy_hartree=abs(mf.e_tot-native["scf"]["energy_hartree"]),
        cluster_energy_hartree=float(np.max(np.abs(np.asarray(chosen["energies_hartree"])-native["cluster"]["energies_hartree"]))),
        tensor_max=float(np.max(np.abs(tensor-native_tensor))))
    report.update(cluster=chosen, tensor=tensor.tolist(), errors=errors)
    save()
    if (errors["scf_energy_hartree"] > 2e-7 or errors["cluster_energy_hartree"] > 2e-7
            or not np.allclose(tensor, native_tensor, atol=2e-6, rtol=2e-6)):
        raise AssertionError("Independent SCF, excitation energies or invariant tensor disagree")
    if args.prepare_only:
        report.update(stage="prepared_comparison", preparation_comparison_passed=True,
                      validation_scope="Independent SCF and cluster invariants only; VJP not checked")
        save()
        return
    vv, oo = np.einsum("pa,xpq,qb->xab", cv, potentials, cv), np.einsum("pi,xpq,qj->xij", co, potentials, co)
    perturbations = [np.kron(np.eye(nocc), v)-np.kron(o, np.eye(cv.shape[1])) for v, o in zip(vv, oo)]
    bound = max(np.linalg.norm(v, 2)+np.linalg.norm(o, 2) for v, o in zip(vv, oo))
    gap = min(value for value in (chosen["lower_gap_hartree"], chosen["upper_gap_hartree"]) if value is not None)
    weight = native["settings"]["xy_weight"]
    cotangent = np.r_[np.array([[1., weight/2, 0.], [weight/2, -1., 0.], [0., 0., 0.]]).ravel(), 0.]
    report.update(stage="independent projector finite differences", finite_differences=[], perturbation_norm_bound=bound)
    for step in min(1e-3, gap/(20*bound))*np.array([1., .3, .1]):
        derivatives = []
        for perturbation in perturbations:
            endpoints = []
            for sign in (-1., 1.):
                energies, vectors = eigh(matrix+sign*step*perturbation, subset_by_index=(0, hi), driver="evr")
                gaps = [energies[hi]-energies[hi-1]]+([energies[lo]-energies[lo-1]] if lo else [])
                if min(gaps) < gap/2:
                    raise AssertionError("Independent FD endpoint lost cluster isolation")
                projected = vectors[:, lo:hi].T@probes
                endpoints.append(np.r_[(projected.T@projected).ravel(), energies[lo:hi].sum()])
            derivatives.append((endpoints[1]-endpoints[0])/(2*step))
        jacobian = np.asarray(derivatives).T
        fd = cotangent@jacobian
        passed = bool(np.allclose(fd, native["vjp"], atol=1e-5, rtol=2e-4)
                      and np.allclose(jacobian@np.asarray(native["direction"]), native["jvp"], atol=1e-5, rtol=2e-4))
        report["finite_differences"].append(dict(step_hartree=float(step), vjp=fd.tolist(), passed=passed,
            max_vjp_error=float(np.max(np.abs(fd-native["vjp"])))))
        save()
    if not all(row["passed"] for row in report["finite_differences"][-2:]):
        raise AssertionError("Two smallest independent FD steps do not agree with native AD")
    report.update(complete=True, passed=True, stage="complete",
                  validation_scope="Independent PySCF SCF/cluster tensor and three-step projector FD against native VJP/JVP")
    save()


if __name__ == "__main__":
    run()
