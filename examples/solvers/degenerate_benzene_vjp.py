"""Native RHF/CIS projector VJP for regular D6h benzene (remote CPU float64).

GradSCF supplies SCF, integrals, low roots, and shared subspace AD. PySCF is
used only by compare_degenerate_benzene_vjp.py, with an independent SCF.
The fixed-reference source is V_a=(S W_a+W_a S)/2, with atomic AO weights
cos(2 phi), sin(2 phi). This Mulliken-style one-body model is not a quadrupole
or a local real-space potential. Orbitals, integrals and probes stay fixed;
only the Fock matrix changes. No relaxed HF or geometry derivative is claimed.
"""
from __future__ import annotations

from dataclasses import replace
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


def peak_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system() == "Darwin" else 1024)


def source_hash():
    root, digest = Path(__file__).resolve().parents[2], hashlib.sha256()
    for path in sorted((root/"src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def identify_cluster(energies, vectors, probes, tolerance):
    bounds = np.r_[0, np.flatnonzero(np.diff(energies) > tolerance)+1, len(energies)]
    groups, selected = [], None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        projected = vectors[:, lo:hi].T@probes
        tensor = projected.T@projected
        row = dict(start=int(lo), stop=int(hi), energies_hartree=energies[lo:hi].tolist(),
            splitting_hartree=float(np.ptp(energies[lo:hi])), tensor=tensor.tolist(),
            lower_gap_hartree=float(energies[lo]-energies[lo-1]) if lo else None,
            upper_gap_hartree=float(energies[hi]-energies[hi-1]) if hi < len(energies) else None)
        groups.append(row)
        if selected is None and row["upper_gap_hartree"] is not None and np.trace(tensor[:2, :2]) > 1e-8:
            if hi-lo != 2 or np.min(np.linalg.eigvalsh(tensor[:2, :2])) <= 1e-8:
                raise RuntimeError("Lowest bright in-plane cluster is not a complete doublet")
            anisotropy = np.max(np.abs(tensor[:2, :2]-np.eye(2)*np.trace(tensor[:2, :2])/2))
            if anisotropy > 1e-6*max(1., np.trace(tensor)) or abs(tensor[2, 2]) > 1e-8*max(1., np.trace(tensor)):
                raise RuntimeError("Bright doublet fails in-plane isotropy")
            selected = row
    return selected, groups


def _compute(args, report, save):
    import jax
    import jax.numpy as jnp
    from gradscf import gto, integrals, scf
    from gradscf.integrals.basis import cartesian_angular_tuples
    from gradscf.tddft._semilocal_response import SemilocalResponseFunctional
    from gradscf.tddft.response import build_restricted_tda_operator
    from gradscf.solvers import EigenResponseConfig, EigenSolverConfig, LinearOperator, LinearSolverConfig, solve_hermitian

    jax.config.update("jax_enable_x64", True)
    angles = np.arange(6)*np.pi/3
    atoms = [(symbol, (radius*np.cos(angle), radius*np.sin(angle), 0.))
             for symbol, radius in (("C", 1.396792), ("H", 2.484212)) for angle in angles]
    report.update(stage="native RHF", geometry_angstrom=atoms, jax=jax.__version__,
                  devices=[str(device) for device in jax.devices()])
    save()
    mol = gto.M(atom=atoms, unit="Angstrom", basis=args.basis, cart=True)
    started = perf_counter()
    mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-12, conv_tol_density=1e-11,
                 conv_tol_grad=1e-10, max_cycle=200, execution_device="cpu").run()
    report["scf"] = dict(converged=bool(mf.converged), energy_hartree=float(mf.e_tot),
        cycles=int(mf.cycles), seconds=perf_counter()-started, stability_test_performed=False)
    save()
    if not mf.converged:
        raise RuntimeError("Native RHF did not converge")
    base = replace(mf.to_reference(), fock_matrix=mf.scf_result.fock_matrix)
    coeff, overlap = np.asarray(mf.mo_coeff), np.asarray(base.overlap_matrix)
    nocc, nao = int(base.nocc), coeff.shape[0]
    nvir, dim = coeff.shape[1]-nocc, nocc*(coeff.shape[1]-nocc)
    topology, parameters = integrals.prepare_basis(mol.to_spec(), mol.basis, cart=True)
    coordinates = np.asarray(parameters.nuclear_coords)
    distances = np.linalg.norm(np.asarray(parameters.centers)[:, None]-coordinates[None], axis=-1)
    shell_owners = np.argmin(distances, axis=1)
    if np.max(distances[np.arange(len(shell_owners)), shell_owners]) > 1e-12:
        raise AssertionError("A native basis shell is not centered on its recorded atom")
    owners, order, basis_definition = [], [], {"C": [], "H": []}
    for shell, (owner, angular, nctr) in enumerate(zip(shell_owners, topology.angular_momenta, topology.contraction_counts)):
        for contraction in range(nctr):
            for powers in cartesian_angular_tuples(angular):
                owners.append(int(owner))
                order.append([int(owner), shell, angular, contraction, *powers])
        if owner in (0, 6):
            rows = np.column_stack((np.asarray(parameters.exponents[shell]), np.asarray(parameters.coefficients[shell])))
            basis_definition[atoms[owner][0]].append([angular, *rows.tolist()])
    if len(owners) != nao:
        raise AssertionError("Native topology AO order does not match the SCF AO dimension")
    if args.basis.lower().replace(" ", "") in ("6-31g*", "6-31g(d)") and dim != 1701:
        raise AssertionError("Cartesian all-orbital 6-31G* benzene must have 1701 singles")
    atom_weights = np.stack((np.cos(2*np.tile(angles, 2)), np.sin(2*np.tile(angles, 2))))
    weights_ao = atom_weights[:, owners]
    potential_np = .5*overlap[None]*(weights_ao[:, :, None]+weights_ao[:, None, :])
    co, cv = coeff[:, :nocc], coeff[:, nocc:]
    probes_np = -np.sqrt(2.)*np.einsum("pi,xpq,qa->iax", co, np.asarray(base.dipole_integrals), cv).reshape(dim, 3)
    potentials, probes = jnp.asarray(potential_np), jnp.asarray(probes_np)
    report.update(nao=nao, nocc=nocc, nvir=nvir, dimension=dim, ao_owners=owners, ao_order=order,
        basis_definition=basis_definition, atom_weights=atom_weights.tolist(), probe_shape=list(probes_np.shape),
        probe_gram=(probes_np.T@probes_np).tolist(), ao_counts=np.bincount(owners, minlength=12).tolist(),
        scf_ov_fock_max=float(np.max(np.abs(co.T@np.asarray(base.fock_matrix)@cv))),
        stage="native low-root discovery")
    save()
    functional = SemilocalResponseFunctional("hf")
    policy = EigenResponseConfig(target="subspace", linear_config=LinearSolverConfig(
        method="gmres", rtol=1e-10, restart=40, maxiter=args.maxiter))

    def operator(theta, reference):
        current = replace(reference, fock_matrix=reference.fock_matrix+jnp.einsum("a,apq->pq", theta, potentials))
        action, diagonal, _ = build_restricted_tda_operator(current, functional,
            response_kernel_options={"two_electron_mode": "direct"})
        return LinearOperator((dim, dim), diagonal.dtype, lambda v: action(v[None, :])[0],
                              diagonal=diagonal, matmat=lambda v: action(v.T).T)

    def config(count):
        return EigenSolverConfig(method=args.method, nroots=count, atol=1e-10, maxiter=args.maxiter,
            max_subspace=min(args.max_subspace, dim), max_dense=dim, seed=11)

    capacity = min(args.max_subspace, dim)
    discovery_limit = dim-1 if capacity == dim else capacity-3
    zero, count, chosen = jnp.zeros(2), min(12, discovery_limit), None
    while chosen is None:
        cfg = config(count)
        state = jax.jit(lambda reference: solve_hermitian(operator(zero, reference), config=cfg))(base)
        present = np.asarray(state.raw_present)
        energies, vectors = np.asarray(state.raw_values)[present], np.asarray(state.raw_vectors)[:, present]
        residuals = np.asarray(state.raw_residual_norms)[present]
        report.update(discovery_ranks=count, low_roots_hartree=energies.tolist(), low_root_residuals=residuals.tolist())
        save()
        if not np.all(np.asarray(state.converged)) or np.max(residuals) > 1e-9:
            raise RuntimeError("Native low-root discovery or its guard did not converge")
        chosen, groups = identify_cluster(energies, vectors, probes_np, args.cluster_tol)
        report["low_root_clusters"] = groups
        if chosen is None:
            limit = discovery_limit
            if count == limit:
                raise RuntimeError("No complete bright doublet found; increase --max-subspace")
            count = min(2*count, limit)
        save()
    lo, hi = chosen["start"], chosen["stop"]
    if min(dim, hi+3) > capacity:
        raise ValueError("Cumulative cluster, guard and expansion slots exceed --max-subspace")
    gap = min(value for value in (chosen["lower_gap_hartree"], chosen["upper_gap_hartree"]) if value is not None)
    report.update(cluster=chosen, exterior_gap_hartree=gap, preparation_complete=True, stage="prepared")
    save()
    if args.prepare_only:
        report.update(stage="prepared_only", requested_stage_complete=True,
                      validation_scope="Native RHF and low-root discovery only; projector VJP not run")
        save()
        return
    configs = [config(rank) for rank in ((hi, lo) if lo else (hi,))]
    loss_weights = jnp.array([[1., args.xy_weight/2, 0.], [args.xy_weight/2, -1., 0.], [0., 0., 0.]])

    def solve(theta, reference):
        op = operator(theta, reference)
        return tuple(solve_hermitian(op, config=cfg, response=policy, probes=probes) for cfg in configs)

    def forward(theta, reference):
        states = solve(theta, reference)
        projected, trace = states[0].projection, states[0].eigenvalue_sum
        if lo:
            projected, trace = projected-states[1].projection, trace-states[1].eigenvalue_sum
        return jnp.concatenate(((probes.T@projected).reshape(-1), trace[None]))

    solve_jit, forward_jit = jax.jit(solve), jax.jit(forward)
    report.update(stage="native subspace VJP", cumulative_ranks=[cfg.nroots for cfg in configs])
    save()
    started = perf_counter()
    states = solve_jit(zero, base)
    jax.block_until_ready(states)
    report["forward_seconds"] = perf_counter()-started
    report["forward_states"] = [dict(rank=cfg.nroots, converged=bool(state.converged),
        response_valid=bool(state.response_valid), status=int(state.status), boundary_gap=float(state.boundary_gap),
        ritz_values=np.asarray(state.raw_values).tolist(), ritz_residuals=np.asarray(state.raw_residual_norms).tolist())
        for cfg, state in zip(configs, states)]
    save()
    if not all(bool(state.converged) and bool(state.response_valid) for state in states):
        raise RuntimeError("A cumulative subspace or guard failed the shared solver gates")
    started = perf_counter()
    values, pullback = jax.vjp(lambda theta: forward_jit(theta, base), zero)
    jax.block_until_ready(values)
    report["vjp_preparation_seconds"] = perf_counter()-started
    cotangent = jnp.concatenate((loss_weights.reshape(-1), jnp.zeros(1)))
    started = perf_counter()
    gradient = np.asarray(pullback(cotangent)[0])
    report["retained_pullback_first_seconds"] = perf_counter()-started
    direction = jnp.array([.6, .8])
    _, tangent = jax.jvp(lambda theta: forward_jit(theta, base), (zero,), (direction,))
    values, tangent = np.asarray(values), np.asarray(tangent)
    duality = abs(float(tangent@np.asarray(cotangent))-float(gradient@np.asarray(direction)))
    report.update(tensor=values[:9].reshape(3, 3).tolist(), trace_hartree=float(values[9]),
        loss=float(values@np.asarray(cotangent)), vjp=gradient.tolist(), direction=np.asarray(direction).tolist(),
        jvp=tangent.tolist(), jvp_vjp_error=duality, nonzero_vjp=bool(np.linalg.norm(gradient) > 1e-8))
    save()
    if (not np.allclose(values[:9], np.asarray(chosen["tensor"]).ravel(), atol=2e-7, rtol=2e-7)
            or not np.isfinite(gradient).all() or not report["nonzero_vjp"]
            or duality > 1e-7*max(1., np.linalg.norm(gradient))):
        raise AssertionError("Native projector agreement, nonzero VJP, or JVP-VJP gate failed")
    report["finite_differences"] = dict(performed=False, independent=False, reason="--skip-fd", steps=[])
    if not args.skip_fd:
        vv, oo = np.einsum("pa,xpq,qb->xab", cv, potential_np, cv), np.einsum("pi,xpq,qj->xij", co, potential_np, co)
        bound = max(np.linalg.norm(v, 2)+np.linalg.norm(o, 2) for v, o in zip(vv, oo))
        report["finite_differences"].update(performed=True, reason="native same-operator FD", norm_bound=bound)
        for step in min(1e-3, gap/(20*bound))*np.array([1., .3, .1]):
            derivatives = []
            for axis in np.eye(2):
                endpoints = []
                for sign in (-1., 1.):
                    theta = jnp.asarray(sign*step*axis)
                    shifted = solve_jit(theta, base)
                    if not all(bool(state.converged) and bool(state.response_valid) for state in shifted):
                        raise RuntimeError("A native FD endpoint failed convergence/isolation")
                    endpoints.append(np.asarray(forward_jit(theta, base)))
                derivatives.append((endpoints[1]-endpoints[0])/(2*step))
            jacobian = np.asarray(derivatives).T
            fd = np.asarray(cotangent)@jacobian
            passed = bool(np.allclose(fd, gradient, atol=1e-5, rtol=2e-4)
                          and np.allclose(jacobian@np.asarray(direction), tangent, atol=1e-5, rtol=2e-4))
            report["finite_differences"]["steps"].append(dict(step_hartree=float(step), vjp=fd.tolist(),
                max_error=float(np.max(np.abs(fd-gradient))), passed=passed))
            save()
        if not all(row["passed"] for row in report["finite_differences"]["steps"][-2:]):
            raise AssertionError("The two smallest native FD steps failed")
    report.update(complete=True, passed=True, stage="complete",
        validation_scope="Native fixed-reference projector AD; PySCF comparison is a separate report",
        independent_comparison_performed=False)
    save()


def run(basis="6-31g*", method="davidson", *, max_subspace=80, maxiter=200, cluster_tol=1e-7,
        xy_weight=.5, skip_fd=False, prepare_only=False,
        output=Path("reproducibility/ad_showcase/A03_degenerate_vjp/benzene_gradscf.json")):
    """Run this example with ordinary Python parameters; no command-line parser."""
    if method not in ("dense", "davidson"):
        raise ValueError("method must be dense or davidson")
    args = SimpleNamespace(basis=basis, method=method, max_subspace=max_subspace, maxiter=maxiter,
        cluster_tol=cluster_tol, xy_weight=xy_weight, skip_fd=skip_fd,
        prepare_only=prepare_only, output=Path(output))
    if args.cluster_tol <= 0 or args.maxiter < 1 or args.max_subspace < 5:
        raise ValueError("Require positive cluster tolerance/maxiter and max_subspace >= 5")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(complete=False, passed=False, preparation_complete=False, method=args.method,
        basis=args.basis, cart=True, frozen=None, integral_backend="native",
        settings={key: value for key, value in vars(args).items() if key != "output"},
        source_python_sha256=source_hash(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hostname=platform.node(), process_id=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), numpy=np.__version__, dtype="float64", stage="starting",
        threads={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        scope="Native fixed-reference RHF/CIS; only AO Fock changes; no HF relaxation or geometry AD.",
        source_definition="V_a=(S W_a+W_a S)/2; W_a constant on each atom's AOs: cos(2phi), sin(2phi). Mulliken-style AO model.",
        units=dict(energy="Hartree", dipole="e bohr", tensor="e^2 bohr^2", theta="Hartree", vjp="e^2 bohr^2/Hartree"),
        loss="Txx-Tyy+xy_weight*Txy", adjoint_residual=None, adjoint_iterations=None, operator_calls=None,
        adjoint_diagnostics="Shared true-residual guard is active; actual residual and counts are not exposed.",
        operator_storage="Existing TDA matrix-free builder; dense method may materialize inside the shared eigensolver.",
        gauge_check_performed=False, memory_scope="Whole-process high-water RSS, including native SCF and AO ERIs.")
    started = perf_counter()
    def save():
        report.update(elapsed_seconds=perf_counter()-started, process_peak_rss_mib=peak_rss_mib())
        args.output.write_text(json.dumps(report, indent=2, allow_nan=True)+"\n")
    save()
    try:
        _compute(args, report, save)
    except Exception as error:
        report.update(complete=False, passed=False, error=f"{type(error).__name__}: {error}")
        save()
        raise
    print(f"Saved {args.output}: stage={report['stage']}, complete={report['complete']}, passed={report['passed']}", flush=True)


if __name__ == "__main__":
    run()
