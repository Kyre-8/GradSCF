"""X09: full-FCI excited-state absorption and bond-length response of H2.

All four 6-31G spatial orbitals and both electrons are active. A moving
Cholesky AO frame avoids an unnecessary HF calculation. The FCI Hamiltonian,
eigenpair response and transition RDMs use the existing shared implementations.
Upward absorption and downward stimulated emission retain their actual weights.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import fci, gto, integrals, scf
from gradscf.ao2mo import transform_integrals
from gradscf.solvers import EigenResponseConfig, EigenSolverConfig, LinearSolverConfig
from gradscf_tools.molecular_ep import NativeRHF


BOND = 1.4  # Bohr; atoms move by +/- half the bond-length change.
BASIS = "6-31g"
INITIAL_ROOT = 2  # Forward inventory: lowest bright excited singlet, ungerade.
TARGET_ROOTS = (4, 5)  # Isolated singlets with nonzero upward strengths.
OUTPUT = Path("reproducibility/ad_showcase/X09_h2_fci_esa")


def coordinates(length):
    return length*jnp.array([[0., 0., -.5], [0., 0., .5]])


def make_problem():
    mol = gto.M(atom=list(zip(["H", "H"], np.asarray(coordinates(BOND)))),
                basis=BASIS, unit="Bohr", cart=True)
    topology, parameters = integrals.prepare_basis(mol.to_spec(), mol.basis)
    plan = integrals.make_plan(topology, backend="native")
    owners = np.argmin(np.linalg.norm(np.asarray(parameters.centers)[:, None]
                                     - np.asarray(parameters.nuclear_coords)[None], axis=-1), axis=1)
    space = fci.make_fci_space(topology.nao, (1, 1))
    assert topology.nao == 4 and space.size == 16
    solver = EigenSolverConfig(method="dense", nroots=space.size, max_dense=32, atol=1e-12)
    response = EigenResponseConfig(target="eigenpairs", linear_config=LinearSolverConfig(
        method="direct", rtol=1e-10, atol=1e-12, max_dense=32))
    charges = jnp.asarray(topology.nuclear_charges)
    initial = jnp.array([0, INITIAL_ROOT])

    def evaluate(length, rotation):
        coords = coordinates(length)
        moving = replace(parameters, centers=coords[owners], nuclear_coords=coords)
        overlap = plan.evaluate("overlap", moving)
        hcore = plan.evaluate("kinetic", moving) + plan.evaluate("nuclear", moving)
        eri = plan.evaluate("eri", moving)
        position = plan.evaluate("dipole", moving, origin=jnp.zeros(3))
        # X.T S X = I. Both the metric and this frame participate in AD.
        frame = jax.scipy.linalg.solve_triangular(jnp.linalg.cholesky(overlap).T,
                                                 jnp.eye(space.norb), lower=False) @ rotation
        h, g = transform_integrals(hcore, frame, eri=eri)
        position = jnp.einsum("xpq,pi,qj->xij", position, frame, frame)
        nuclear = scf.nuclear_repulsion_energy(coords, charges)
        state = fci.solve_fci(h, g, space, ecore=nuclear, config=solver, response=response)
        ci, energies = state.coefficients, state.total_energies
        densities = jax.vmap(lambda left: jax.vmap(lambda right:
            fci.trans_rdm1(left, right, space))(ci))(ci[initial])
        moments = -jnp.einsum("xpq,ijqp->ijx", position, densities)
        moments += jnp.einsum("x,ij->ijx", charges @ coords,
                              jnp.einsum("iab,jab->ij", ci[initial], ci))
        gaps = energies[None, :] - energies[initial, None]
        strengths = (2/3)*gaps*jnp.sum(moments**2, axis=-1)
        return dict(energies=energies, coefficients=ci, frame=frame, overlap=overlap,
                    position=position, h1=h, eri=g, nuclear=nuclear,
                    transition_densities=densities, moments=moments, signed_f=strengths,
                    residual_norms=state.residual_norms, response_valid=state.response_valid,
                    converged=state.converged)

    return mol, space, evaluate


def inspect(point, space):
    energies = np.asarray(point["energies"])
    differences = np.abs(energies[:, None]-energies[None, :])
    np.fill_diagonal(differences, np.inf)
    spin = np.asarray(jax.vmap(lambda c: fci.spin_square(c, space)[0])(point["coefficients"]))
    # Inversion interchanges the two identical atoms' two s basis functions.
    permutation = np.eye(4)[[2, 3, 0, 1]]
    frame = np.asarray(point["frame"])
    inversion = frame.T @ np.asarray(point["overlap"]) @ permutation @ frame
    parity = np.array([np.sum(c*(inversion @ c @ inversion.T))
                       for c in np.asarray(point["coefficients"])])
    assert np.all(point["converged"]) and np.all(point["response_valid"])
    assert differences.min() > 1e-4  # The actual minimum is checked, not assumed.
    np.testing.assert_allclose(spin[[0, INITIAL_ROOT, *TARGET_ROOTS]], 0., atol=1e-9)
    np.testing.assert_allclose(np.abs(parity), 1., atol=1e-9)
    return spin, parity, differences.min(axis=1)


def run():
    start = perf_counter()
    mol, space, evaluate = make_problem()
    calculate = jax.jit(evaluate)
    identity = jnp.eye(space.norb)
    point = calculate(BOND, identity)
    spin, parity, gaps = inspect(point, space)
    derivative = np.asarray(jax.jit(lambda length, rotation: jax.jvp(
        lambda r: evaluate(r, rotation)["signed_f"], (length,), (jnp.ones_like(length),))[1])(
            jnp.asarray(BOND), identity))
    assert np.isfinite(derivative).all()
    assert np.min(np.abs(derivative[1, list(TARGET_ROOTS)])) > 1e-4

    # Reuse only its native cross-overlap operation. No HF solve is called.
    overlap_engine = NativeRHF(list(zip(["H", "H"], np.asarray(coordinates(BOND)))),
                               basis=BASIS, unit="Bohr")
    checks, tracking = [], []
    for step in (1e-3, 3e-4, 1e-4):
        displaced = []
        for sign in (1, -1):
            length = BOND + sign*step
            current = calculate(length, identity)
            current_spin, _, current_gaps = inspect(current, space)
            cross = np.asarray(overlap_engine.cross_overlap(coordinates(BOND), coordinates(length)))
            frame_overlap = np.asarray(point["frame"]).T @ cross @ np.asarray(current["frame"])
            # One alpha and one beta electron: each determinant overlap is a product.
            state_overlap = np.einsum("iab,ac,bd,jcd->ij", np.asarray(point["coefficients"]),
                                     frame_overlap, frame_overlap, np.asarray(current["coefficients"]))
            assignment = np.argmax(np.abs(state_overlap), axis=1)
            np.testing.assert_array_equal(assignment, np.arange(space.size))
            minimum_overlap = float(np.min(np.abs(np.diag(state_overlap))))
            assert minimum_overlap > .999
            np.testing.assert_allclose(current_spin, spin, atol=1e-9)
            tracking.append(dict(bond_bohr=length, root_assignment=assignment.tolist(),
                                 minimum_wavefunction_overlap=minimum_overlap,
                                 minimum_gap_hartree=float(current_gaps.min())))
            displaced.append(np.asarray(current["signed_f"]))
        numerical = (displaced[0]-displaced[1])/(2*step)
        error = float(np.max(np.abs(numerical-derivative)))
        checks.append(dict(step_bohr=step, signed_f_derivative_per_bohr=numerical.tolist(),
                           max_derivative_error=error))
        print("bond step", step, "max df/dR error", error, flush=True)
    np.testing.assert_allclose(numerical, derivative, atol=3e-6, rtol=1e-5)

    # Full-CI frame invariance: rotate all orbitals, without freezing any state response.
    rotation = np.linalg.qr(np.random.default_rng(83).normal(size=(4, 4)))[0]
    rotated = calculate(BOND, jnp.asarray(rotation))
    rotated_derivative = np.asarray(jax.jit(lambda length: jax.jvp(
        lambda r: evaluate(r, jnp.asarray(rotation))["signed_f"],
        (length,), (jnp.ones_like(length),))[1])(jnp.asarray(BOND)))
    np.testing.assert_allclose(rotated["energies"], point["energies"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(rotated["signed_f"], point["signed_f"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(rotated_derivative, derivative, atol=1e-8, rtol=0)

    # Finite-Hilbert-space closure. The finite AO basis does not imply TRK = Ne.
    eye = jnp.eye(space.size)
    hamiltonian = fci.build_hamiltonian(point["h1"], point["eri"], space).apply(eye)
    hamiltonian = np.asarray(hamiltonian + point["nuclear"]*eye)
    commutator = np.zeros_like(hamiltonian)
    for position in point["position"]:
        dipole = np.asarray(jax.vmap(lambda column: fci.contract_1e(
            -position, column.reshape(space.shape), space).ravel(), in_axes=1, out_axes=1)(eye))
        commutator += 2*dipole@hamiltonian@dipole - dipole@dipole@hamiltonian - hamiltonian@dipole@dipole
    closure = []
    for row, initial in enumerate((0, INITIAL_ROOT)):
        ci = np.asarray(point["coefficients"][initial]).ravel()
        rhs = float(ci @ commutator @ ci / 3)
        total = float(np.sum(np.asarray(point["signed_f"])[row]))
        np.testing.assert_allclose(total, rhs, atol=1e-10, rtol=0)
        forbidden = (np.abs(spin) > 1e-8) | (parity[initial]*parity > 0)
        assert np.max(np.abs(np.asarray(point["signed_f"])[row, forbidden])) < 1e-12
        closure.append(dict(initial_root=initial, signed_sum=total, commutator_value=rhs))

    sources = ("src/gradscf/fci/solver.py", "src/gradscf/fci/rdm.py",
               "src/gradscf/solvers/eigen/response.py", "src/gradscf_tools/molecular_ep.py",
               "src/gradscf/integrals/backends/native/ffi.py")
    data = dict(method="full FCI", symbols=["H", "H"], bond_bohr=BOND,
        coordinates_bohr=np.asarray(coordinates(BOND)).tolist(), basis=BASIS, cart=True,
        electrons=[1, 1], active_orbitals=list(range(4)), determinant_count=space.size,
        initial_roots=[0, INITIAL_ROOT], derivative_targets=list(TARGET_ROOTS),
        energies_hartree=np.asarray(point["energies"]).tolist(), spin_square=spin.tolist(), parity=parity.tolist(),
        nearest_gaps_hartree=gaps.tolist(), ci_coefficients=np.asarray(point["coefficients"]).tolist(),
        orbital_frame=np.asarray(point["frame"]).tolist(), transition_rdm1=np.asarray(point["transition_densities"]).tolist(),
        transition_dipoles_au=np.asarray(point["moments"]).tolist(), signed_oscillator_strengths=np.asarray(point["signed_f"]).tolist(),
        signed_f_derivative_per_bohr=derivative.tolist(), finite_difference_checks=checks, state_tracking=tracking,
        max_eigen_residual=float(np.max(np.asarray(point["residual_norms"]))), finite_hilbert_closure=closure,
        frame_invariance_strength_error=float(np.max(np.abs(rotated["signed_f"]-point["signed_f"]))),
        frame_invariance_derivative_error=float(np.max(np.abs(rotated_derivative-derivative))),
        frame="moving AO Cholesky; all orbitals active; no HF solve",
        derivative="complete native integral + overlap/Cholesky + shared eigenpair response",
        eigen_tolerance=1e-12, response_rtol=1e-10, response_atol=1e-12, gap_atol=1e-8, gap_rtol=1e-8,
        hostname=platform.node(), python_version=platform.python_version(), jax_version=jax.__version__,
        numpy_version=np.__version__, dtype="float64", devices=[str(x) for x in jax.devices()],
        source_hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        elapsed_seconds=perf_counter()-start)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/"gradscf.json").write_text(json.dumps(data, indent=2)+"\n")
    print("selected ESA f", np.asarray(point["signed_f"])[1, list(TARGET_ROOTS)], flush=True)
    print("selected ESA df/dR", derivative[1, list(TARGET_ROOTS)], flush=True)
    print("finite-space signed sums", closure, flush=True)
    return data


if __name__ == "__main__":
    run()

# Measured on jy_gpu_1/c20, CPU float64, 2026-10-16:
# All 16 roots isolated; minimum gap 0.0449523181 Eh; max eigen residual 2.009e-15.
# ESA f(2->4,2->5) = [0.193430408197, 0.809989157124].
# df/dR / Bohr^-1 = [-0.204756783205, 0.360304876963].
# Max AD-FD errors at h=1e-3/3e-4/1e-4: 3.830e-6 / 3.446e-7 / 3.837e-8.
# Smallest physical wavefunction overlap: 0.999999561224.
# Full-frame rotation strength/derivative differences: 3.442e-15 / 6.772e-15.
# Complete signed sums: 0.705658047038 (ground), 0.377782587615 (excited).
# They match finite-Hilbert double commutators, not the continuum TRK value 2.
# Native run and internal checks, including compilation: 9.847 seconds.
