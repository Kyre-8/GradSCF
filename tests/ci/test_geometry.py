"""CISD nuclear response in a smooth HF frame with degenerate pi orbitals."""
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def check_nitrogen_cisd_geometry(*, basis="sto-3g", conv_tol=1e-11,
                                 max_cycle=150, max_space=32, report_path=None):
    """One native N2 geometry calculation and independent gradient checks."""
    pytest.importorskip("pyscf")
    from pyscf import gto as pgto, scf as pscf, ci as pci
    from gradscf import gto, scf, ci, integrals
    from gradscf.integrals.molecular.ao2mo import transform_integrals
    from gradscf_tools.molecular_response import MolecularEnergy

    coordinates = np.array([[0., 0., -1.04], [0., 0., 1.04]])
    atoms = list(zip(("N", "N"), coordinates))
    report = dict(complete=False, basis=basis, cart=True, coordinates_bohr=coordinates.tolist(),
                  ci_conv_tol=conv_tol, ci_max_cycle=max_cycle, ci_max_space=max_space)
    def record(phase, **values):
        report.update(phase=phase, **values)
        if report_path is not None:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2)+"\n")
        print("CISD geometry:", phase, flush=True)

    mol = gto.M(atom=atoms, basis=basis, unit="Bohr", cart=True)
    topology, parameters = integrals.prepare_basis(mol.to_spec(), basis, cart=True)
    overlap = np.asarray(integrals.make_plan(topology, backend="native").evaluate("overlap", parameters))
    nao = overlap.shape[0]
    seed_scale = mol.nelectron/np.trace(overlap)
    seed = seed_scale*np.eye(nao)
    record("native_hf", nao=nao, nelectron=mol.nelectron, seed_scale=float(seed_scale))
    mf = scf.RHF(mol, integral_backend="native", init_guess=seed,
                 conv_tol=1e-13, conv_tol_grad=1e-10, conv_tol_density=1e-11).run()
    record("native_hf_result", native_hf_converged=bool(mf.converged),
           native_hf_energy_hartree=float(mf.e_tot),
           minimum_orbital_gap_hartree=float(np.min(np.abs(np.diff(np.asarray(mf.mo_energy))))))
    assert mf.converged
    assert np.min(np.abs(np.diff(np.asarray(mf.mo_energy)))) < 1e-9

    record("reference_hf")
    pmol = pgto.M(atom=atoms, basis=basis, unit="Bohr", cart=True, verbose=0)
    pmf = pscf.RHF(pmol)
    independent_seed = pmol.nelectron/np.trace(pmf.get_ovlp())*np.eye(pmol.nao_nr())
    pmf.run(dm0=independent_seed, conv_tol=1e-13, conv_tol_grad=1e-11)
    record("reference_hf_result", reference_hf_converged=bool(pmf.converged),
           reference_hf_energy_hartree=float(pmf.e_tot),
           hf_absolute_error_hartree=abs(float(mf.e_tot)-pmf.e_tot))
    assert pmf.converged
    np.testing.assert_allclose(mf.e_tot, pmf.e_tot, atol=2e-10, rtol=0)
    record("reference_cisd")
    external = pci.CISD(pmf).run(conv_tol=1e-12, max_cycle=150)
    assert external.converged
    analytic = external.nuc_grad_method().kernel()
    record("reference_cisd_result", reference_cisd_energy_hartree=float(external.e_tot),
           reference_cisd_gradient_hartree_bohr=np.asarray(analytic).tolist())

    reference = MolecularEnergy(mf, gradient_tolerance=1e-10)
    nmo, nocc = mf.mo_coeff.shape[1], mol.nelectron//2
    space = ci.make_ci_space(nmo, nocc, max_excitation=2)
    config = ci.CIConfig(conv_tol=conv_tol, max_cycle=max_cycle, max_space=max_space)
    record("gradient_lowering", nmo=nmo, nocc=nocc, determinants=space.size)

    def energy(geometry):
        state, inputs = reference.solve_with_inputs(geometry)
        integrals_ao = inputs["eri"]
        layout = {"eri" if integrals_ao.ndim == 4 else "eri_pair_matrix": integrals_ao}
        h, g = transform_integrals(inputs["hcore"], state.mo_coeff, **layout)
        result = ci.solve_ci(h, g, space, config=config,
                             nuclear_repulsion=inputs["nuclear_repulsion"])
        return jnp.where(state.stationary & result.converged[0], result.total_energies[0], jnp.nan)

    started = time.perf_counter()
    lowered = jax.jit(jax.value_and_grad(energy)).lower(reference.coordinates)
    gradient_lower_seconds = time.perf_counter()-started
    record("gradient_compilation", gradient_lower_seconds=gradient_lower_seconds)
    started = time.perf_counter()
    evaluate = lowered.compile()
    gradient_compile_seconds = time.perf_counter()-started
    record("gradient_execution", gradient_compile_seconds=gradient_compile_seconds)
    print("CISD gradient: lower = %.3f s; compile = %.3f s"
          % (gradient_lower_seconds, gradient_compile_seconds), flush=True)
    started = time.perf_counter()
    value, gradient = evaluate(reference.coordinates)
    jax.block_until_ready((value, gradient))
    gradient_execute_seconds = time.perf_counter()-started
    record("gradient_result", gradient_execute_seconds=gradient_execute_seconds,
           energy_hartree=float(value), gradient_hartree_bohr=np.asarray(gradient).tolist(),
           energy_absolute_error_hartree=abs(float(value)-external.e_tot),
           analytic_gradient_max_absolute_error=float(np.max(np.abs(np.asarray(gradient)-analytic))))
    assert np.isfinite(value) and np.isfinite(np.asarray(gradient)).all()
    np.testing.assert_allclose(value, external.e_tot, atol=3e-9, rtol=0)
    np.testing.assert_allclose(gradient, analytic, atol=5e-7, rtol=0)
    np.testing.assert_allclose(np.asarray(gradient).sum(axis=0), 0., atol=3e-8, rtol=0)
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    # Reuse the existing compiled value-and-gradient executable at displaced
    # geometries. These timings include the unused displaced gradients.
    finite_errors = []
    finite_values = []
    record("finite_differences")
    started = time.perf_counter()
    for step in (3e-4, 1e-4):
        plus = evaluate(reference.coordinates+step*direction)
        minus = evaluate(reference.coordinates-step*direction)
        jax.block_until_ready((plus, minus))
        finite = (plus[0]-minus[0])/(2*step)
        np.testing.assert_allclose(jnp.sum(gradient*direction), finite, atol=6e-7, rtol=0)
        finite_errors.append(abs(float(jnp.sum(gradient*direction)-finite)))
        finite_values.append(dict(step_bohr=step, derivative=float(finite), absolute_error=finite_errors[-1]))
    displaced_value_and_gradient_seconds = time.perf_counter()-started
    print("N2 CISD: determinants = %d; E = %.12f Ha; dE/dR = %.10f Ha/bohr; "
          "energy error = %.3e Ha; analytic gradient error = %.3e Ha/bohr; "
          "FD error = %.3e Ha/bohr"
          % (space.size, value, jnp.sum(gradient*direction), abs(float(value)-external.e_tot),
             np.max(np.abs(np.asarray(gradient)-analytic)), max(finite_errors)), flush=True)
    record("complete", complete=True, finite_differences=finite_values,
           displaced_value_and_gradient_seconds=displaced_value_and_gradient_seconds)
    print("CISD execution = %.3f s; four displaced value-and-gradient executions = %.3f s"
          % (gradient_execute_seconds, displaced_value_and_gradient_seconds), flush=True)
    return report


def test_nitrogen_cisd_geometry_matches_pyscf_gradient():
    check_nitrogen_cisd_geometry()
