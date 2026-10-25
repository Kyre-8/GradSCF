"""Complete canonical RHF singlet CIS(D) excitation-energy geometry response.

The scalar is Omega_CIS(D), not E_HF + Omega or an absolute excited-state
energy. Coordinates/derivatives use Bohr and Hartree. All numerical algorithms
are the existing SCF, Hermitian response, AO CIS and selected correction paths.
"""
from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import ci, gto, scf
from gradscf._memory import tree_storage_bytes
from gradscf.ci.corrections import cis_d_from_orbitals
from gradscf.ci.solver import solve_cis_response
from gradscf.solvers import (EigenResponseConfig, EigenSolverConfig,
                             LinearSolverConfig, solve_hermitian)
from gradscf.solvers.diagnostics import require_converged_derivative
from gradscf_tools.molecular_response import MolecularEnergy


def test_canonical_water_cis_d_nuclear_gradient_and_hvp():
    # Same fixed geometry as test_direct_cis.water_direct_hf; no optimization.
    molecule = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587',
                     basis='sto-3g', unit='Angstrom', cart=True)
    mf = scf.RHF(molecule, integral_backend='native', jk_backend='direct',
        max_memory=256., conv_tol=1e-13, conv_tol_density=1e-11,
        conv_tol_grad=1e-11, max_cycle=250).run()
    assert mf.converged
    moving = MolecularEnergy(mf, jk_backend='direct', max_memory=256.,
                             gradient_tolerance=1e-11)
    seed = scf.as_reference(mf)
    nmo, nocc = mf.mo_coeff.shape[1], molecule.nelectron//2
    response = EigenResponseConfig(target='eigenpairs',
        linear_config=LinearSolverConfig(method='direct', max_dense=16, rtol=1e-11))
    cis_config = ci.CIConfig(nroots=2, solver='dense', conv_tol=1e-11)
    oracle = ci.CIS_D(mf, nroots=2, solver='dense', conv_tol=1e-11).run()
    assert np.all(oracle.converged)

    def omega(coordinates):
        state, inputs = moving.solve_with_inputs(coordinates)
        transported = state.mo_coeff
        fock = state.fock_matrix.mean(axis=0)
        matrix = transported.T @ fock @ transported
        matrix = .5*(matrix+matrix.T)
        retained = (state, inputs, moving._memory_inputs(), seed)
        canonical = solve_hermitian(matrix,
            config=EigenSolverConfig(method='dense', nroots=nmo, atol=1e-11),
            response=response, max_memory=256.,
            reserved_bytes=tree_storage_bytes(*retained, exclude=(matrix,)))
        coefficients = transported @ canonical.vectors
        reference = replace(seed, mo_coeff=coefficients, mo_energy=canonical.values,
            rdm1=state.density_matrix, fock_matrix=fock, h1e=inputs['hcore'],
            overlap_matrix=inputs['overlap'], atom_coords=coordinates,
            nuclear_repulsion=inputs['nuclear_repulsion'], mf_energy=state.total_energy,
            rep_tensor=None, eri_pair_matrix=None, eri_ovov=None, eri_ovvo=None,
            eri_oovv=None, df_factors=None, response_df_factors_j=None,
            response_df_factors_k=None, direct_basis=inputs['direct_basis'])
        singles = solve_cis_response(reference, config=cis_config, response=response,
            max_memory=256., reserved_bytes=tree_storage_bytes(*retained,
                canonical, coefficients, exclude=(reference,)))
        assert singles.amplitude_response
        corrected = cis_d_from_orbitals(coefficients, canonical.values, singles,
            nocc=nocc, direct_basis=inputs['direct_basis'], max_memory=256.,
            reserved_bytes=tree_storage_bytes(*retained, reference, canonical,
                exclude=(coefficients, canonical.values, singles, inputs['direct_basis'])))
        valid = state.stationary & jnp.all(canonical.response_valid) & jnp.all(corrected.valid)
        value = require_converged_derivative(
            jnp.where(valid, corrected.excitation_energies[0], jnp.nan), valid)
        diagnostics = dict(hf_energy=state.total_energy, cis_energy=corrected.cis_energies[0],
            correction=corrected.corrections[0], omega=value,
            minimum_orbital_gap=jnp.min(jnp.diff(canonical.values)),
            cis_root_gap=corrected.cis_energies[1]-corrected.cis_energies[0],
            minimum_denominator=jnp.min(corrected.min_abs_denominators),
            valid=valid)
        return value, diagnostics

    coordinates = moving.coordinates
    direction = np.random.default_rng(749).normal(size=coordinates.shape)
    direction -= direction.mean(axis=0)
    direction = jnp.asarray(direction/np.linalg.norm(direction))
    evaluate = jax.jit(jax.value_and_grad(omega, has_aux=True))
    print('CIS(D) geometry: complete value/gradient', flush=True)
    (value, diagnostics), gradient = jax.block_until_ready(evaluate(coordinates))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((value, diagnostics, gradient)))
    assert bool(diagnostics['valid'])
    assert abs(float(diagnostics['correction'])) > 1e-6
    assert float(diagnostics['minimum_orbital_gap']) > 1e-4
    assert float(diagnostics['cis_root_gap']) > 1e-4
    assert float(diagnostics['minimum_denominator']) > 1e-4
    np.testing.assert_allclose(value, oracle.e[0], atol=3e-9, rtol=0)
    np.testing.assert_allclose(diagnostics['correction'], oracle.correction[0], atol=3e-9, rtol=0)
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=3e-8, rtol=0)
    print('CIS(D) center: '+json.dumps({key: float(x) for key, x in diagnostics.items()}), flush=True)

    scalar = lambda r: omega(r)[0]
    print('CIS(D) geometry: complete Hessian-vector product', flush=True)
    hvp = jax.jit(lambda r: jax.jvp(jax.grad(scalar), (r,), (direction,))[1])(coordinates)
    jax.block_until_ready(hvp)
    assert np.isfinite(np.asarray(hvp)).all()
    errors = []
    for step in (3e-4, 1e-4):
        (plus, plus_diagnostics), plus_gradient = evaluate(coordinates+step*direction)
        (minus, minus_diagnostics), minus_gradient = evaluate(coordinates-step*direction)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(
            (plus, minus, plus_diagnostics, minus_diagnostics, plus_gradient, minus_gradient)))
        assert bool(plus_diagnostics['valid']) and bool(minus_diagnostics['valid'])
        finite_gradient = (plus-minus)/(2*step)
        finite_hvp = (plus_gradient-minus_gradient)/(2*step)
        np.testing.assert_allclose(jnp.sum(gradient*direction), finite_gradient,
                                   atol=3e-7, rtol=2e-5)
        np.testing.assert_allclose(hvp, finite_hvp, atol=3e-6, rtol=2e-5)
        errors.append(dict(step_bohr=step,
            gradient_direction_error=float(abs(jnp.sum(gradient*direction)-finite_gradient)),
            hvp_max_error=float(jnp.max(jnp.abs(hvp-finite_hvp)))))
    print('CIS(D) geometry verified: '+json.dumps(dict(
        quantity='Omega_CIS(D)', energy_unit='Hartree', coordinate_unit='Bohr',
        coordinates_bohr=np.asarray(coordinates).tolist(), direction=np.asarray(direction).tolist(),
        gradient=np.asarray(gradient).tolist(), hvp=np.asarray(hvp).tolist(),
        finite_differences=errors)), flush=True)
