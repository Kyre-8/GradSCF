"""Relaxed nuclear response of CCSD Vee without a full spin 2-RDM export.

The independent oracle differentiates E_CC(h_MO(R), strength*G_MO(R)) at
strength=1. The HF frame is fixed for that partial derivative and reconverges
with R; no stop_gradient freezes its nuclear response. Units are atomic.
"""
import importlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc, gto, scf
from gradscf._memory import tree_storage_bytes
from gradscf.ao2mo import transform_integrals
from gradscf.solvers.diagnostics import require_converged_derivative
from gradscf_tools.molecular_response import MolecularEnergy


def test_water_ccsd_two_body_nuclear_gradient_and_hvp(monkeypatch):
    molecule = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587',
                     basis='sto-3g', unit='Angstrom', cart=True)
    mf = scf.RHF(molecule, integral_backend='native', jk_backend='direct',
        max_memory=256., conv_tol=1e-13, conv_tol_density=1e-11,
        conv_tol_grad=1e-11, max_cycle=250).run()
    assert mf.converged
    moving = MolecularEnergy(mf, jk_backend='direct', max_memory=256.,
                             gradient_tolerance=1e-11)
    config = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11, adjoint_tol=1e-11)
    nmo, nocc = mf.mo_coeff.shape[1], molecule.nelectron//2
    properties = importlib.import_module('gradscf.cc.properties')

    def forbidden(*args, **kwargs):
        pytest.fail('Contracted Vee exported a complete spin 2-RDM')

    monkeypatch.setattr(properties, 'active_density_parts', forbidden)
    monkeypatch.setattr(properties, '_rdm2_dense_from_states', forbidden)

    def frame(coordinates):
        hf, inputs = moving.solve_with_inputs(coordinates)
        basis = inputs['direct_basis']
        h, g = transform_integrals(inputs['hcore'], hf.mo_coeff,
            direct_basis=basis, max_memory=256.,
            reserved_bytes=tree_storage_bytes(hf, inputs, moving._memory_inputs(),
                exclude=(inputs['hcore'], hf.mo_coeff, basis)))
        return hf, h, g, inputs['nuclear_repulsion']

    def states(coordinates):
        hf, h, g, nuclear = frame(coordinates)
        held = tree_storage_bytes(hf, moving._memory_inputs())
        right = cc.run_cc(h, g, nocc=nocc, nuclear_repulsion=nuclear,
            config=config, max_memory=256., reserved_bytes=held)
        left = cc.solve_lambda(h, g, right, nocc=nocc, config=config,
            max_memory=256., reserved_bytes=held)
        return hf, h, g, nuclear, right, left

    solve_states = jax.jit(states)

    def contracted(coordinates):
        hf, h, g, nuclear, right, left = solve_states(coordinates)
        # The restricted contraction already supplies the factor 1/2.
        vee = properties._contract_rdm2_from_states(g, right, left,
            nmo=nmo, nocc=nocc, dtype=jnp.float64, config=config, max_memory=256.,
            reserved_bytes=tree_storage_bytes(hf, h, nuclear, moving._memory_inputs()))
        valid = hf.stationary & right.converged & left.converged
        return require_converged_derivative(jnp.where(valid, vee, jnp.nan), valid)

    def scaled_cc(hf, h, g, nuclear, strength):
        scaled = strength*g
        result = cc.run_cc(h, scaled, nocc=nocc, nuclear_repulsion=nuclear,
            config=config, max_memory=256.,
            reserved_bytes=tree_storage_bytes(hf, g, moving._memory_inputs(), exclude=(scaled,)))
        valid = hf.stationary & result.converged
        return require_converged_derivative(
            jnp.where(valid, result.total_energy, jnp.nan), valid)

    def oracle(coordinates):
        hf, h, g, nuclear = frame(coordinates)
        # Existing raw CC preparation rebuilds reference/Fock terms with G.
        # HF is outside this derivative but remains live for outer R response.
        return jax.grad(lambda strength: scaled_cc(hf, h, g, nuclear, strength))(jnp.asarray(1.))

    coordinates = moving.coordinates
    print('CC Vee geometry: point states and energy decomposition', flush=True)
    hf, h, g, nuclear, right, left = jax.block_until_ready(solve_states(coordinates))
    assert bool(hf.stationary) and bool(right.converged) and bool(left.converged)
    assert abs(float(right.correlation_energy)) > 1e-6
    assert np.linalg.norm(np.asarray(left.adjoint)) > 1e-6
    one = properties._density_from_lambda(h, g, right, left,
        nocc=nocc, frozen=None, config=config, max_memory=256.)
    point = jax.jit(contracted)(coordinates)
    one_body = jnp.einsum('pq,qp->', h, one, precision='highest')
    for value in (one, point, one_body, right.total_energy, left.residual_norm):
        assert np.isfinite(np.asarray(value)).all()
    np.testing.assert_allclose(jnp.trace(one), molecule.nelectron, atol=3e-8, rtol=0)
    np.testing.assert_allclose(nuclear+one_body+point, right.total_energy, atol=3e-8, rtol=0)

    print('CC Vee geometry: fixed-frame coupling derivative', flush=True)
    fixed_energy = jax.jit(lambda strength: scaled_cc(hf, h, g, nuclear, strength))
    coupling_derivative = jax.jit(jax.grad(fixed_energy))(jnp.asarray(1.))
    assert np.isfinite(coupling_derivative)
    np.testing.assert_allclose(point, coupling_derivative, atol=3e-8, rtol=0)
    coupling_errors = []
    for step in (1e-4, 5e-5):
        plus, minus = fixed_energy(1+step), fixed_energy(1-step)
        assert np.isfinite(plus) and np.isfinite(minus)
        finite = (plus-minus)/(2*step)
        np.testing.assert_allclose(coupling_derivative, finite, atol=3e-7, rtol=2e-6)
        coupling_errors.append(dict(step=step, error=float(abs(finite-coupling_derivative))))
    print('CC Vee point: '+json.dumps(dict(vee_hartree=float(point),
        coupling_derivative_hartree=float(coupling_derivative),
        cc_total_hartree=float(right.total_energy), correlation_hartree=float(right.correlation_energy),
        nuclear_repulsion_hartree=float(nuclear), one_body_hartree=float(one_body),
        decomposition_error=float(abs(nuclear+one_body+point-right.total_energy)),
        cc_residual=float(right.residual_norm), lambda_residual=float(left.residual_norm),
        minimum_denominator=float(right.min_abs_denominator), coupling_finite_differences=coupling_errors)), flush=True)

    direction = np.random.default_rng(750).normal(size=coordinates.shape)
    direction -= direction.mean(axis=0)
    direction = jnp.asarray(direction/np.linalg.norm(direction))

    def bundle(function):
        return jax.jit(lambda r: jax.jvp(jax.value_and_grad(function),
            (r,), (direction,)))(coordinates)

    print('CC Vee geometry: contracted full gradient and HVP', flush=True)
    actual = jax.block_until_ready(bundle(contracted))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(actual))
    print('CC Vee geometry: coupling-oracle nuclear gradient and HVP', flush=True)
    expected = jax.block_until_ready(bundle(oracle))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(expected))
    (value, gradient), (jvp, hvp) = actual
    (reference, reference_gradient), (reference_jvp, reference_hvp) = expected
    assert np.linalg.norm(np.asarray(gradient)) > 1e-6
    assert np.linalg.norm(np.asarray(hvp)) > 1e-6
    np.testing.assert_allclose(value, reference, atol=3e-8, rtol=0)
    np.testing.assert_allclose(gradient, reference_gradient, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(hvp, reference_hvp, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(jvp, jnp.sum(gradient*direction), atol=3e-9, rtol=0)
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=3e-8, rtol=0)

    print('CC Vee geometry: two displaced nuclear differences', flush=True)
    evaluate = jax.jit(jax.value_and_grad(contracted))
    errors = []
    for step in (3e-4, 1e-4):
        plus, plus_gradient = evaluate(coordinates+step*direction)
        minus, minus_gradient = evaluate(coordinates-step*direction)
        assert all(np.isfinite(np.asarray(x)).all() for x in (plus, minus, plus_gradient, minus_gradient))
        finite_gradient = (plus-minus)/(2*step)
        finite_hvp = (plus_gradient-minus_gradient)/(2*step)
        np.testing.assert_allclose(jvp, finite_gradient, atol=3e-7, rtol=2e-5)
        np.testing.assert_allclose(hvp, finite_hvp, atol=3e-6, rtol=2e-5)
        errors.append(dict(step_bohr=step, gradient_error=float(abs(jvp-finite_gradient)),
                           hvp_max_error=float(jnp.max(abs(hvp-finite_hvp)))))
    print('CC Vee geometry verified: '+json.dumps(dict(
        quantity='Vee=0.5 G:Gamma', energy_unit='Hartree', coordinate_unit='Bohr',
        coordinates_bohr=np.asarray(coordinates).tolist(), direction=np.asarray(direction).tolist(),
        gradient=np.asarray(gradient).tolist(), hvp=np.asarray(hvp).tolist(),
        oracle_gradient_max_error=float(jnp.max(abs(gradient-reference_gradient))),
        oracle_hvp_max_error=float(jnp.max(abs(hvp-reference_hvp))),
        finite_differences=errors)), flush=True)
