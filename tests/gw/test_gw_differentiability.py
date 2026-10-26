"""Differentiability acceptance tests for the GW module.

Verifies, on H2 / sto-3g (HF start):
1. QP energies are differentiable w.r.t. the mean-field inputs (mo_energy,
   mo_coeff) through the implicit custom VJP, matching central finite
   differences.
2. The implicit and unrolled differentiation modes agree.

These tests are the acceptance gate for the "differentiable GW" design
goal; they must pass before any GW stage is considered complete.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from gradscf import dft, gto
from gradscf.gw import g0w0_cd_restricted, g0w0_cd_unrestricted

_NW = 40


def _h2_inputs():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", cart=True)
    mf = dft.RKS(mol, xc="hf").run()
    assert mf.converged
    res = mf.scf_result
    inputs = mf._scf_inputs
    df_factors = getattr(inputs, "df_factors", None)
    if df_factors is None:
        from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors

        df_factors = eri_pair_matrix_to_df_factors(
            inputs.eri_pair_matrix, nao=res.mo_coeff.shape[0], tol=1e-12
        )
    return dict(
        mo_energy=jnp.asarray(res.mo_energy),
        mo_coeff=jnp.asarray(res.mo_coeff),
        nocc=1,
        df_factors=jnp.asarray(df_factors),
        fock_matrix=jnp.asarray(res.fock_matrix),
        hcore_matrix=jnp.asarray(res.hcore_matrix),
        density_matrix=jnp.asarray(res.density_matrix),
    )


def _homo_qp(inputs, mo_energy, mo_coeff, diff_mode="implicit"):
    res = g0w0_cd_restricted(
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
        nocc=inputs["nocc"],
        df_factors=inputs["df_factors"],
        fock_matrix=inputs["fock_matrix"],
        hcore_matrix=inputs["hcore_matrix"],
        density_matrix=inputs["density_matrix"],
        nw=_NW,
        diff_mode=diff_mode,
    )
    return res.mo_energy[inputs["nocc"] - 1]


def _finite_diff_grad(fn, x, eps=1e-5):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        xp = x.copy()
        xm = x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        grad[idx] = (fn(jnp.asarray(xp)) - fn(jnp.asarray(xm))) / (2.0 * eps)
    return grad


def test_qp_energy_grad_wrt_mo_energy_matches_finite_diff():
    inputs = _h2_inputs()
    fn = lambda e: float(_homo_qp(inputs, e, inputs["mo_coeff"]))
    grad_ad = np.asarray(jax.grad(lambda e: _homo_qp(inputs, e, inputs["mo_coeff"]))(inputs["mo_energy"]))
    grad_fd = _finite_diff_grad(fn, inputs["mo_energy"])
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-4, atol=1e-6)


def test_qp_energy_grad_wrt_mo_coeff_matches_finite_diff():
    inputs = _h2_inputs()
    fn = lambda c: float(_homo_qp(inputs, inputs["mo_energy"], c))
    grad_ad = np.asarray(jax.grad(lambda c: _homo_qp(inputs, inputs["mo_energy"], c))(inputs["mo_coeff"]))
    grad_fd = _finite_diff_grad(fn, inputs["mo_coeff"])
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-3, atol=1e-5)


def test_qp_energy_implicit_matches_unrolled():
    inputs = _h2_inputs()
    grad_implicit = jax.grad(lambda e: _homo_qp(inputs, e, inputs["mo_coeff"], "implicit"))(
        inputs["mo_energy"]
    )
    grad_unrolled = jax.grad(lambda e: _homo_qp(inputs, e, inputs["mo_coeff"], "unrolled"))(
        inputs["mo_energy"]
    )
    np.testing.assert_allclose(grad_implicit, grad_unrolled, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("diff_mode", ["implicit", "unrolled"])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_jit_preserves_full_gw_result(diff_mode, unrestricted):
    inputs = _h2_inputs()
    driver = g0w0_cd_restricted
    if unrestricted:
        driver = g0w0_cd_unrestricted
        for key in ("mo_energy", "mo_coeff", "fock_matrix"):
            inputs[key] = (inputs[key], inputs[key])
        inputs["density_matrix"] = (inputs["density_matrix"] / 2,) * 2
        inputs["nocc"] = (1, 1)

    def run(energy):
        return driver(**{**inputs, "mo_energy": energy}, nw=_NW, diff_mode=diff_mode)

    eager = run(inputs["mo_energy"])
    compiled = jax.jit(run)(inputs["mo_energy"])
    assert np.max(np.abs(eager.sigma_qp)) > 1e-3
    np.testing.assert_allclose(compiled.mo_energy, eager.mo_energy, rtol=0, atol=1e-10)
    np.testing.assert_allclose(compiled.sigma_qp, eager.sigma_qp, rtol=0, atol=1e-10)
    np.testing.assert_allclose(compiled.qp_residual, eager.qp_residual, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(compiled.converged_mask, eager.converged_mask)
    assert bool(compiled.converged) == bool(eager.converged)


def test_pole_only_gradient_matches_finite_difference():
    inputs = _h2_inputs()
    fn = lambda poles: g0w0_cd_restricted(
        **inputs, mo_energy_poles=poles, nw=_NW
    ).mo_energy[0]
    grad_ad = jax.grad(fn)(inputs["mo_energy"])
    grad_fd = _finite_diff_grad(fn, inputs["mo_energy"])
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-4, atol=1e-6)


def test_evaluate_only_preserves_selected_poles_and_reports_residual():
    inputs = _h2_inputs()
    poles = inputs["mo_energy"] + jnp.array([0.02, -0.01])
    run = jax.jit(lambda e: g0w0_cd_restricted(
        **inputs, mo_energy_poles=e, nw=_NW, orbs=(0,), evaluate_only=True
    ))
    result = run(poles)
    np.testing.assert_allclose(result.mo_energy[0], poles[0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.mo_energy[1], inputs["mo_energy"][1], rtol=0, atol=1e-14)
    # HF start: delta_v vanishes, so the residual is directly checkable.
    np.testing.assert_allclose(
        result.qp_residual[0], poles[0] - inputs["mo_energy"][0] - result.sigma_qp[0].real,
        rtol=0, atol=1e-12,
    )
    assert not bool(result.converged)
    assert float(result.qp_residual[1]) == 0.0
    assert complex(result.sigma_qp[1]) == 0.0j
    assert bool(result.converged_mask[1])
