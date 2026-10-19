"""Implicit derivatives of the coupled finite-grid scGW/number equations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw import scgw_matsubara_restricted
from gradscf.scf.autodiff import SCFDifferentiationConfig


_AD = SCFDifferentiationConfig(tolerance=1e-9, max_iter=20, restart=30)


def _inputs():
    return dict(
        mo_energy=jnp.array([-0.6, 0.65]), mo_coeff=jnp.eye(2), nocc=1,
        hcore_matrix=jnp.array([[-0.55, 0.04], [0.04, 0.6]]),
        df_factors=jnp.array([[[0.15, 0.07], [0.07, 0.1]], [[-0.08, 0.12], [0.12, 0.13]]]),
        nw=6, beta=5.0, tol=1e-10, particle_tol=1e-12, max_iter=100,
    )


def _observables(result):
    return jnp.array([result.total_energy, result.chemical_potential,
                      result.density_mo[0, 0], result.density_mo[0, 1]])


def test_compiled_primal_matches_eager_state():
    kw = _inputs()
    expected = scgw_matsubara_restricted(**kw)
    actual = jax.jit(lambda h: scgw_matsubara_restricted(
        **{**kw, "hcore_matrix": h}, differentiation=_AD
    ))(kw["hcore_matrix"])
    assert actual.converged
    for field in ("green_iw", "self_energy_iw", "sigma_moment", "density_mo", "total_energy", "chemical_potential"):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), rtol=0, atol=1e-9)


def test_noninteracting_canonical_internal_energy_gradient():
    kw = _inputs()
    kw.update(mo_energy=jnp.array([-0.5, 0.7]), hcore_matrix=jnp.diag(jnp.array([-0.5, 0.7])),
              df_factors=jnp.zeros((1, 2, 2)), beta=4.0)
    def energy(shift):
        return scgw_matsubara_restricted(
            **{**kw, "hcore_matrix": kw["hcore_matrix"].at[0, 0].add(shift)}, differentiation=_AD
        ).total_energy
    value, derivative = jax.jit(jax.value_and_grad(energy))(0.0)
    gap, x = 1.2, 1.2  # beta*gap/4
    expected_energy = 0.2 - gap * np.tanh(x)
    expected_derivative = 1 + np.tanh(x) + x / np.cosh(x)**2
    np.testing.assert_allclose(value, expected_energy, rtol=0, atol=1e-10)
    np.testing.assert_allclose(derivative, expected_derivative, rtol=1e-7, atol=1e-9)


def test_noninteracting_degenerate_density_has_finite_canonical_response():
    # Three coincident levels with N=2: each per-spin occupation is 1/3.
    # The full density is smooth under a traceless splitting even though
    # individual reference eigenvectors have no unique derivative.
    beta = 4.0
    direction = jnp.array([[0.2, 0.1, 0.0], [0.1, -0.2, 0.0], [0.0, 0.0, 0.0]])
    kwargs = dict(mo_energy=jnp.zeros(3), mo_coeff=jnp.eye(3), nocc=1,
                  df_factors=jnp.zeros((1, 3, 3)), beta=beta, nw=4,
                  max_iter=5, tol=1e-10, particle_tol=1e-12)

    def density(scale):
        return scgw_matsubara_restricted(
            **kwargs, hcore_matrix=scale * direction, differentiation=_AD
        ).density_mo

    value, tangent = jax.jit(lambda scale: jax.jvp(
        density, (scale,), (jnp.ones_like(scale),)
    ))(jnp.array(0.0))
    np.testing.assert_allclose(value, 2 * np.eye(3) / 3, rtol=0, atol=1e-11)
    expected = -2 * beta * (1 / 3) * (2 / 3) * direction
    np.testing.assert_allclose(tangent, expected, rtol=0, atol=1e-9)
    reverse = jax.jit(jax.grad(lambda scale: jnp.sum(density(scale) * direction)))(0.0)
    np.testing.assert_allclose(reverse, jnp.sum(expected * direction), rtol=0, atol=1e-9)


def test_noninteracting_limit_keeps_quadratic_hartree_exchange_energy():
    beta = 4.0
    vertex = jnp.array([[[0.1, 0.04, 0.0], [0.04, -0.08, 0.02], [0.0, 0.02, 0.05]]])
    kwargs = dict(mo_energy=jnp.zeros(3), mo_coeff=jnp.eye(3), nocc=1,
                  hcore_matrix=jnp.zeros((3, 3)), beta=beta, nw=4,
                  max_iter=100, tol=1e-12, particle_tol=1e-12)

    def energy(scale, response):
        return scgw_matsubara_restricted(
            **kwargs, df_factors=scale * vertex, differentiation=response
        ).total_energy

    np.testing.assert_allclose(jax.jit(jax.grad(lambda scale: energy(scale, _AD)))(0.0),
                               0.0, rtol=0, atol=1e-12)
    # D=2I/3 at B=0. E_H+E_x is quadratic in B; correlation starts at B^4.
    # Check finite nonzero couplings, rather than claim higher AD at B=0.
    expected = (2 * jnp.sum(jnp.trace(vertex, axis1=-2, axis2=-1)**2)
                - jnp.sum(vertex**2)) / 9
    for step in (1e-3, 2e-3):
        finite_coefficient = (energy(step, None) + energy(-step, None)) / (2 * step**2)
        np.testing.assert_allclose(finite_coefficient, expected, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("beta,nw", [(5.0, 6), (20.0, 12)])
def test_uniform_energy_shift_includes_chemical_potential_response(beta, nw):
    kw = {**_inputs(), "beta": beta, "nw": nw, "max_iter": 200}
    def observables(shift):
        r = scgw_matsubara_restricted(
            **{**kw, "hcore_matrix": kw["hcore_matrix"] + shift * jnp.eye(2)}, differentiation=_AD
        )
        return jnp.array([r.total_energy, r.chemical_potential, jnp.trace(r.density_mo),
                          jnp.sum(r.green_iw.real), r.correlation_energy])
    _, tangent = jax.jit(lambda s: jax.jvp(observables, (s,), (jnp.ones_like(s),)))(jnp.array(0.0))
    np.testing.assert_allclose(tangent, [2.0, 1.0, 0.0, 0.0, 0.0], rtol=0, atol=2e-7)


@pytest.mark.parametrize("variable", ["diagonal", "off_diagonal", "df_factors"])
def test_implicit_response_matches_reconverged_finite_difference(variable):
    kw = _inputs()
    def run(scale, differentiation):
        args = dict(kw)
        if variable == "df_factors":
            args["df_factors"] = kw["df_factors"] * (1 + scale)
        else:
            direction = jnp.array([[0.3, 0.0], [0.0, -0.2]]) if variable == "diagonal" else jnp.array([[0.0, 0.2], [0.2, 0.0]])
            args["hcore_matrix"] = kw["hcore_matrix"] + scale * direction
        return _observables(scgw_matsubara_restricted(**args, differentiation=differentiation))
    step = 1e-4
    fd = (run(step, None) - run(-step, None)) / (2 * step)
    observed = lambda scale: run(scale, _AD)
    _, tangent = jax.jit(lambda s: jax.jvp(observed, (s,), (jnp.ones_like(s),)))(jnp.array(0.0))
    np.testing.assert_allclose(tangent, fd, rtol=2e-5, atol=2e-7)
    reverse = jax.jit(jax.grad(lambda scale: observed(scale)[0]))(0.0)
    np.testing.assert_allclose(reverse, tangent[0], rtol=1e-6, atol=1e-8)


def test_initial_spectrum_is_only_a_guess():
    kw = _inputs()
    loss = lambda guess: scgw_matsubara_restricted(**{**kw, "mo_energy": guess}, differentiation=_AD).total_energy
    np.testing.assert_array_equal(jax.jit(jax.grad(loss))(kw["mo_energy"]), jnp.zeros(2))


def test_nonconverged_compiled_primal_rejects_response():
    kw = {**_inputs(), "max_iter": 1}
    fn = jax.jit(jax.value_and_grad(lambda scale: scgw_matsubara_restricted(
        **{**kw, "df_factors": kw["df_factors"] * scale}, differentiation=_AD
    ).total_energy))
    with pytest.raises(Exception, match="scGW.*converge"):
        jax.block_until_ready(fn(1.0))


def test_unrolled_mode_is_not_a_silent_stopped_gradient():
    with pytest.raises(ValueError, match="implicit"):
        scgw_matsubara_restricted(**_inputs(), differentiation=SCFDifferentiationConfig(mode="unrolled"))


def test_unresolved_cold_charge_response_is_rejected():
    kw = _inputs()
    kw.update(mo_energy=jnp.array([-1.0, 1.0]), hcore_matrix=jnp.diag(jnp.array([-1.0, 1.0])),
              df_factors=jnp.zeros((1, 2, 2)), beta=100.0)
    derivative = jax.jit(jax.grad(lambda shift: scgw_matsubara_restricted(
        **{**kw, "hcore_matrix": kw["hcore_matrix"] + shift * jnp.eye(2)}, differentiation=_AD
    ).chemical_potential))(0.0)
    assert not jnp.isfinite(derivative)


@pytest.mark.parametrize("adjoint", [False, True])
def test_negative_self_consistent_charge_susceptibility_is_rejected(adjoint):
    from gradscf.gw.scgw_response import _charge_linear_solve

    # The charge Schur complement is -0.46. Finite-T stable branches have
    # dN/dmu >= 0; its magnitude alone cannot certify a physical response.
    matrix = jnp.array([[1.0, 0.2], [0.3, -0.4]])
    solve = lambda rhs: _charge_linear_solve(
        lambda vector: matrix @ vector, rhs, _AD, 5.0, 1e-10, True
    )
    rhs = jnp.array([1.0, 0.1])
    if adjoint:
        _, pullback = jax.vjp(solve, rhs)
        result = pullback(jnp.ones_like(rhs))[0]
    else:
        result = solve(rhs)
    assert np.all(np.isnan(result))


@pytest.mark.parametrize("adjoint", [False, True])
def test_positive_charge_susceptibility_preserves_checked_response(adjoint):
    from gradscf.gw.scgw_response import _charge_linear_solve

    matrix = jnp.array([[1.0, 0.2], [0.3, 0.4]])
    solve = lambda rhs: _charge_linear_solve(
        lambda vector: matrix @ vector, rhs, _AD, 5.0, 1e-10, True
    )
    rhs = jnp.array([1.0, 0.1])
    if adjoint:
        _, pullback = jax.vjp(solve, rhs)
        result = pullback(jnp.ones_like(rhs))[0]
        expected = np.linalg.solve(np.asarray(matrix).T, np.ones(2))
    else:
        result = solve(rhs)
        expected = np.linalg.solve(np.asarray(matrix), np.asarray(rhs))
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-10)


def test_matrix_free_tangent_matches_dense_residual_jacobian():
    from gradscf.gw.scgw_response import _pack_state, _residual
    kw = {**_inputs(), "nw": 4}
    primal = scgw_matsubara_restricted(**kw)
    seed = _pack_state(primal.fock_mo, primal.self_energy_iw, primal.sigma_moment,
                       primal.chemical_potential, primal.grid)
    params = {"hcore": kw["hcore_matrix"], "b": kw["df_factors"]}
    direction = jnp.array([[0.3, 0.1], [0.1, -0.2]])
    matrix = jax.jacfwd(lambda z: _residual(z, params, primal.grid, 1))(seed)
    _, parameter_tangent = jax.jvp(
        lambda h: _residual(seed, {**params, "hcore": h}, primal.grid, 1),
        (params["hcore"],), (direction,),
    )
    dense = jnp.linalg.solve(matrix, -parameter_tangent)
    def solved(scale):
        r = scgw_matsubara_restricted(
            **{**kw, "hcore_matrix": params["hcore"] + scale * direction}, differentiation=_AD
        )
        return _pack_state(r.fock_mo, r.self_energy_iw, r.sigma_moment, r.chemical_potential, r.grid)
    _, actual = jax.jit(lambda s: jax.jvp(solved, (s,), (jnp.ones_like(s),)))(jnp.array(0.0))
    np.testing.assert_allclose(actual, dense, rtol=1e-6, atol=2e-8)
    np.testing.assert_allclose(matrix @ actual + parameter_tangent, 0.0, rtol=0, atol=1e-8)


def test_insufficient_adjoint_budget_returns_nan():
    kw = _inputs()
    policy = SCFDifferentiationConfig(tolerance=1e-12, max_iter=1, restart=1)
    derivative = jax.jit(jax.grad(lambda scale: scgw_matsubara_restricted(
        **{**kw, "df_factors": kw["df_factors"] * scale}, differentiation=policy
    ).total_energy))(1.0)
    assert not jnp.isfinite(derivative)


def test_implicit_gradient_is_independent_of_mixing_trajectory():
    kw = _inputs()
    derivatives = []
    for mixing in (0.3, 0.7):
        def loss(scale):
            return scgw_matsubara_restricted(
                **{**kw, "df_factors": kw["df_factors"] * scale,
                   "mixing": mixing, "mo_energy": kw["mo_energy"] + mixing * 0.2},
                differentiation=_AD,
            ).total_energy
        derivatives.append(jax.jit(jax.value_and_grad(loss))(1.0))
    np.testing.assert_allclose(derivatives[0], derivatives[1], rtol=1e-6, atol=1e-8)


def test_h2_df_response_matches_reconverged_finite_difference():
    from gradscf import dft, gto
    from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", cart=True)
    mf = dft.RKS(mol, xc="hf").run()
    assert mf.converged
    ref = mf.scf_result
    factors = getattr(mf._scf_inputs, "df_factors", None)
    if factors is None:
        factors = eri_pair_matrix_to_df_factors(mf._scf_inputs.eri_pair_matrix, nao=2, tol=1e-12)
    kw = dict(mo_energy=ref.mo_energy, mo_coeff=ref.mo_coeff, nocc=1,
              hcore_matrix=ref.hcore_matrix, df_factors=factors,
              nuclear_repulsion=ref.nuclear_repulsion, beta=8.0, nw=8,
              max_iter=200, tol=1e-10, particle_tol=1e-12)
    def loss(scale, policy):
        return scgw_matsubara_restricted(**{**kw, "df_factors": factors * scale}, differentiation=policy).total_energy
    step = 1e-4
    fd = (loss(1.0 + step, None) - loss(1.0 - step, None)) / (2 * step)
    actual = jax.jit(jax.grad(lambda scale: loss(scale, _AD)))(1.0)
    np.testing.assert_allclose(actual, fd, rtol=2e-5, atol=2e-7)


def test_three_orbital_implicit_response_matches_finite_difference():
    rng = np.random.default_rng(81)
    b = rng.normal(size=(2, 3, 3)) * 0.1
    b = jnp.asarray(0.5 * (b + b.transpose(0, 2, 1)))
    kw = dict(mo_energy=jnp.array([-0.6, 0.5, 1.2]), mo_coeff=jnp.eye(3), nocc=1,
              hcore_matrix=jnp.array([[-0.6, 0.02, 0.01], [0.02, 0.5, -0.03], [0.01, -0.03, 1.2]]),
              nw=4, beta=4.0, max_iter=100, tol=1e-10, particle_tol=1e-12)
    def loss(scale, policy):
        return scgw_matsubara_restricted(**kw, df_factors=b * scale, differentiation=policy).total_energy
    step = 1e-4
    fd = (loss(1 + step, None) - loss(1 - step, None)) / (2 * step)
    actual = jax.jit(jax.grad(lambda s: loss(s, _AD)))(1.0)
    np.testing.assert_allclose(actual, fd, rtol=2e-5, atol=2e-7)
