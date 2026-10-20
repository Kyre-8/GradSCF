"""QP selection regressions for fixed G/W equations (CPU float64, Hartree)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw import qp
from gradscf.solvers.nonlinear import ScalarRootConfig


def test_qp_forward_does_not_accept_a_negative_weight_root():
    # This Dyson equation has a zero, but F'=-1 gives a negative formal Z.
    root, accepted = qp._qp_forward(lambda x: -x, jnp.array([.1]),
        tol=1e-10, maxiter=100, method='hybrid')
    np.testing.assert_allclose(root, 0., atol=1e-10)
    assert not bool(accepted[0])


def test_qp_forward_selects_the_on_shell_candidate_in_a_two_root_model():
    # A causal one-bath self-energy has two roots with positive weights.
    # Starting between the bath pole and zero sends the old local solve left.
    def equation(x):
        return x-.09/(x+.02)
    roots, accepted = qp._qp_forward(equation, jnp.array([-.01]),
        tol=1e-10, maxiter=160, method='hybrid')
    expected = (-.02+np.sqrt(.02**2+4*.09))/2
    np.testing.assert_allclose(roots, expected, atol=1e-9, rtol=0)
    assert bool(accepted[0])


def test_candidates_record_multiple_roots_and_fixed_gw_ad_slopes():
    def equation(x):
        return x-.09/(x+.02)
    result = qp.solve_quasiparticle_roots(equation, jnp.array([0.]),
        initial_guess=jnp.array([-.01]),
        config=ScalarRootConfig(method='hybrid', maxiter=160, ftol=1e-10, xtol=1e-10))
    expected = (-.02+np.sqrt(.02**2+4*.09))/2
    np.testing.assert_allclose(result.roots, expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose(result.on_shell, 4.5, atol=1e-12)
    np.testing.assert_allclose(result.weight, 1/(1+.09/(expected+.02)**2), atol=1e-9)
    assert int(result.candidate_count[0]) >= 2
    assert bool(result.branch_valid[0])


def test_no_bracket_crossing_is_certified_as_a_root():
    # A discontinuity can give a sign change; it must still fail F=0.
    result = qp.solve_quasiparticle_roots(lambda x: jnp.where(x < 0, -1., 1.),
        jnp.array([.1]), config=ScalarRootConfig(method='hybrid', maxiter=30))
    assert not bool(result.converged[0])
    assert int(result.candidate_count[0]) == 0


def test_candidate_selection_can_run_under_jit():
    config = ScalarRootConfig(method='hybrid', maxiter=120, ftol=1e-10, xtol=1e-10)
    result = jax.jit(lambda reference: qp.solve_quasiparticle_roots(
        lambda x: x-.04/(x+1.), reference, config=config))(jnp.array([0.]))
    np.testing.assert_allclose(result.roots, (-1+np.sqrt(1.16))/2, atol=1e-9)
    assert bool(result.converged[0])


@pytest.mark.parametrize('mode', ['linearized', 'evaluate_only'])
def test_noniterative_qp_modes_do_not_certify_a_negative_weight(mode):
    from gradscf.gw.g0w0 import _qp_loop
    energy, eta = jnp.array([-.5, .5]), .001
    weight = -2*jnp.pi*(1-eta**2)**2/(1+eta**2)
    wmn = jnp.zeros((1, 2, 2)).at[0, 0, 0].set(weight)
    result = _qp_loop(mo_energy=energy, b_mn=jnp.zeros((1, 2, 2)),
        channels=((energy, jnp.zeros((1, 1, 1)), 2.),), wmn=wmn,
        freqs=jnp.array([1.]), wts=jnp.array([1.]), ef=0., eta=eta,
        delta_v=jnp.zeros(2), nocc=1, orbs=(0,), diff_mode='implicit',
        return_diagnostics=True, **{mode: True})
    assert abs(float(result[4][0])) < 1e-12  # A zero exists in both modes.
    assert float(result[5][0]) < 0
    assert not bool(result[3])
