"""QP frequency clusters use local spectral resolution, not root xtol."""
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw.pbc.query import _solve_subspace_qp
from gradscf.solvers.nonlinear import ScalarRootConfig


@pytest.mark.parametrize('complex_frame', [False, True])
@pytest.mark.parametrize('delta', [2e-9, 0.])
def test_resolved_avoided_crossing_and_true_crossing_are_distinguished(delta, complex_frame):
    # H(w)=w0 I + a*(w-w0)*sigma_z + delta*sigma_x.  For delta>0
    # the two exact roots are w0 +/- delta/sqrt(1-a^2), with Z=1/(1-a^2).
    # Their spectral gap is much less than xtol but is numerically resolved.
    a, w0 = .4, .2
    frame = (jnp.array([[1., 1j], [1j, 1.]])/jnp.sqrt(2.)
             if complex_frame else jnp.eye(2, dtype=jnp.complex128))
    def correlation(omega):
        x = omega-w0
        matrix = jnp.array([[a*x, delta], [delta, -a*x]], dtype=jnp.complex128)
        return frame.conj().T@matrix@frame
    energy = jnp.full((2,), w0)
    zero = jnp.zeros((2, 2), dtype=jnp.complex128)
    config = ScalarRootConfig(method='hybrid', ftol=1e-13, xtol=1e-8, maxiter=100)
    result = _solve_subspace_qp(energy, zero, zero, correlation,
                                config=config, occupied=False)
    assert np.max(abs(np.asarray(result['residual']))) < config.ftol
    if delta:
        exact = w0+np.array([-1., 1.])*delta/np.sqrt(1-a*a)
        np.testing.assert_allclose(result['roots'], exact, atol=2e-13, rtol=0)
        assert np.asarray(result['converged']).all()
        assert np.asarray(result['branch_valid']).all()
        np.testing.assert_allclose(result['weight'], 1/(1-a*a), atol=3e-5, rtol=0)
    else:
        # A genuinely degenerate frequency with nonscalar PH'P remains
        # uncertified even though its scalar equations have positive slopes.
        assert not np.asarray(result['converged']).any()
        assert not np.asarray(result['branch_valid']).any()
        assert np.isnan(np.asarray(result['weight'])).all()


def test_remote_third_state_derivative_cannot_relax_a_crossing_certificate():
    # The third eigenvalue lies in [2,4] Ha, uniformly away from the crossing.
    # Its H'(0)=1e12 must not set the two-state derivative certification scale.
    energy = jnp.array([0., 0., 3.])
    zero = jnp.zeros((3, 3), dtype=jnp.complex128)
    def correlation(omega):
        return jnp.diag(jnp.array([.4*omega, -.4*omega, jnp.tanh(1e12*omega)]))
    config = ScalarRootConfig(method='hybrid', ftol=1e-13, xtol=1e-8, maxiter=100)
    result = _solve_subspace_qp(energy, zero, zero, correlation,
                                config=config, occupied=False)
    assert np.max(abs(np.asarray(result['residual']))) < config.ftol
    np.testing.assert_allclose(result['roots'], [0., 0., 4.], atol=2e-13, rtol=0)
    assert not np.asarray(result['converged'])[:2].any()
    assert not np.asarray(result['branch_valid'])[:2].any()
    assert np.isnan(np.asarray(result['weight'])[:2]).all()
    assert bool(result['converged'][2]) and bool(result['branch_valid'][2])
    np.testing.assert_allclose(result['weight'][2], 1., atol=1e-13)


def test_true_crossing_with_finite_root_residual_is_not_an_eps_only_singleton():
    energy = jnp.zeros(2)
    zero = jnp.zeros((2, 2), dtype=jnp.complex128)
    # Common quadratic curvature lets the shared solver stop at a finite
    # residual under this deliberately coarse, explicit root budget. The
    # continuation target favors a nearby negative iterate over exact zero.
    # H(0) still crosses and H'(0) has eigenvalues +/- .4. No solver is mocked.
    initial = jnp.full((2,), -1e-5)
    config = ScalarRootConfig(method='hybrid', ftol=1e-7, xtol=1e-3, maxiter=100)
    result = _solve_subspace_qp(energy, zero, zero,
        lambda omega: jnp.diag(jnp.array([.4*omega, -.4*omega]))+omega**2*jnp.eye(2),
        config=config, occupied=False, initial_guess=initial)
    residual = abs(np.asarray(result['residual']))
    assert residual.min() > 1e-14 and residual.max() < config.ftol
    assert np.min(abs(np.asarray(result['roots']))) > 1e-14
    assert not np.asarray(result['converged']).any()
    assert not np.asarray(result['branch_valid']).any()
    assert np.isnan(np.asarray(result['weight'])).all()


def test_true_crossing_curvature_needs_a_conservative_finite_residual_guard():
    # For a negative approximate root -t, the Newton error estimate is
    # u=t*(c+t)/(c+2*t)<t. Here its curvature deficit exceeds eigensolve
    # roundoff, so a coefficient-one crossing band can miss the true crossing.
    # This checks a local guard; it is not a global nonlinear error bound.
    energy = jnp.zeros(2)
    zero = jnp.zeros((2, 2), dtype=jnp.complex128)
    initial = jnp.full((2,), -1e-2)
    config = ScalarRootConfig(method='hybrid', ftol=1e-3, xtol=1e-1, maxiter=100)
    result = _solve_subspace_qp(energy, zero, zero,
        lambda omega: jnp.diag(jnp.array([.4*omega, -.4*omega]))+omega**2*jnp.eye(2),
        config=config, occupied=False, initial_guess=initial)
    roots, residual = np.asarray(result['roots']), abs(np.asarray(result['residual']))
    assert (roots < -1e-12).all()
    assert residual.max() > 1e-8 and residual.max() < config.ftol
    assert not np.asarray(result['converged']).any()
    assert not np.asarray(result['branch_valid']).any()
    assert np.isnan(np.asarray(result['weight'])).all()
