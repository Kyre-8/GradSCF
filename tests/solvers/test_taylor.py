"""Root coefficients, independent of the DFPT physical adapters."""
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.solvers.nonlinear import root_taylor


def test_taylor_normalization_and_outer_ad():
    def coefficients(a):
        result = root_taylor(lambda x, s: x*x-a-s, jnp.sqrt(a)[None], order=3)
        return result.coefficients[:,0]
    np.testing.assert_allclose(jax.jit(coefficients)(1.), [.5, -.125, .0625], atol=1e-12)
    expected = [-.25, .1875, -.15625]
    np.testing.assert_allclose(jax.jacfwd(coefficients)(1.), expected, atol=1e-12)
    np.testing.assert_allclose(jax.jacrev(coefficients)(1.), expected, atol=1e-12)


def test_failed_taylor_linear_solve_masks_coefficients():
    result = root_taylor(lambda x, s: jnp.zeros_like(x)+s, jnp.zeros(1), order=1)
    assert not result.converged
    assert np.isnan(result.coefficients).all()
