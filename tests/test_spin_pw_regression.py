"""Polarized PW energy and potential after upstream parameter binding repair."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_pw_energy_potential_spin_symmetry_and_jit_match_reference():
    pytest.importorskip("jax_xc")
    pytest.importorskip("pyscf")
    from pyscf.dft import libxc
    from gradscf.dft.xc._jax_xc import eval_jax_xc_energy_density_from_unrestricted_density_gradients

    def energy(rho):
        return eval_jax_xc_energy_density_from_unrestricted_density_gradients(
            "lda_c_pw", rho[0], rho[1], jnp.zeros(3), jnp.zeros(3))

    evaluate = jax.jit(jax.value_and_grad(energy))
    for total in (.01, .1, 1.):
        for zeta in (-.8, -.2, 0., .2, .8, .999999):
            rho = jnp.array([total*(1+zeta)/2, total*(1-zeta)/2])
            value, potential = evaluate(rho)
            exc, vxc, _, _ = libxc.eval_xc("LDA_C_PW", np.asarray(rho)[:, None], spin=1, deriv=1)
            np.testing.assert_allclose(value, total*exc[0], atol=2e-12, rtol=2e-11)
            np.testing.assert_allclose(potential, vxc[0][0], atol=2e-9, rtol=2e-9)
            reverse, v_reverse = evaluate(rho[::-1])
            np.testing.assert_allclose(value, reverse, atol=1e-14)
            np.testing.assert_allclose(potential, v_reverse[::-1], atol=1e-12)
