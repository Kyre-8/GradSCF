"""BSE forwards the same checked response configuration to TDA and full RPA."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import bse
from gradscf.solvers import LinearSolverConfig


@pytest.mark.parametrize("tda", [True, False])
def test_bse_response_config_and_facade_invalidation(tda):
    energy = jnp.array([-.8, .35, .7])
    screening = energy+jnp.array([-.1, -.02, .03])
    factors = jnp.array([[[.2, .08, .03], [.08, .13, .04], [.03, .04, .15]],
                         [[.1, .03, -.04], [.03, .09, .02], [-.04, .02, .12]]])
    dipole = jnp.ones((3, 3, 3))*.2
    space = bse.make_bse_space(3, 1)
    linear = LinearSolverConfig(method="direct", rtol=1e-11, max_dense=2)
    config = bse.BSEConfig(tda=tda, solver="dense", nroots=2, response_config=linear)
    def loss(t):
        out = bse.run_bse(energy, screening, factors*(1+.1*t), space, config=config)
        return bse.oscillator_strengths(out, dipole, space).sum()
    derivative = jax.jit(jax.grad(loss))(0.)
    np.testing.assert_allclose(derivative, (loss(1e-4)-loss(-1e-4))/2e-4, atol=2e-9, rtol=0)
    ref = bse.BSEReference(energy, screening, factors, 1, dipole_mo=dipole)
    calc = bse.BSE(ref, tda=tda, solver="dense", nroots=2, response_config=linear).run()
    np.testing.assert_allclose(calc.oscillator_strength().sum(), loss(0.), atol=1e-12, rtol=0)
    calc.response_config = LinearSolverConfig(method="gmres")
    with pytest.raises(RuntimeError, match="changed"):
        calc.oscillator_strength()


def test_bse_rejects_invalid_response_config():
    with pytest.raises(TypeError, match="response_config"):
        bse.BSEConfig(response_config="direct")
