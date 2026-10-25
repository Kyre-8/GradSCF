"""Opt-in CUDA coverage for staged MP results and their nested response."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import mp
from test_series_staging import model


@pytest.mark.skipif(os.environ.get("GRADSCF_TEST_GPU") != "1", reason="Opt-in CUDA gate")
def test_series_gpu_outputs_and_nested_response():
    devices = jax.devices("gpu")
    assert jax.default_backend() == "gpu" and len(devices) == 1
    h, g, no = model()
    mean_field = jnp.diag(jnp.linspace(-1., 1., h.shape[0]))-h
    cfg = mp.MPConfig(order=4, algorithm="tensor", with_coefficients=True)
    evaluate = lambda x: mp.run_mp(h+(1-x)*mean_field, x*g, nocc=no, config=cfg)
    energy = lambda x: evaluate(x).correlation_energy

    def bundle(x):
        value, tangent = jax.jvp(energy, (x,), (jnp.ones_like(x),))
        _, pullback = jax.vjp(energy, x)
        curvature = jax.jvp(jax.grad(energy), (x,), (.2*jnp.ones_like(x),))[1]
        return evaluate(x), value, tangent, pullback(jnp.ones_like(value))[0], curvature

    result, value, tangent, adjoint, curvature = jax.jit(jax.jit(bundle))(jnp.asarray(1.))
    for leaf in jax.tree.leaves((result, value, tangent, adjoint, curvature)):
        assert leaf.devices() == {devices[0]}
    assert result.valid and result.t2 is not None and result.wavefunction_coefficients is not None
    for leaf in jax.tree.leaves((result.corrections, value, result.t2,
                                result.wavefunction_coefficients, tangent, adjoint, curvature)):
        assert np.isfinite(np.asarray(leaf)).all()
    # Frozen CPU reference of this deterministic three-orbital fixture, Hartree.
    expected = np.array([-2.0311098517910836e-6, 6.1549725205459445e-9,
                         -2.4267943368525522e-11])
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(result.corrections, expected, atol=2e-14, rtol=1e-9,
                               equal_nan=False)
    orders = np.arange(2, 5)
    np.testing.assert_allclose([tangent, adjoint], np.sum(orders*expected), atol=2e-14,
                               equal_nan=False)
    np.testing.assert_allclose(curvature, .2*np.sum(orders*(orders-1)*expected), atol=2e-14,
                               equal_nan=False)
