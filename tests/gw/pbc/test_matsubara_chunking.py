"""Time chunks preserve each complex contraction and the complete GW map."""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from test_matsubara import _model


def test_multiple_time_chunks_with_remainder_preserve_complex_bubble_and_sigma():
    module = importlib.import_module('gradscf.gw.pbc.matsubara')
    model = _model()
    value = jnp.arange(2*11*2*2, dtype=jnp.float64).reshape(2, 11, 2, 2)/100
    green = value+1j*jnp.sin(value*7)
    screened = tuple(jnp.cos(value[0]*(q+1))+1j*jnp.sin(value[1]*(q+2))
                     for q in range(2))
    bubble = module._bubble(green, model.factors, model.table)
    sigma = module._sigma_tau(green, model.factors, model.inverse, screened)
    for chunk in (1, 4, 11, 20):
        blocked = module._bubble(green, model.factors, model.table, time_batch_size=chunk)
        for actual, expected in zip(blocked, bubble):
            np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=2e-13)
        blocked_sigma = module._sigma_tau(green, model.factors, model.inverse,
                                          screened, time_batch_size=chunk)
        np.testing.assert_allclose(blocked_sigma, sigma, atol=3e-14, rtol=2e-13)
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match='batch|integer'):
            module._bubble(green, model.factors, model.table, time_batch_size=invalid)


def test_chunked_streaming_full_gw_matches_unchunked_dense_diagnostics():
    module = importlib.import_module('gradscf.gw.pbc.matsubara')
    model = _model()
    grid = module.matsubara_grid(nw=8, beta=15., time_oversampling=4,
                                 time_quadrature='gauss')
    fock = model.reference_fock.astype(jnp.complex128)
    coupling = jnp.array([[.27, .03+.05j], [.03-.05j, .21]])
    bath = jnp.array([[.3, -.04+.08j], [-.04-.08j, -.2]])
    resolvent = jnp.linalg.inv(1j*grid.fermion[:, None, None]*jnp.eye(2)-bath)
    sigma = jnp.broadcast_to(coupling@resolvent@coupling.conj().T, (2, 16, 2, 2))
    moment = jnp.broadcast_to(coupling@coupling.conj().T, (2, 2, 2))
    assert float(jnp.max(jnp.abs(fock@moment-moment@fock))) > 1e-3
    green, density = module.dyson_green_and_density(fock, sigma, .03, grid, moment)
    dense = module.gw_matsubara_step(green, fock, .03, model, grid, moment,
                                    density=density, time_batch_size=None)
    streamed = module.gw_matsubara_step(green, fock, .03, model, grid, moment,
        density=density, time_diagnostics=False, time_batch_size=7)
    for key in ('polarizability_inu', 'screened_inu', 'polarizability_second_moment',
                'screened_second_moment'):
        for actual, expected in zip(streamed[key], dense[key]):
            np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=2e-11)
    for key in ('sigma_iw', 'sigma_moment'):
        np.testing.assert_allclose(streamed[key], dense[key], atol=3e-12, rtol=2e-11)
