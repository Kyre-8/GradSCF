"""Streaming retention must leave the full complex GW map unchanged."""
import importlib
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from test_matsubara import _model


def test_q_streaming_matches_dense_noncommuting_complex_dressed_map():
    module = importlib.import_module('gradscf.gw.pbc.matsubara')
    model = _model()
    grid = module.matsubara_grid(nw=8, beta=15., time_oversampling=4,
                                 time_quadrature='gauss')
    fock = jnp.array([[[-.7, .14+.21j], [.14-.21j, .6]],
                      [[-.5, -.11+.08j], [-.11-.08j, .9]]])
    coupling = jnp.array([[[.31, .03+.04j], [.03-.04j, .21]],
                          [[.24, -.02+.07j], [-.02-.07j, .35]]])
    bath = jnp.array([[[.4, -.12+.09j], [-.12-.09j, -.3]],
                     [[-.2, .08+.13j], [.08-.13j, .5]]])
    eye = jnp.eye(2, dtype=jnp.complex128)
    resolvent = jnp.linalg.inv(1j*grid.fermion[None, :, None, None]*eye-bath[:, None])
    sigma = coupling[:, None]@resolvent@coupling[:, None].conj().swapaxes(-1, -2)
    moment = coupling@coupling.conj().swapaxes(-1, -2)
    assert float(jnp.max(jnp.abs(fock@moment-moment@fock))) > 1e-3
    green, density = module.dyson_green_and_density(fock, sigma, .03, grid, moment)
    dense = module.gw_matsubara_step(green, fock, .03, model, grid, moment,
                                    density=density, time_diagnostics=True)
    streamed = module.gw_matsubara_step(green, fock, .03, model, grid, moment,
                                       density=density, time_diagnostics=False)
    for key in ('polarizability_inu', 'screened_inu', 'polarizability_second_moment',
                'screened_second_moment'):
        for expected, actual in zip(dense[key], streamed[key]):
            np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=2e-11)
    for key in ('sigma_iw', 'sigma_moment'):
        np.testing.assert_allclose(streamed[key], dense[key], atol=3e-12, rtol=2e-11)
    assert streamed['time_diagnostics_retained'] is False
    assert dense['time_diagnostics_retained'] is True
    for key in ('green_tau', 'screened_tau', 'screened_zero', 'screened_beta', 'sigma_tau'):
        assert key in dense and key not in streamed
    assert np.max(abs(np.asarray(density).imag)) > 1e-3


def test_scgw_streamed_and_dense_diagnostics_have_the_same_fixed_point(monkeypatch):
    module = importlib.import_module('gradscf.gw.pbc.scgw')
    original = module.gw_matsubara_step
    model = _model()
    options = dict(nw=12, beta=12., max_iter=60, tol=2e-8, particle_tol=2e-10,
                   mixing=.6, time_quadrature='gauss', time_oversampling=4)
    streamed = module.scgw_matsubara_kpoints(model, **options)

    def dense(*args, **kwargs):
        kwargs['time_diagnostics'] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'gw_matsubara_step', dense)
    retained = module.scgw_matsubara_kpoints(model, **options)
    for key in ('self_energy_iw', 'sigma_moment', 'fock_mo', 'density_mo', 'chemical_potential'):
        np.testing.assert_allclose(getattr(streamed, key), getattr(retained, key),
                                   atol=4e-8, rtol=0)
    for result in (streamed, retained):
        assert float(result.fixed_point_residual) < 2e-8
        assert abs(float(result.particle_number_error)) <= 2e-10
        assert result.n_iter > 1


def test_scgw_releases_previous_full_step_before_evaluating_the_next_map(monkeypatch):
    module = importlib.import_module('gradscf.gw.pbc.scgw')
    original = module._physical_map
    probes = []

    class RetentionProbe:
        pass

    def checked(*args, **kwargs):
        if probes:
            assert probes[-1]() is None, 'Previous full step remains live during the next map'
        step = original(*args, **kwargs)
        probe = RetentionProbe()
        probes.append(weakref.ref(probe))
        step['_retention_probe'] = probe
        return step

    monkeypatch.setattr(module, '_physical_map', checked)
    result = module.scgw_matsubara_kpoints(_model(), nw=8, beta=12.,
        max_iter=60, tol=1e-7, particle_tol=1e-9, mixing=.6)
    assert result.n_iter > 1 and len(probes) == result.n_iter
