from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('xc', [None, 'pbe'])
def test_periodic_ofdft_response_and_energy_series(xc):
    from gradscf import ofdft
    from gradscf.dfpt import ofdft_problem
    data = ofdft.periodic_inputs(jnp.eye(3)*5., (3,3,3), nelectron=2.)
    probe = jnp.cos(2*jnp.pi*data.coordinates[:,0]/5.)
    data = replace(data, external_potential=.1*probe)
    cfg = ofdft.OFDFTConfig(xc=None, tolerance=1e-10, maxiter=400)
    xc_fn = None
    if xc == 'pbe':
        from test_pbe import pbe_energy_density
        def xc_fn(params, features):
            rho = jnp.stack([features.rho/2]*2)
            grad = jnp.stack([features.grad_rho.T/2]*2)
            return jnp.sum(features.weights*pbe_energy_density(rho, grad))
    result = ofdft.run_ofdft(data, config=cfg, xc_energy_fn=xc_fn)
    assert result.converged
    problem = ofdft_problem(data, result, config=cfg, xc_energy_fn=xc_fn)
    response = problem.response(potential=probe)
    assert response.converged
    step = 1e-4
    plus = ofdft.run_ofdft(replace(data, external_potential=data.external_potential+step*probe), config=cfg, xc_energy_fn=xc_fn)
    minus = ofdft.run_ofdft(replace(data, external_potential=data.external_potential-step*probe), config=cfg, xc_energy_fn=xc_fn)
    assert plus.converged and minus.converged
    np.testing.assert_allclose(response.density, (plus.density-minus.density)/(2*step), atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(response.values['electron_number'], 0., atol=1e-10)
    series = problem.series(potential=probe, order=2)
    assert series.converged
    np.testing.assert_allclose(series.energy[0], jnp.sum(data.weights*result.density*probe), atol=1e-9, rtol=0)
    np.testing.assert_allclose(2*series.energy[1], jnp.sum(data.weights*response.density*probe), atol=1e-8, rtol=0)


def test_molecular_gaussian_ofdft_response():
    from gradscf import ofdft, gto, dfpt
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr')
    method = ofdft.OFDFT(mol, kinetic='tfvw', xc=None, grids_level=0, tolerance=1e-9).run()
    assert method.converged
    data, config = method.inputs, method.config
    b = jnp.array([[.2, .1], [.1, -.1]])
    problem = dfpt.DFPT(method)
    out = problem.response(potential=b)
    assert out.converged
    def solve(t):
        return ofdft.run_ofdft(replace(data, external_potential=data.external_potential+t*b),
                                kinetic=method.kinetic, config=config)
    plus, minus = solve(1e-4), solve(-1e-4)
    assert plus.converged and minus.converged
    np.testing.assert_allclose(out.density, (plus.density-minus.density)/2e-4, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(out.values['electron_number'], 0., atol=1e-9)
    series = problem.series(potential=b, order=2)
    assert series.converged
    np.testing.assert_allclose(series.density[0], out.density, atol=1e-8)


def test_ofdft_number_and_volume_response():
    from gradscf import ofdft, dfpt
    make_inputs = lambda length, number: ofdft.periodic_inputs(jnp.eye(3)*length, (3,3,3), nelectron=number)
    data = make_inputs(5., 2.)
    cfg = ofdft.OFDFTConfig(xc=None, tolerance=1e-10)
    ground = ofdft.run_ofdft(data, config=cfg)
    p = dfpt.ofdft_problem(data, ground, config=cfg)
    for direction, expected_count, expected_density in [((0., 1.), 1., 1/125), ((1., 0.), 0., -6/625)]:
        tangent = jax.jvp(make_inputs, (5., 2.), direction)[1]
        out = p.response((tangent, {}, {}))
        assert out.converged
        np.testing.assert_allclose(out.values['electron_number'], expected_count, atol=1e-10)
        np.testing.assert_allclose(out.density, expected_density, atol=1e-10)


def test_periodic_gaussian_ofdft_response():
    from gradscf import integrals, ofdft, dfpt
    top, pars = integrals.prepare_basis('H 1 1 1; H 3 1 1', 'sto-3g', unit='Bohr')
    data = ofdft.periodic_gaussian_inputs(
        ofdft.periodic_inputs(jnp.eye(3)*5., (7,7,7), nelectron=2.), top, pars)
    probe = jnp.cos(2*jnp.pi*data.coordinates[:,0]/5.)
    data = replace(data, external_potential=.2*probe)
    cfg = ofdft.OFDFTConfig(xc=None, tolerance=1e-9)
    ground = ofdft.run_ofdft(data, config=cfg)
    assert ground.converged
    out = dfpt.ofdft_problem(data, ground, config=cfg).response(potential=probe)
    assert out.converged
    def solve(t):
        return ofdft.run_ofdft(replace(data, external_potential=data.external_potential+t*probe), config=cfg)
    plus, minus = solve(1e-4), solve(-1e-4)
    assert plus.converged and minus.converged
    np.testing.assert_allclose(out.density, (plus.density-minus.density)/2e-4, atol=2e-7, rtol=2e-5)
