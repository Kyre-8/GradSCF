"""Complex two-k finite-temperature matrix GW controls (remote numerics)."""
from dataclasses import replace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _model(scale=.25, *, deep=False):
    source = importlib.import_module('gradscf.gw.pbc.source')
    energy = jnp.array([[-.8, .5], [-.6, .7]])
    if deep:
        energy = energy.at[:, 0].add(-15.)
    zero = jnp.array([[[[.4, .13+.08j], [.13-.08j, .2]],
                       [[.2, -.04+.1j], [-.04-.1j, .3]]],
                      [[[.3, .11-.06j], [.11+.06j, .25]],
                       [[.1, .06+.08j], [.06-.08j, .18]]]])
    cross = jnp.array([[[.31+.04j, .17-.09j], [.07+.12j, .22-.03j]],
                       [[.12-.08j, .04+.11j], [.16-.03j, .19+.07j]]])
    factors = (scale*zero, scale*jnp.stack((cross, cross.conj().swapaxes(-1, -2))))
    table = np.array([[0, 1], [1, 0]])
    fock = jax.vmap(jnp.diag)(energy)
    model = source.PeriodicGWModel(
        energy=energy, coeff=jnp.tile(jnp.eye(2, dtype=jnp.complex128), (2, 1, 1)),
        hcore=fock, reference_potential=jnp.zeros_like(fock),
        reference_fock=fock, nocc=1, kpoints=np.array([[0., 0., 0.], [-np.pi, 0., 0.]]),
        lattice=np.eye(3), table=table, inverse=np.argsort(table, axis=0),
        factors=factors, madelung=0., metadata={'fc': False},
    )
    density = jnp.tile(jnp.diag(jnp.array([1., 0.])), (2, 1, 1))
    hartree, exchange = source.hartree_exchange(model, density)
    return replace(model, hcore=fock-hartree-exchange, reference_potential=exchange)


def _module():
    return importlib.import_module('gradscf.gw.pbc.matsubara')


def _fermi(x):
    small = np.exp(-np.abs(x))
    return np.where(x >= 0, small/(1+small), 1/(1+small))


def _bose(x):
    small = np.exp(-abs(x))
    return small/(-np.expm1(-abs(x))) if x > 0 else -1/(-np.expm1(-abs(x)))


def _lehmann_pi(model, mu, grid):
    energy = np.asarray(model.energy)
    occupation = _fermi(grid.beta*(energy-mu))
    blocks = []
    for q, factor in enumerate(model.factors):
        b = np.asarray(factor)
        pi = np.zeros((len(grid.boson), b.shape[1], b.shape[1]), complex)
        for k in range(len(energy)):
            p = model.table[k, q]
            for a in range(energy.shape[1]):
                for c in range(energy.shape[1]):
                    delta = energy[k, a]-energy[p, c]
                    numerator = occupation[k, a]-occupation[p, c]
                    vertex = np.outer(b[k, :, a, c], b[k, :, a, c].conj())
                    for w, nu in enumerate(np.asarray(grid.boson)):
                        weight = (-grid.beta*occupation[k, a]*(1-occupation[k, a])
                                  if abs(delta) < 1e-12 and nu == 0
                                  else numerator/(1j*nu+delta) if abs(delta)+abs(nu) else 0.)
                        pi[w] += 2/len(energy)*weight*vertex
        blocks.append(pi)
    return tuple(blocks)


def _second_order_sigma(model, mu, grid):
    """Literal G0 Pi0 convolution, independent of the RPA pole machinery."""
    energy = np.asarray(model.energy)
    occupation = _fermi(grid.beta*(energy-mu))
    nk, nmo = energy.shape
    sigma = np.zeros((nk, len(grid.fermion), nmo, nmo), complex)
    for q, factor in enumerate(model.factors):
        b = np.asarray(factor)
        for k in range(nk):
            internal = model.inverse[k, q]
            for m in range(nmo):
                for l in range(nk):
                    partner = model.table[l, q]
                    for a in range(nmo):
                        for c in range(nmo):
                            pole = energy[partner, c]-energy[l, a]
                            difference = occupation[l, a]-occupation[partner, c]
                            amplitude = np.einsum('Pp,P,Q,Qr->pr',
                                b[internal, :, m, :].conj(), b[l, :, a, c],
                                b[l, :, a, c].conj(), b[internal, :, m, :])
                            if abs(pole) < 1e-12:
                                weight = occupation[l, a]*(1-occupation[l, a])/(
                                    1j*np.asarray(grid.fermion)-(energy[internal, m]-mu))
                                sigma[k] += 2/nk**2*weight[:, None, None]*amplitude
                                continue
                            if abs(difference) < 1e-15:
                                continue
                            boson = _bose(grid.beta*pole)
                            weight = (1-occupation[internal, m]+boson)/(
                                1j*np.asarray(grid.fermion)-(energy[internal, m]-mu)-pole)
                            sigma[k] += 2/nk**2*difference*weight[:, None, None]*amplitude
    return sigma


def test_complex_dyson_static_limit_preserves_imaginary_density():
    module = _module()
    grid = module.matsubara_grid(nw=8, beta=12.)
    fock = jnp.array([[[-.7, .14+.21j], [.14-.21j, .6]],
                      [[-.5, -.11+.08j], [-.11-.08j, .9]]])
    sigma = jnp.zeros((2, 16, 2, 2), dtype=jnp.complex128)
    green, density = module.dyson_green_and_density(fock, sigma, .03, grid)
    e, c = np.linalg.eigh(np.asarray(fock))
    expected = np.einsum('kai,ki,kbi->kab', c, 1/(1+np.exp(12*(e-.03))), c.conj())
    np.testing.assert_allclose(density, expected, atol=2e-13, rtol=0)
    assert np.max(abs(np.asarray(density).imag)) > .05
    np.testing.assert_allclose(green, np.asarray(green)[:, ::-1].conj().swapaxes(-1, -2), atol=2e-13)


@pytest.mark.parametrize('deep,beta', [(False, 12.), (True, 400.)])
def test_reference_bubble_is_literal_complex_lehmann_at_small_grid(deep, beta):
    module = _module()
    model = _model(deep=deep)
    grid = module.matsubara_grid(nw=8, beta=beta)
    mu = .03 if not deep else 0.
    sigma = jnp.zeros((2, 16, 2, 2), dtype=jnp.complex128)
    green, density = module.dyson_green_and_density(model.reference_fock, sigma, mu, grid)
    step = module.gw_matsubara_step(green, model.reference_fock, mu, model, grid, density=density)
    expected = _lehmann_pi(model, mu, grid)
    for actual, reference in zip(step['polarizability_inu'], expected):
        np.testing.assert_allclose(actual, reference, atol=4e-13, rtol=2e-12)
        np.testing.assert_allclose(actual, np.asarray(actual)[::-1].conj().swapaxes(-1, -2), atol=4e-13)
    assert np.max(abs(expected[0]-expected[0][::-1])) > 1e-7
    np.testing.assert_allclose(step['sigma_iw'], np.asarray(step['sigma_iw'])[:, ::-1].conj().swapaxes(-1, -2), atol=2e-11)


@pytest.mark.parametrize('deep,beta,nw', [(False, 12., 12), (True, 400., 64)])
def test_weak_interaction_self_energy_has_literal_second_order_orientation(deep, beta, nw):
    module = _module()
    # RPA-Pi is O(scale^4); its contribution to Sigma is O(scale^6).
    model = _model(scale=.002, deep=deep)
    grid = module.matsubara_grid(nw=nw, beta=beta)
    sigma = jnp.zeros((2, 2*nw, 2, 2), dtype=jnp.complex128)
    green, density = module.dyson_green_and_density(model.reference_fock, sigma, .03, grid)
    step = module.gw_matsubara_step(green, model.reference_fock, .03, model, grid, density=density)
    oracle = _second_order_sigma(model, .03, grid)
    np.testing.assert_allclose(step['sigma_iw'], oracle, atol=2e-19, rtol=2e-5)
    far = replace(grid, fermion=jnp.array([1e7]))
    oracle_moment = 1j*1e7*_second_order_sigma(model, .03, far)[:, 0]
    oracle_moment = .5*(oracle_moment+oracle_moment.conj().swapaxes(-1, -2))
    np.testing.assert_allclose(step['sigma_moment'], oracle_moment, atol=2e-19, rtol=2e-5)
    assert np.max(abs(oracle[..., 0, 1])) > 1e-16


def test_dressed_complex_bubble_converges_to_independent_two_pole_lehmann():
    module = _module()
    model = _model()
    beta, mu = 12., .03
    weights = np.array([.7, .3])
    offsets = np.array([.2, -.2*.7/.3])
    poles = np.asarray(model.energy)[..., None]+offsets
    occupations = _fermi(beta*(poles-mu))
    density = jax.vmap(jnp.diag)(jnp.asarray(np.sum(weights*occupations, axis=-1)))
    moment = jnp.tile(jnp.eye(2)[None], (2, 1, 1))*float(np.sum(weights*offsets**2))
    errors = []
    for nw in (16, 32):
        grid = module.matsubara_grid(nw=nw, beta=beta)
        diagonal = np.sum(weights/(1j*np.asarray(grid.fermion)[None, :, None, None]
                                  -(poles[:, None]-mu)), axis=-1)
        green = jax.vmap(jax.vmap(jnp.diag))(jnp.asarray(diagonal))
        step = module.gw_matsubara_step(green, model.reference_fock, mu, model, grid,
                                       density=density, sigma_moment=moment)
        expected = np.zeros_like(np.asarray(step['polarizability_inu'][0]))
        b = np.asarray(model.factors[0])
        for k in range(2):
            for a in range(2):
                for c in range(2):
                    vertex = np.outer(b[k, :, a, c], b[k, :, a, c].conj())
                    for r in range(2):
                        for s in range(2):
                            d = poles[k, a, r]-poles[k, c, s]
                            numerator = occupations[k, a, r]-occupations[k, c, s]
                            denominator = 1j*np.asarray(grid.boson)+d
                            zero = (abs(d) < 1e-12)&(np.asarray(grid.boson) == 0)
                            value = numerator/np.where(zero, 1., denominator)
                            value = np.where(zero, -beta*occupations[k, a, r]
                                             *(1-occupations[k, a, r]), value)
                            expected += weights[r]*weights[s]*value[:, None, None]*vertex
        errors.append(np.max(abs(np.asarray(step['polarizability_inu'][0])[nw-3:nw+4]
                                    -expected[nw-3:nw+4])))
        # The endpoint moment must retain both noncommuting density factors.
        expected_moment = np.zeros((2, 2, 2), complex)
        for q, factor in enumerate(model.factors):
            for k in range(2):
                internal = model.inverse[k, q]
                vertex = np.asarray(factor)[internal]
                expected_moment[k] -= .5*np.einsum('Pma,mn,Qnb,PQ->ab',
                    vertex.conj(), np.eye(2)-np.asarray(density)[internal], vertex,
                    np.asarray(step['screened_zero'][q]))
                expected_moment[k] -= .5*np.einsum('Pma,mn,Qnb,PQ->ab',
                    vertex.conj(), np.asarray(density)[internal], vertex,
                    np.asarray(step['screened_beta'][q]))
        np.testing.assert_allclose(step['sigma_moment'], expected_moment, atol=2e-13)
    assert errors[1] < errors[0]/3
    assert errors[1] < 2e-5


def test_complex_orbital_frame_covariance_of_bubble_sigma_and_density():
    module = _module()
    source = importlib.import_module('gradscf.gw.pbc.source')
    model = _model()
    angle = jnp.array([.31, -.24])
    phase = jnp.exp(1j*jnp.array([.7, -1.1]))
    rotation = jnp.stack((jnp.stack((jnp.cos(angle), -jnp.sin(angle)*phase.conj()), axis=-1),
                          jnp.stack((jnp.sin(angle)*phase, jnp.cos(angle)), axis=-1)), axis=-2)
    rotate = lambda value: rotation.conj().swapaxes(-1, -2)@value@rotation
    transformed = replace(model, coeff=model.coeff@rotation,
        hcore=rotate(model.hcore), reference_fock=rotate(model.reference_fock),
        reference_potential=rotate(model.reference_potential),
        factors=source.rotate_factors(model, rotation))
    grid = module.matsubara_grid(nw=12, beta=12.)
    sigma = jnp.zeros((2, 24, 2, 2), dtype=jnp.complex128)
    green, density = module.dyson_green_and_density(model.reference_fock, sigma, .03, grid)
    other_green, other_density = module.dyson_green_and_density(transformed.reference_fock, sigma, .03, grid)
    first = module.gw_matsubara_step(green, model.reference_fock, .03, model, grid, density=density)
    second = module.gw_matsubara_step(other_green, transformed.reference_fock, .03,
                                     transformed, grid, density=other_density)
    np.testing.assert_allclose(other_density, rotate(density), atol=2e-13, rtol=0)
    for left, right in zip(first['polarizability_inu'], second['polarizability_inu']):
        np.testing.assert_allclose(left, right, atol=3e-13, rtol=0)
    expected_sigma = rotation[:, None].conj().swapaxes(-1, -2)@first['sigma_iw']@rotation[:, None]
    np.testing.assert_allclose(second['sigma_iw'], expected_sigma, atol=3e-12, rtol=2e-11)
    np.testing.assert_allclose(second['sigma_moment'], rotate(first['sigma_moment']), atol=3e-13, rtol=0)


def test_periodic_scgw_zero_interaction_has_no_inferred_qp_bands():
    module = importlib.import_module('gradscf.gw.pbc.scgw')
    model = _model(scale=0.)
    result = module.scgw_matsubara_kpoints(model, nw=8, beta=12., tol=1e-10)
    assert bool(result.converged)
    assert result.mo_energy is None
    assert result.total_energy is None
    np.testing.assert_allclose(result.static_mo_energy, model.energy, atol=2e-12)
    np.testing.assert_allclose(result.self_energy_iw, 0., atol=0.)
    assert abs(float(result.particle_number_error)) < 1e-9
    assert float(result.fixed_point_residual) < 1e-10


def test_periodic_scgw_maps_one_returned_full_matrix_state_and_number():
    module = importlib.import_module('gradscf.gw.pbc.scgw')
    model = _model()
    result = module.scgw_matsubara_kpoints(model, nw=24, beta=12., max_iter=60,
                                         tol=2e-8, mixing=.6, particle_tol=2e-10)
    assert bool(result.converged)
    assert float(result.fixed_point_residual) < 2e-8
    assert abs(float(result.particle_number_error)) < 2e-10
    np.testing.assert_allclose(result.fock_mo, result.mapped_fock_mo, atol=2e-8, rtol=0)
    np.testing.assert_allclose(result.self_energy_iw, result.mapped_self_energy_iw, atol=2e-8, rtol=0)
    np.testing.assert_allclose(result.density_mo, np.asarray(result.density_mo).conj().swapaxes(-1, -2), atol=2e-13)
    assert np.max(abs(np.asarray(result.self_energy_iw)[..., 0, 1])) > 1e-7


def test_periodic_scgw_rejects_ad_and_unsupported_q0_branch():
    module = importlib.import_module('gradscf.gw.pbc.scgw')
    model = _model()
    with pytest.raises(NotImplementedError, match='differentiation|AD|eager'):
        jax.jvp(lambda x: module.scgw_matsubara_kpoints(
            replace(model, hcore=model.hcore*x), nw=8, beta=12.).density_mo,
            (jnp.array(1.),), (jnp.array(1.),))
    with pytest.raises(NotImplementedError, match='q0|fc|head'):
        module.scgw_matsubara_kpoints(replace(model, metadata={'fc': True}), nw=8, beta=12.)


def _padded_grid(module, factor):
    import inspect
    assert 'time_oversampling' in inspect.signature(module.matsubara_grid).parameters, \
        'Missing PBC time oversampling capability'
    return module.matsubara_grid(nw=6, beta=12., time_oversampling=factor)


def test_periodic_time_grid_padding_preserves_frequencies_and_resolves_boson_endpoints():
    module = _module()
    from gradscf.gw.matsubara import matsubara_grid as molecular_grid
    baseline = molecular_grid(nw=6, beta=12.)
    for factor in (1, 2, 4):
        grid = _padded_grid(module, factor)
        assert len(grid.tau) == 12*factor
        np.testing.assert_array_equal(grid.fermion, baseline.fermion)
        np.testing.assert_array_equal(grid.boson, baseline.boson)
        np.testing.assert_array_equal(grid.boson_weights,
            baseline.boson_weights if factor == 1 else np.ones(13))
    np.testing.assert_array_equal(_padded_grid(module, 1).tau, baseline.tau)
    for factor in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match='oversampling|integer'):
            _padded_grid(module, factor)


def _dressed_bath_oracles(model, poles, weights, mu, grid):
    """Literal full two-pole bubble and weak-coupling G Pi convolution."""
    occupation = _fermi(grid.beta*(poles-mu))
    nk, nmo, npole = poles.shape
    polarizability = []
    sigma = np.zeros((nk, len(grid.fermion), nmo, nmo), complex)
    for q, factor in enumerate(model.factors):
        b = np.asarray(factor)
        pi = np.zeros((len(grid.boson), b.shape[1], b.shape[1]), complex)
        for k in range(nk):
            internal = model.inverse[k, q]
            for left in range(nk):
                partner = model.table[left, q]
                for a in range(nmo):
                    for c in range(nmo):
                        outer = np.outer(b[left, :, a, c], b[left, :, a, c].conj())
                        for r in range(npole):
                            for s in range(npole):
                                pole = poles[partner, c, s]-poles[left, a, r]
                                delta_f = occupation[left, a, r]-occupation[partner, c, s]
                                zero = (abs(pole) < 1e-12)&(np.asarray(grid.boson) == 0)
                                quotient = delta_f/np.where(zero, 1., 1j*np.asarray(grid.boson)-pole)
                                thermal = occupation[left, a, r]*(1-occupation[left, a, r])
                                quotient = np.where(zero, -grid.beta*thermal, quotient)
                                if k == 0:
                                    pi += 2/nk*weights[r]*weights[s]*quotient[:, None, None]*outer
                                for m in range(nmo):
                                    amplitude = np.einsum('Pp,P,Q,Qv->pv',
                                        b[internal, :, m].conj(), b[left, :, a, c],
                                        b[left, :, a, c].conj(), b[internal, :, m])
                                    for t in range(npole):
                                        z = 1j*np.asarray(grid.fermion)-(poles[internal, m, t]-mu)
                                        if abs(pole) < 1e-12:
                                            value = thermal/z
                                        elif abs(delta_f) > 1e-15:
                                            value = delta_f*(1-occupation[internal, m, t]
                                                +_bose(grid.beta*pole))/(z-pole)
                                        else:
                                            continue
                                        sigma[k] += 2/nk**2*weights[r]*weights[s]*weights[t] \
                                            *value[:, None, None]*amplitude
        polarizability.append(pi)
    return tuple(polarizability), sigma


def test_time_padding_reduces_dressed_bath_bubble_and_self_energy_alias():
    module = _module()
    model = _model(scale=.002)
    mu, weights = .03, np.array([.7, .3])
    offsets = np.array([.2, -.2*.7/.3])
    poles = np.asarray(model.energy)[..., None]+offsets
    occupation = _fermi(12.*(poles-mu))
    density = jax.vmap(jnp.diag)(jnp.asarray(np.sum(weights*occupation, axis=-1)))
    moment = jnp.tile(jnp.eye(2)[None], (2, 1, 1))*float(np.sum(weights*offsets**2))
    errors = []
    for factor in (1, 2, 4):
        grid = _padded_grid(module, factor)
        diagonal = np.sum(weights/(1j*np.asarray(grid.fermion)[None, :, None, None]
                                  -(poles[:, None]-mu)), axis=-1)
        green = jax.vmap(jax.vmap(jnp.diag))(jnp.asarray(diagonal))
        step = module.gw_matsubara_step(green, model.reference_fock, mu, model, grid,
                                       density=density, sigma_moment=moment)
        pi, sigma = _dressed_bath_oracles(model, poles, weights, mu, grid)
        pi_error = np.linalg.norm(np.asarray(step['polarizability_inu'][0])-pi[0])/np.linalg.norm(pi[0])
        sigma_error = np.linalg.norm(np.asarray(step['sigma_iw'])-sigma)/np.linalg.norm(sigma)
        errors.append((pi_error, sigma_error))
    errors = np.asarray(errors)
    print('Fixed nw=6, beta=12 dressed-bath relative Pi/Sigma errors t1/t2/t4:', errors)
    assert np.all(errors[1] < .5*errors[0]), errors
    assert np.all(errors[2] <= 1.1*errors[1]), errors
