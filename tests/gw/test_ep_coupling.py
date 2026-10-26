"""Fixed-phonon Fan/DW kernels: independent poles, covariance, and AD."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _case(nw=4):
    from gradscf.gw.ep_coupling import PhononModel
    from gradscf.gw.matsubara import matsubara_grid, _reference
    grid = matsubara_grid(nw=nw, beta=13.0)
    fock = jnp.diag(jnp.array([-0.4, 0.7]))
    model = PhononModel(jnp.array([0.002, 0.18]),
                        jnp.array([[[0.11, 0.08], [0.08, -0.05]],
                                   [[0.03, -0.04], [-0.04, 0.06]]]))
    return model, fock, grid, _reference(fock, 0.1, grid)[-1]


def _pole_oracle(model, energies, mu, beta, z):
    result = np.zeros((len(z), len(energies), len(energies)), complex)
    for l, omega in enumerate(np.asarray(model.energies)):
        bose = 1 / np.expm1(beta * omega)
        for a, energy in enumerate(energies):
            fermi = 1 / (1 + np.exp(beta * (energy - mu)))
            weight = ((1 + bose - fermi) / (z - (energy - mu) - omega)
                      + (bose + fermi) / (z - (energy - mu) + omega))
            vertex = np.asarray(model.couplings)[l, :, a]
            result += weight[:, None, None] * np.outer(vertex, vertex.conj())
    return result


def test_coarse_grid_fan_matches_independent_soft_phonon_poles():
    from gradscf.gw.ep_coupling import fan_self_energy
    model, fock, grid, green_tau = _case()
    result = fan_self_energy(green_tau, fock, 0.1, model, grid)
    reference = _pole_oracle(model, [-0.4, 0.7], 0.1, grid.beta, 1j * np.asarray(grid.fermion))
    np.testing.assert_allclose(result['sigma_iw'], reference, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result['sigma_iw'], result['sigma_iw'][::-1].conj(), atol=1e-13)
    high = _pole_oracle(model, [-0.4, 0.7], 0.1, grid.beta, np.array([1e8j]))
    np.testing.assert_allclose(high[0] * 1e8j, result['sigma_moment'], rtol=2e-8, atol=1e-9)


def test_phonon_propagators_and_debye_waller_zero_temperature_limit():
    from gradscf.gw.ep_coupling import PhononModel, phonon_propagator_iw, phonon_propagator_tau, bose_occupation, debye_waller
    omega = jnp.array([0.2, 0.7])
    beta = 1e5
    np.testing.assert_allclose(bose_occupation(omega, beta), 0, atol=1e-100)
    np.testing.assert_allclose(phonon_propagator_iw(omega, jnp.array([0.0, 0.4])),
                               -2 * omega / (jnp.array([0.0, 0.4])[:, None]**2 + omega**2))
    tau = jnp.array([0.0, 0.2, beta - 0.2, beta])
    np.testing.assert_allclose(phonon_propagator_tau(omega, tau, beta)[1], -jnp.exp(-0.2 * omega))
    np.testing.assert_allclose(phonon_propagator_tau(omega, tau, beta), phonon_propagator_tau(omega, beta - tau, beta))
    quadratic = jnp.zeros((2, 2, 1, 1)).at[0, 0, 0, 0].set(0.6).at[1, 1, 0, 0].set(-0.2)
    model = PhononModel(omega, jnp.ones((2, 1, 1)), quadratic)
    np.testing.assert_allclose(debye_waller(model, beta), [[0.2]], atol=1e-14)
    np.testing.assert_allclose(debye_waller(PhononModel(omega, model.couplings), beta), [[0.0]])


@pytest.mark.parametrize('omega', [0.0, -0.2, np.inf, np.nan])
def test_invalid_mode_energy_rejected(omega):
    from gradscf.gw.ep_coupling import PhononModel, validate_model
    with pytest.raises(ValueError, match='positive'):
        validate_model(PhononModel(jnp.array([omega]), jnp.ones((1, 1, 1))))


def test_model_validation_under_jit_and_shapes():
    from gradscf.gw.ep_coupling import PhononModel, validate_model
    def checked(omega):
        model = PhononModel(jnp.reshape(omega, (1,)), jnp.ones((1, 1, 1)))
        validate_model(model, norb=1)
        return model.energies.sum()
    np.testing.assert_allclose(jax.jit(checked)(0.1), 0.1)
    with pytest.raises(Exception, match='positive'):
        jax.jit(checked)(-0.1).block_until_ready()
    with pytest.raises(ValueError, match='shape'):
        validate_model(PhononModel(jnp.ones(2), jnp.ones((1, 1, 1))))
    with pytest.raises(ValueError, match='Hermitian'):
        validate_model(PhononModel(jnp.ones(1), jnp.array([[[0.0, 1.0], [0.0, 0.0]]])))
    with pytest.raises(ValueError, match='real'):
        validate_model(PhononModel(jnp.ones(1), jnp.ones((1, 1, 1), complex)))


def test_fan_uses_dressed_green_and_is_basis_covariant():
    from gradscf.gw.ep_coupling import PhononModel, fan_self_energy, phonon_propagator_tau
    model, fock, grid, green_tau = _case(12)
    perturbation = jnp.sin(jnp.pi * grid.tau / grid.beta)[:, None, None] * jnp.array([[0.03, -0.02], [-0.02, 0.01]])
    bare = fan_self_energy(green_tau, fock, 0.1, model, grid)
    result = fan_self_energy(green_tau + perturbation, fock, 0.1, model, grid)
    direct_tau = -jnp.einsum('lik,tkj,ljm,tl->tim', model.couplings, perturbation, model.couplings,
                             phonon_propagator_tau(model.energies, grid.tau, grid.beta))
    correction = grid.beta / len(grid.tau) * jnp.einsum('wt,tij->wij', jnp.exp(1j * grid.fermion[:, None] * grid.tau), direct_tau)
    np.testing.assert_allclose(result['sigma_iw'] - bare['sigma_iw'], correction, atol=1e-13)
    rotation = jnp.array([[0.8, -0.6], [0.6, 0.8]])
    rotate = lambda a: rotation.T @ a @ rotation
    transformed = PhononModel(model.energies, rotate(model.couplings))
    rotated = fan_self_energy(rotate(green_tau + perturbation), rotate(fock), 0.1, transformed, grid)
    np.testing.assert_allclose(rotated['sigma_iw'], rotate(result['sigma_iw']), atol=1e-12)
    np.testing.assert_allclose(rotated['sigma_moment'], rotate(result['sigma_moment']), atol=1e-12)


@pytest.mark.parametrize('variable', ['energies', 'couplings'])
def test_fan_jit_autodiff_matches_finite_difference(variable):
    from gradscf.gw.ep_coupling import PhononModel, fan_self_energy, validate_model
    model, fock, grid, green_tau = _case()
    def loss(scale):
        scaled = PhononModel(model.energies * (scale if variable == 'energies' else 1),
                             model.couplings * (scale if variable == 'couplings' else 1))
        validate_model(scaled)
        sigma = fan_self_energy(green_tau, fock, 0.1, scaled, grid)['sigma_iw']
        return jnp.sum(jnp.abs(sigma)**2)
    step = 1e-5
    finite = (loss(1.0 + step) - loss(1.0 - step)) / (2 * step)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(1.0), finite, rtol=1e-7, atol=1e-8)


def test_periodic_complex_fan_contraction_and_local_gauge_covariance():
    from gradscf.gw.ep_coupling import periodic_fan_self_energy_tau
    rng = np.random.default_rng(791)
    green = rng.normal(size=(4, 3, 2, 2)) + 1j * rng.normal(size=(4, 3, 2, 2))
    coupling = rng.normal(size=(2, 3, 2, 2, 2)) + 1j * rng.normal(size=(2, 3, 2, 2, 2))
    d_tau = -rng.uniform(size=(4, 2, 2))
    mapping = np.array([[1, 2, 0], [2, 0, 1]])
    weights = np.array([0.3, 0.7])
    expected = np.zeros_like(green)
    for t in range(4):
        for k in range(3):
            for q in range(2):
                for mode in range(2):
                    vertex = coupling[q, k, mode]
                    expected[t, k] -= weights[q] * d_tau[t, q, mode] * vertex @ green[t, mapping[q, k]] @ vertex.conj().T
    result = periodic_fan_self_energy_tau(green, coupling, d_tau, mapping, weights)
    np.testing.assert_allclose(result, expected, atol=1e-13)
    phases = np.exp(1j * rng.uniform(size=(3, 2)))
    rotate = lambda value: phases.conj()[None, :, :, None] * value * phases[None, :, None, :]
    rotated_g = phases.conj()[None, :, None, :, None] * coupling * phases[mapping][:, :, None, None, :]
    transformed = jax.jit(periodic_fan_self_energy_tau)(rotate(green), rotated_g, d_tau, mapping, weights)
    np.testing.assert_allclose(transformed, rotate(result), atol=1e-13)


@pytest.mark.parametrize('mapping,weights,message', [
    (np.array([[0, 2]]), np.array([1.0]), 'mapping'),
    (np.array([[0.0, 1.0]]), np.array([1.0]), 'integer'),
    (np.array([[0, 1]]), np.array([-1.0]), 'weights'),
    (np.array([[0, 1]]), np.array([0.4]), 'weights'),
])
def test_periodic_mapping_and_quadrature_validation(mapping, weights, message):
    from gradscf.gw.ep_coupling import periodic_fan_self_energy_tau
    with pytest.raises(ValueError, match=message):
        periodic_fan_self_energy_tau(jnp.ones((3, 2, 1, 1)), jnp.ones((1, 2, 1, 1, 1)),
                                      jnp.ones((3, 1, 1)), mapping, weights)


def test_molecular_fan_rejects_complex_reference_frame():
    from gradscf.gw.ep_coupling import PhononModel, fan_self_energy
    model, fock, grid, green = _case()
    complex_model = PhononModel(model.energies, model.couplings.astype(complex))
    with pytest.raises(ValueError, match='real'):
        fan_self_energy(green, fock, 0.1, complex_model, grid)
    with pytest.raises(ValueError, match='real'):
        fan_self_energy(green, fock.astype(complex), 0.1, model, grid)


def test_nonreference_green_converges_to_independent_poles():
    from gradscf.gw.ep_coupling import PhononModel, fan_self_energy
    from gradscf.gw.matsubara import _reference
    fock1 = jnp.array([[-0.6, 0.1], [0.1, 0.6]])
    fock2 = jnp.array([[-0.2, -0.06], [-0.06, 0.9]])
    fock = 0.4 * fock1 + 0.6 * fock2
    errors = []
    for nw in (24, 48, 96):
        model, _, grid, _ = _case(nw)
        green = 0.4 * _reference(fock1, 0.1, grid)[-1] + 0.6 * _reference(fock2, 0.1, grid)[-1]
        result = fan_self_energy(green, fock, 0.1, model, grid)['sigma_iw']
        exact = np.zeros_like(result)
        for fraction, matrix in ((0.4, fock1), (0.6, fock2)):
            energies, coeff = np.linalg.eigh(matrix)
            rotated = PhononModel(model.energies, coeff.T @ model.couplings @ coeff)
            poles = _pole_oracle(rotated, energies, 0.1, grid.beta, 1j * np.asarray(grid.fermion))
            exact += fraction * (coeff @ poles @ coeff.T)
        errors.append(np.max(np.abs(result[nw:nw+3] - exact[nw:nw+3])))
    assert errors[2] < errors[1] / 3.5 < errors[0] / 12.25
    assert errors[-1] < 2e-5


def test_list_model_inputs_match_arrays_and_invalid_beta_rejected():
    from gradscf.gw.ep_coupling import PhononModel, validate_model, fan_self_energy, bose_occupation, phonon_propagator_tau
    model, fock, grid, green = _case()
    lists = PhononModel(model.energies.tolist(), model.couplings.tolist())
    validate_model(lists)
    expected = fan_self_energy(green, fock, 0.1, model, grid)
    actual = fan_self_energy(green, fock, 0.1, lists, grid)
    np.testing.assert_allclose(actual['sigma_iw'], expected['sigma_iw'])
    for beta in (0.0, -1.0, np.inf):
        with pytest.raises(ValueError, match='beta'):
            bose_occupation(model.energies, beta)
        with pytest.raises(ValueError, match='beta'):
            phonon_propagator_tau(model.energies, grid.tau, beta)
