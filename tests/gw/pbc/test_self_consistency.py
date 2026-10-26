"""Independent finite-k controls for complex periodic GW self-consistency.

Numerical execution belongs on the remote validation host. These tests use a
two-k Hamiltonian, not independent molecular calculations at each k point.
"""
from dataclasses import replace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _source():
    return importlib.import_module('gradscf.gw.pbc.source')


def _model(scale=.12, madelung=.015):
    source = _source()
    energy = jnp.array([[-.8, .5], [-.6, .7]])
    # q=0 is a Hermitian density vertex; q=pi joins the two k frames.
    zero = jnp.array([[[[.4, .13+.08j], [.13-.08j, .2]],
                       [[.2, -.04+.1j], [-.04-.1j, .3]]],
                      [[[.3, .11-.06j], [.11+.06j, .25]],
                       [[.1, .06+.08j], [.06-.08j, .18]]]])
    cross = jnp.array([[[.31+.04j, .17-.09j], [.07+.12j, .22-.03j]],
                       [[.12-.08j, .04+.11j], [.16-.03j, .19+.07j]]])
    transfer = jnp.stack((cross, cross.conj().swapaxes(-1, -2)))
    factors = (scale*zero, scale*transfer)
    table = np.array([[0, 1], [1, 0]])
    diagonal = jax.vmap(jnp.diag)(energy)
    model = source.PeriodicGWModel(
        energy=energy, coeff=jnp.tile(jnp.eye(2, dtype=jnp.complex128), (2, 1, 1)),
        hcore=diagonal, reference_potential=jnp.zeros_like(diagonal),
        reference_fock=diagonal, nocc=1,
        kpoints=np.array([[0., 0., 0.], [-np.pi, 0., 0.]]),
        lattice=np.eye(3), table=table, inverse=np.argsort(table, axis=0),
        factors=factors, madelung=madelung,
        metadata={'fc': False, 'scope': 'two-k finite Hamiltonian'},
    )
    density = jnp.tile(jnp.diag(jnp.array([1., 0.])), (2, 1, 1))
    hartree, exchange = source.hartree_exchange(model, density)
    # This Hamiltonian has the declared HF reference without an offset term.
    return replace(model, hcore=diagonal-hartree-exchange,
                   reference_potential=exchange)


def test_complex_model_hartree_exchange_has_one_k_weight_and_density_madelung():
    source = _source()
    model = _model()
    density = np.array([[[.8, .12+.09j], [.12-.09j, .2]],
                        [[.7, -.08+.03j], [-.08-.03j, .3]]])
    nk, nmo = model.energy.shape
    j_expected = np.zeros((nk, nmo, nmo), complex)
    x_expected = np.zeros_like(j_expected)
    zero = np.asarray(model.factors[0])
    for k in range(nk):
        for a in range(nmo):
            for b in range(nmo):
                for l in range(nk):
                    for m in range(nmo):
                        for n in range(nmo):
                            j_expected[k, a, b] += (2/nk*density[l, m, n]
                                *np.vdot(zero[k, :, b, a], zero[l, :, n, m]))
                for q, factor in enumerate(model.factors):
                    km = model.inverse[k, q]
                    vertex = np.asarray(factor)[km]
                    for m in range(nmo):
                        for n in range(nmo):
                            x_expected[k, a, b] -= (density[km, m, n]/nk
                                *np.vdot(vertex[:, m, a], vertex[:, n, b]))
        x_expected[k] -= model.madelung*density[k]
    actual_j, actual_x = source.hartree_exchange(model, jnp.asarray(density))
    np.testing.assert_allclose(actual_j, j_expected, atol=2e-14, rtol=0)
    np.testing.assert_allclose(actual_x, x_expected, atol=2e-14, rtol=0)
    assert np.max(abs(np.asarray(actual_x).imag)) > 1e-5
    np.testing.assert_allclose(actual_j, np.asarray(actual_j).conj().swapaxes(-1, -2), atol=2e-14)
    np.testing.assert_allclose(actual_x, np.asarray(actual_x).conj().swapaxes(-1, -2), atol=2e-14)


def test_pair_compression_preserves_cross_k_gram_and_reports_discarded_error():
    source = _source()
    rng = np.random.default_rng(41)
    left = rng.normal(size=(7, 3))+1j*rng.normal(size=(7, 3))
    right = rng.normal(size=(3, 8))+1j*rng.normal(size=(3, 8))
    original = (left@right).reshape(7, 2, 2, 2).transpose(1, 0, 2, 3)
    reduced, info = source.compress_pair_factors(jnp.asarray(original), tolerance=1e-12)
    before = original.transpose(1, 0, 2, 3).reshape(7, -1)
    after = np.asarray(reduced).transpose(1, 0, 2, 3).reshape(reduced.shape[1], -1)
    np.testing.assert_allclose(after.conj().T@after, before.conj().T@before,
                               atol=2e-11, rtol=2e-13)
    assert reduced.shape[1] == 3
    assert info['relative_gram_residual'] < 1e-12
    assert info['rank'] == 3


def test_full_response_cache_retains_complex_positive_and_negative_branches():
    source = _source()
    model = _model()
    cache = source.build_correlation(model, model.energy, nw=8, eta=.001)
    q = 1
    frequency = float(np.asarray(cache['frequencies'])[2])
    factor = np.asarray(model.factors[q])
    pi = np.zeros((factor.shape[1], factor.shape[1]), complex)
    for k in range(2):
        other = model.table[k, q]
        for m, n, sign in ((0, 1, 1), (1, 0, -1)):
            vertex = factor[k, :, m, n]
            denominator = 1j*frequency+model.energy[k, m]-model.energy[other, n]
            pi += 2/2*sign*np.outer(vertex, vertex.conj())/denominator
    wc = np.linalg.solve(np.eye(pi.shape[0])-pi, pi)
    target, internal = 0, model.inverse[0, q]
    vertex = factor[internal]
    expected = np.einsum('Pmp,PQ,Qmr->mpr', vertex.conj(), wc, vertex)/2
    actual = cache['positive'][q][2, target]
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-12)
    assert np.max(abs(np.asarray(cache['positive'][q])-np.asarray(cache['negative'][q]))) > 1e-10


@pytest.mark.parametrize('bands', [(0,), (0, 1)])
@pytest.mark.parametrize('frequency', [-1.1, .9])
def test_projected_diagonal_correlation_matches_full_matrix_and_frequency_derivative(bands, frequency):
    """Project before a real-pole solve without changing either CD branch.

    Both frequencies have active G residues; their signs exercise both
    time-ordered finite-eta Lehmann conventions. An all-band cluster also
    checks the complete-space trace average used for KS degeneracy.
    """
    source = _source()
    model = _model(scale=.4)
    cache = source.build_correlation(model, model.energy, nw=8, eta=.003)
    omega = jnp.asarray(frequency)
    selected = jnp.asarray(bands)
    def full_trace(value):
        matrix = source.correlation_at(cache, 0, value)
        return jnp.mean(jnp.diag(matrix)[selected])
    projected = lambda value: source.correlation_diagonal(cache, 0, value, bands)
    expected, expected_derivative = jax.jvp(full_trace, (omega,), (jnp.ones_like(omega),))
    actual, derivative = jax.jvp(projected, (omega,), (jnp.ones_like(omega),))
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-12)
    np.testing.assert_allclose(derivative, expected_derivative, atol=2e-13, rtol=2e-11)


@pytest.mark.parametrize('kernel_name', ['_screening_frequencies', '_correlation_kernel'])
def test_numerical_cache_updates_reuse_compiled_source_kernels(kernel_name):
    """Changed G/W numbers must not capture a new executable per outer step."""
    source = _source()
    kernel = getattr(source, kernel_name)
    kernel.clear_cache()
    model = _model(scale=.4)
    first = source.build_correlation(model, model.energy, nw=4, eta=.003)
    first_value = source.correlation_at(first, 0, jnp.asarray(.9))
    first_value.block_until_ready()
    compiled = kernel._cache_size()
    assert compiled > 0
    second = source.build_correlation(model, model.energy+jnp.array([[-.06, .04], [-.03, .02]]),
                                       nw=4, eta=.003)
    second_value = source.correlation_at(second, 0, jnp.asarray(.9))
    second_value.block_until_ready()
    assert kernel._cache_size() == compiled
    # Reusing an executable cannot freeze the old screening or G poles.
    assert np.max(abs(np.asarray(second_value-first_value))) > 1e-10


@pytest.mark.parametrize('method', ['g0w0', 'evgw', 'qsgw'])
def test_zero_interaction_all_k_methods_recover_hamiltonian(method):
    model = _model(scale=0., madelung=0.)
    if method == 'qsgw':
        driver = importlib.import_module('gradscf.gw.pbc.qsgw').qsgw_cd_kpoints
    else:
        module = importlib.import_module('gradscf.gw.pbc.evgw')
        driver = module.g0w0_model if method == 'g0w0' else module.evgw_cd_kpoints
    result = driver(model, nw=8, eta=.001)
    assert bool(result.converged)
    np.testing.assert_allclose(result.mo_energy, model.energy, atol=2e-12, rtol=0)
    np.testing.assert_allclose(result.screening_energy, model.energy, atol=2e-12, rtol=0)
    assert np.asarray(result.qp_computed_mask).all()
    assert np.asarray(result.converged_mask).all()
    assert result.method.startswith('periodic_')


@pytest.mark.parametrize('diis_space', [0, 6])
def test_evgw_rebuilds_both_g_and_w_and_checks_the_final_onshell_equation(diis_space):
    source = _source()
    module = importlib.import_module('gradscf.gw.pbc.evgw')
    # A resolved interaction contrast keeps the EV-versus-one-shot change
    # above the 1e-9 Ha outer tolerance without weakening its assertion.
    model = _model(scale=.6)
    result = module.evgw_cd_kpoints(model, nw=16, eta=.001, max_iter=30,
                                    tol=1e-9, damping=.15, diis_space=diis_space)
    assert bool(result.converged)
    np.testing.assert_allclose(result.screening_energy, result.mo_energy, atol=0, rtol=0)
    residual = source.diagonal_dyson_residual(model, result.mo_energy, nw=16, eta=.001)
    assert np.max(abs(np.asarray(residual))) < 1e-9
    one_shot = module.g0w0_model(model, nw=16, eta=.001, self_energy='diagonal')
    assert np.max(abs(np.asarray(result.mo_energy-one_shot.mo_energy))) > 1e-8
    assert np.max(abs(np.asarray(result.mo_energy-model.energy))) > 1e-7


def test_evgw_onshell_jvp_reuses_its_complex_value_for_diagnostics(monkeypatch):
    """A value-and-slope evaluation must not repeat an expensive CD solve."""
    source = _source()
    module = importlib.import_module('gradscf.gw.pbc.evgw')
    model = _model()
    calls = []
    intercept = jnp.array([.013+.004j, -.017+.006j])
    slope = jnp.array([.21+.03j, -.16-.02j])
    def scalar(k, omega):
        calls.append(k)
        return intercept[k]+slope[k]*omega
    monkeypatch.setattr(module, 'build_correlation', lambda *args, **kwargs: {})
    monkeypatch.setattr(module, 'correlation_at',
        lambda cache, k, omega: scalar(k, omega)*jnp.eye(2), raising=False)
    monkeypatch.setattr(module, 'correlation_diagonal',
        lambda cache, k, omega, bands: scalar(k, omega), raising=False)
    energy = model.energy+jnp.array([[-.02, .03], [-.01, .02]])
    result = module._evaluate_diagonal(model, energy, nw=8, eta=.001)
    sigma = intercept[:, None]+slope[:, None]*energy
    _, exchange = source.hartree_exchange(model, model.reference_density)
    shift = jax.vmap(jnp.diag)(exchange-model.reference_potential).real
    expected = energy-model.energy-shift-sigma.real
    np.testing.assert_allclose(result.qp_residual, expected, atol=2e-14, rtol=0)
    np.testing.assert_allclose(result.sigma_qp, sigma, atol=2e-14, rtol=0)
    np.testing.assert_allclose(result.qp_weight,
        jnp.broadcast_to(1/(1-slope.real[:, None]), energy.shape), atol=2e-14, rtol=0)
    assert len(calls) == energy.size


@pytest.mark.parametrize('diis_space', [0, 6])
def test_qsgw_effective_fock_is_complex_mode_a_in_the_returned_orbital_frame(diis_space):
    module = importlib.import_module('gradscf.gw.pbc.qsgw')
    model = _model(scale=.3)
    result = module.qsgw_cd_kpoints(model, nw=16, eta=.001, max_iter=50,
                                   tol=1e-8, tol_density=1e-8, damping=.2,
                                   diis_space=diis_space)
    rotation = jnp.einsum('kai,kaj->kij', model.coeff.conj(), result.mo_coeff)
    actual = module.effective_fock(model, result.mo_energy, rotation, nw=16, eta=.001)
    diagonal = jax.vmap(jnp.diag)(result.mo_energy)
    np.testing.assert_allclose(actual, diagonal, atol=1e-8, rtol=0)
    assert np.max(abs(np.asarray(result.mo_coeff).imag)) > 1e-7
    assert np.max(abs(np.asarray(result.mo_energy-model.energy))) > 1e-7


def test_periodic_evgw_and_qsgw_explicitly_reject_automatic_differentiation():
    model = _model(scale=0., madelung=0.)
    evgw = importlib.import_module('gradscf.gw.pbc.evgw').evgw_cd_kpoints
    qsgw = importlib.import_module('gradscf.gw.pbc.qsgw').qsgw_cd_kpoints
    for driver in (evgw, qsgw):
        with pytest.raises(NotImplementedError, match='eager|AD|differentiat'):
            jax.jit(lambda energy: driver(replace(model, energy=energy), nw=4).mo_energy)(model.energy)


def test_native_source_hartree_matches_actual_scf_without_a_frozen_offset():
    source = _source()
    from gradscf.pbc import dft, gto
    cell = gto.M(atom='H 2 2 1.3; H 2 2 2.7', a=np.eye(3)*5., unit='Bohr',
        basis='gth-szv', pseudo='gth-pbe', mesh=(5, 5, 5), precision=1e-10)
    mf = dft.KRKS(cell, kpts=cell.make_kpts((2, 1, 1)), xc='pbe',
        temperature=None, conv_tol=1e-11, conv_tol_density=1e-10,
        conv_tol_grad=1e-9, max_cycle=150).run()
    assert mf.converged
    model = source.build_periodic_model(mf, tolerance=1e-12)
    assert model.metadata['reference_hartree_residual'] < 1e-9
    assert model.metadata['fc'] is False
    assert len(model.factors) == 2
    with pytest.raises(NotImplementedError, match='fc|cell'):
        source.build_periodic_model(mf, fc=True)


def test_native_gamma_complex_density_exchange_matches_direct_fft_contraction():
    """Native FFT K is independent of the factor-model index contraction.

    Gamma has no reciprocal wrap mismatch, so the native raw-difference and
    source-factor Coulomb windows agree. A complex off-diagonal density is
    essential: a diagonal occupation cannot expose rho versus rho.T.
    """
    source = _source()
    from gradscf.pbc import dft, gto
    from gradscf.integrals.periodic.fft import get_kpoint_jk
    cell = gto.M(atom='H 2 2 1.3; H 2 2 2.7', a=np.eye(3)*5., unit='Bohr',
        basis='gth-szv', pseudo='gth-pbe', mesh=(5, 5, 5), precision=1e-10)
    mf = dft.KRKS(cell, kpts=cell.make_kpts((1, 1, 1)), xc='pbe',
        temperature=None, conv_tol=1e-11, conv_tol_density=1e-10,
        conv_tol_grad=1e-9, max_cycle=150).run()
    assert mf.converged
    model = source.build_periodic_model(mf, tolerance=1e-12)
    density = jnp.array([[[.75, .12+.19j], [.12-.19j, .25]]])
    ao_density = model.coeff@density@model.coeff.conj().swapaxes(-1, -2)
    native_j, native_k = get_kpoint_jk(mf.inputs, jnp.stack((ao_density, ao_density)),
                                      mesh=mf.cell.mesh, exxdiv='ewald')
    project = lambda matrix: model.coeff.conj().swapaxes(-1, -2)@matrix@model.coeff
    actual_j, actual_x = source.hartree_exchange(model, density)
    np.testing.assert_allclose(actual_j, project(native_j), atol=2e-11, rtol=0)
    np.testing.assert_allclose(actual_x, -project(native_k[0]), atol=2e-11, rtol=0)
    assert np.max(abs(np.asarray(actual_x).imag)) > 1e-5
