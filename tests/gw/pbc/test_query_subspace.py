"""Complete query subspaces: nonlinear QPs and complex-frame covariance."""
import importlib
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers.nonlinear import ScalarRootConfig


query = importlib.import_module('gradscf.gw.pbc.query')


def test_only_complete_requested_degenerate_clusters_are_accepted():
    energy = np.array([[-.7, -.4, -.4, .2, .5], [-.8, -.5, -.3, .1, .6]])
    groups = query._query_subspaces(energy, np.array([1, 2, 3]), 1e-7)
    assert groups == [((1, 2), (3,)), ((1,), (2,), (3,))]
    with pytest.raises(ValueError, match='complete.*subspace'):
        query._query_subspaces(energy, np.array([2, 3]), 1e-7)


def test_nonlinear_complete_subspace_matches_exact_matrix_roots():
    # Sigma(omega) = A + beta*omega*I. No linearization is used by the solve;
    # this analytically soluble case checks all QP branches and matrix terms.
    energy = jnp.array([-.45, -.45])
    exchange = jnp.array([[-.12, .015+.023j], [.015-.023j, -.07]])
    vxc = jnp.array([[-.1, -.011+.007j], [-.011-.007j, -.08]])
    a = jnp.array([[.02, .03-.017j], [.03+.017j, -.025]])
    beta = .17
    imaginary = jnp.array([[.008, .001+.003j], [.001-.003j, .012]])
    sigma = lambda omega: a+beta*omega*jnp.eye(2)+1j*imaginary
    config = ScalarRootConfig(ftol=1e-11, xtol=1e-11, maxiter=80)
    result = query._solve_subspace_qp(energy, exchange, vxc, sigma,
                                    config=config, occupied=True)
    exact = np.linalg.eigvalsh(np.diag(energy)+exchange-vxc+a)/(1-beta)
    np.testing.assert_allclose(result['roots'], exact, atol=2e-11, rtol=0)
    assert np.asarray(result['converged']).all()
    np.testing.assert_allclose(result['weight'], 1/(1-beta), atol=2e-12, rtol=0)
    np.testing.assert_array_equal(result['branch_valid'], [True, True])
    assert np.asarray(result['candidate_count']).min() >= 1
    for p, omega in enumerate(np.asarray(result['roots'])):
        vector = np.asarray(result['vectors'])[:, p]
        matrix = np.diag(energy)+exchange-vxc+a+beta*omega*np.eye(2)
        np.testing.assert_allclose(matrix@vector, omega*vector, atol=2e-11, rtol=0)
    assert np.max(np.asarray(result['projected_residual'])) < 2e-11
    np.testing.assert_allclose(np.asarray(result['sigma']).imag.sum(),
                               np.trace(np.asarray(imaginary)).real, atol=2e-11)

    # A genuine complex unitary changes off-diagonal matrix elements and
    # leaves the complete-subspace quasiparticle spectrum unchanged.
    frame = np.array([[1., 1j], [1j, 1.]])/np.sqrt(2.)
    rotate = lambda matrix: frame.conj().T@matrix@frame
    rotated = query._solve_subspace_qp(energy, rotate(exchange), rotate(vxc),
        lambda omega: rotate(sigma(omega)), config=config, occupied=True)
    np.testing.assert_allclose(rotated['roots'], result['roots'], atol=2e-11, rtol=0)
    np.testing.assert_allclose(rotated['sigma'], result['sigma'], atol=2e-11, rtol=0)
    assert np.asarray(rotated['converged']).all()


def test_subspace_failed_root_keeps_failure_certificate():
    energy = jnp.array([-.4, -.4])
    zero = jnp.zeros((2, 2), dtype=jnp.complex128)
    result = query._solve_subspace_qp(energy, zero, zero,
        lambda omega: jnp.eye(2)*(.3+.2*omega**2),
        config=ScalarRootConfig(ftol=1e-12, xtol=1e-12, maxiter=1), occupied=True)
    assert not np.asarray(result['converged']).any()
    assert np.min(np.asarray(result['projected_residual'])) > 1e-12


def test_frequency_degeneracy_with_splitting_derivative_is_not_a_gauge_independent_branch():
    energy = jnp.zeros(2)
    zero = jnp.zeros((2, 2), dtype=jnp.complex128)
    derivative = jnp.diag(jnp.array([.2, .4]))
    config = ScalarRootConfig(method='hybrid', ftol=1e-10, xtol=1e-10)
    for frame in (jnp.eye(2), jnp.array([[1., 1j], [1j, 1.]])/jnp.sqrt(2.)):
        sigma = lambda omega: omega*(frame.conj().T@derivative@frame)
        result = query._solve_subspace_qp(energy, zero, zero, sigma,
                                         config=config, occupied=False)
        assert np.max(np.abs(result['residual'])) < 1e-10
        assert not np.asarray(result['branch_valid']).any()
        assert not np.asarray(result['converged']).any()
        assert np.isnan(np.asarray(result['weight'])).all()


@pytest.mark.parametrize('touching', [False, True])
@pytest.mark.parametrize('splitting', [0., .04])
@pytest.mark.parametrize('self_energy', ['matrix', 'diagonal'])
def test_complete_query_subspace_preserves_coverage_and_ks_frame(monkeypatch, touching, splitting,
                                                               self_energy):
    pair = np.array([[[.18+.02j, .03-.04j], [.07+.03j, .13+.01j], [.01, .02]],
                     [[.09-.01j, -.02+.03j], [.04+.02j, .08-.01j], [.02j, .01j]]])
    vxc = np.array([[-.1, .012+.018j], [.012-.018j, -.07]])
    spectrum = np.array([-.3, 0., 0., .3]) if touching else np.array([-.5, -.5, .5])
    selected = [1, 2] if touching else [0, 1]
    if splitting:
        # The same requested matrix window must retain its off-diagonal
        # operator when a KS degeneracy splits. A cluster-only switch drops
        # these terms and introduces a different Dyson approximation.
        if touching:
            spectrum[1], spectrum[2] = -splitting/2, splitting/2
        else:
            spectrum[1] += splitting
    nmo = len(spectrum)
    fractions = np.array([1., .5, .5, 0.]) if touching else np.array([1., 1., 0.])
    if touching:
        pair = np.concatenate([pair, np.zeros((2, 1, 2))], axis=1)
    inputs = SimpleNamespace(boundary=None, kpoints=jnp.zeros((1, 3)), weights=jnp.ones(nmo))
    mf = SimpleNamespace(inputs=inputs, cell=SimpleNamespace(mesh=(nmo, 1, 1)), exxdiv=None,
        result=SimpleNamespace(mo_coeff_spin=jnp.tile(jnp.eye(nmo), (2, 1, 1, 1))))
    monkeypatch.setattr(query, 'build_band_inputs', lambda *a, **k:
                         SimpleNamespace(ao=jnp.eye(nmo)[None, None]))
    monkeypatch.setattr(query, 'kpoint_product_factor_builder', lambda *a, **k: None)
    monkeypatch.setattr(query, 'kpoint_transition_block', lambda *a, **k:
        dict(factors=jnp.zeros((2, 2), complex), gram=jnp.zeros((2, 2), complex),
             differences=jnp.array([-1., 1.]), signs=jnp.array([1., -1.])))

    def evaluate(frame):
        coefficients = np.eye(nmo, dtype=complex)
        coefficients[np.ix_(selected, selected)] = frame
        group = tuple(selected)
        data = dict(energy=jnp.asarray(spectrum[None]), selected=np.array(selected), nocc=2,
            source_energy=jnp.asarray(spectrum[None]), coefficients=jnp.asarray(coefficients[None]),
            target_values=jnp.asarray(coefficients[None]),
            shifted_energy=jnp.asarray(spectrum[None, None]), shifted_coeff=jnp.eye(nmo)[None, None],
            shifted_occupation=jnp.asarray(fractions[None, None]) if touching else None,
            shifted=np.zeros((1, 1, 3)), canonical=np.zeros((1, 3)),
            vxc=jnp.zeros((1, nmo)).at[0, jnp.asarray(selected)].set(
                jnp.diag(jnp.asarray(frame.conj().T@vxc@frame)).real),
            vxc_blocks={(0, group): jnp.asarray(frame.conj().T@vxc@frame)},
            ef=0., table=np.zeros((1, 1), int))
        monkeypatch.setattr(query, '_prepare', lambda *a, **k: data)
        monkeypatch.setattr(query, 'mixed_product_factors', lambda *a, **k:
                             jnp.asarray((pair@frame).reshape(2, -1)))
        result = query.g0w0_bands(mf, np.zeros((1, 3)), selected, nw=8, band_touching=touching,
            self_energy=self_energy,
            root_config=ScalarRootConfig(ftol=1e-10, xtol=1e-10))
        expected_mask = np.isin(np.arange(nmo), selected)[None]
        np.testing.assert_array_equal(result.qp_computed_mask, expected_mask)
        assert np.asarray(result.converged_mask).all()
        assert np.asarray(result.qp_branch_valid)[expected_mask].all()
        np.testing.assert_allclose(np.asarray(result.qp_weight)[expected_mask], 1., atol=2e-13)
        assert np.asarray(result.qp_candidate_count)[expected_mask].min() >= 1
        vectors = np.asarray(result.qp_vectors)[0][np.ix_(selected, selected)]
        assert vectors.shape == (2, 2)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=0), 1., atol=2e-13)
        np.testing.assert_array_equal(result.mo_coeff, coefficients[None])
        np.testing.assert_array_equal(np.asarray(result.mo_energy)[~expected_mask], spectrum[None][~expected_mask])
        np.testing.assert_allclose(result.sigma_qp, 0., atol=2e-14)
        assert np.max(np.abs(result.qp_residual)) < 1e-10
        return np.asarray(result.mo_energy)[0, selected]

    original = evaluate(np.eye(2))
    exchange = -np.einsum('gmp,m,gmq->pq', pair.conj(), fractions, pair)
    if self_energy == 'matrix':
        exact = np.linalg.eigvalsh(np.diag(spectrum[selected])+exchange-vxc)
    elif splitting:
        exact = spectrum[selected]+np.diag(exchange-vxc).real
    else:
        exact = spectrum[selected]+np.trace(exchange-vxc).real/len(selected)
    np.testing.assert_allclose(original, exact, atol=2e-13, rtol=0)
    if not splitting:
        # Only an exactly degenerate KS Hamiltonian is unchanged by this
        # orbital rotation while the reported KS energies stay diagonal.
        frame = np.array([[1., 1j], [1j, 1.]])/np.sqrt(2.)
        np.testing.assert_allclose(evaluate(frame), original, atol=2e-13, rtol=0)


def test_negative_head_matrix_uses_actual_cross_frame_overlap():
    left = jnp.array([[1., 0.], [0., 1.]])
    right = jnp.array([[1., 1j], [1j, 1.]])/jnp.sqrt(2.)
    weights = jnp.ones(2)
    actual = query._negative_head_exchange(-.6, left, right, weights, 1, 3, matrix=True)
    overlap = np.asarray(left[:, :1]).conj().T@np.asarray(right)
    expected = .2*overlap.conj().T@overlap
    assert abs(expected[0, 1]) > .09
    np.testing.assert_allclose(actual, expected, atol=2e-15)
    np.testing.assert_allclose(query._negative_head_exchange(.6, left, right, weights, 1, 3, matrix=True),
                               np.zeros((2, 2)), atol=0)


def test_complex_transition_contour_matrix_matches_independent_numpy():
    from gradscf.gw.freq import scaled_legendre_grid
    from gradscf.gw.screened import solve_transition_screening
    from gradscf.gw.pbc.product_basis import lehmann_weights
    from gradscf.solvers import LinearSolverConfig

    gram = np.array([[.11, .019+.013j], [.019-.013j, .07]])
    projected = np.array([[[.12+.03j, .08-.01j], [.05-.02j, -.07+.01j]],
                          [[.04-.01j, .09+.05j], [.08+.03j, .06-.04j]]])
    differences, signs = np.array([-.8, .9]), np.array([1., -1.])
    energies, omega, eta, ef = np.array([-.4, .5]), .8, .013, 0.
    frequencies, quadrature = map(np.asarray, scaled_legendre_grid(24))
    config = LinearSolverConfig(method='direct', rtol=1e-12, atol=1e-14)

    def numpy_action(z, rhs, *, imaginary):
        weights = 2*signs/(z+differences+(0. if imaginary else 1j*eta*signs))
        return np.linalg.solve(np.eye(2)-weights[:, None]*gram,
                               weights[:, None]*rhs.reshape(2, -1)).reshape(rhs.shape)

    def contour(p):
        positive, negative = [], []
        for target, axis in ((positive, 1.), (negative, -1.)):
            for frequency in frequencies:
                weights = lehmann_weights(axis*frequency, differences, signs, 1)
                solved = solve_transition_screening(jnp.asarray(gram), weights,
                    jnp.asarray(p.reshape(2, -1)), config=config).solution.reshape(p.shape)
                actual = query._screened_subspace(jnp.asarray(p), solved, (0, 1))
                expected = np.einsum('rmp,rmq->mpq', p.conj(),
                                    numpy_action(1j*axis*frequency, p, imaginary=True))
                np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-12)
                target.append(np.asarray(actual))
        positive, negative = np.array(positive), np.array(negative)
        assert np.max(np.abs(positive-negative)) > 1e-5
        delta = omega-energies-1j*eta*np.sign(ef-energies)
        plus = delta[None, :]-1j*frequencies[:, None]
        minus = delta[None, :]+1j*frequencies[:, None]
        expected = -np.sum(quadrature[:, None, None, None]*(
            positive/plus[:, :, None, None]+negative/minus[:, :, None, None]), axis=(0, 1))/(2*np.pi)
        for m, energy in enumerate(energies):
            if ef < energy < omega:
                pole = omega-energy+1j*eta*np.sign(energy-ef)
                rhs = p[:, m]
                expected += rhs.conj().T@numpy_action(pole, rhs, imaginary=False)
        block = dict(gram=jnp.asarray(gram), differences=jnp.asarray(differences),
                     signs=jnp.asarray(signs), projected=jnp.asarray(p))
        actual = query._sigma_imag_full(omega, jnp.asarray(positive), jnp.asarray(negative),
            jnp.asarray(energies), ef, jnp.asarray(frequencies), jnp.asarray(quadrature), eta)
        actual += query._sigma_residue_full(omega, jnp.asarray(energies), [block], ef, eta, 1, config)
        np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=3e-12)
        assert np.max(np.abs(expected-expected.conj().T)) > 1e-6
        return np.asarray(actual)

    sigma = contour(projected)
    frame = np.array([[1., 1j], [1j, 1.]])/np.sqrt(2.)
    transformed = contour(projected@frame)
    np.testing.assert_allclose(transformed, frame.conj().T@sigma@frame, atol=3e-14, rtol=3e-12)
