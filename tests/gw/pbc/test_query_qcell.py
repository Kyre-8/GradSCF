"""Nonzero Gamma-cell quadrature and full shifted transition response."""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw.pbc.product_basis import kpoint_transition_block, lehmann_weights


q0 = importlib.import_module('gradscf.gw.pbc.q0')
query = importlib.import_module('gradscf.gw.pbc.query')


@pytest.mark.parametrize('order', [2, 4])
def test_gamma_cell_weights_integrate_volume_and_reciprocal_second_moment(order):
    lattice = np.array([[6., 0., 0.], [.8, 7., 0.], [.3, .5, 8.]])
    mesh = (3, 2, 1)
    points, weights = q0.gamma_cell_quadrature(lattice, mesh, order=order)
    reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    np.testing.assert_allclose(np.sum(weights), 1/np.prod(mesh), atol=2e-16)
    np.testing.assert_allclose(weights@points, 0., atol=2e-16)
    expected = reciprocal.T@np.diag(1/(12*np.asarray(mesh)**2))@reciprocal
    actual = np.einsum('q,qi,qj->ij', weights, points, points)*np.prod(mesh)
    np.testing.assert_allclose(actual, expected, atol=3e-16, rtol=2e-14)
    assert np.all(np.linalg.norm(points, axis=1) > 0.)
    assert points.shape == (order**3, 3)
    assert np.all(weights > 0.)


@pytest.mark.parametrize('order', [0, 1, 3, True])
def test_gamma_cell_rejects_orders_containing_singular_origin(order):
    with pytest.raises(ValueError, match='even|order'):
        q0.gamma_cell_quadrature(np.eye(3)*6., (2, 2, 2), order=order)


def test_shifted_full_transition_uses_distinct_partner_energies_and_matches_dense_response():
    source = jnp.array([[-.6, .4], [-.5, .7]])
    shifted = jnp.array([[-.58, .43], [-.53, .68]])
    vertices = np.array([[[.13+.02j, .06-.01j], [.03, .09+.04j]],
                         [[.07-.03j, .1+.02j], [.05+.02j, -.04+.01j]]])
    table = np.tile(np.arange(2)[:, None], (1, 2))
    def build(q, *, pairs, k_indices):
        assert q == 0
        np.testing.assert_array_equal(pairs, [[0, 1], [1, 0]])
        return jnp.asarray(vertices[np.asarray(k_indices)])
    block = kpoint_transition_block(build, source, 1, table, 0, full=True,
                                    partner_energy=shifted)
    expected_difference = np.array([-1.03, .98, -1.18, 1.23])
    expected_sign = np.array([1., -1., 1., -1.])
    expected_vertices = np.concatenate(vertices, axis=1)
    np.testing.assert_allclose(block['differences'], expected_difference, atol=2e-16)
    np.testing.assert_array_equal(block['signs'], expected_sign)
    np.testing.assert_allclose(block['factors'], expected_vertices, atol=0.)
    for frequency in (-.35, 0., .35):
        weights = lehmann_weights(frequency, block['differences'], block['signs'], 2)
        actual = block['factors']@jnp.diag(weights)@block['factors'].conj().T
        expected_weights = expected_sign/(1j*frequency+expected_difference)
        expected = expected_vertices@np.diag(expected_weights)@expected_vertices.conj().T
        np.testing.assert_allclose(actual, expected, atol=3e-16, rtol=2e-14)


def test_cell_residue_weights_are_distinct_from_source_k_normalization():
    # Two q-cell samples can share one response, while the screening source
    # has three k points. Neither the number of q samples nor their weights
    # may replace the 1/Nk in the independent-particle response.
    vector, pairs = np.array([.25, .17]), np.array([.15, .2])
    differences, signs = np.array([-.8, .8]), np.array([1., -1.])
    gram = np.outer(vector, vector)
    projected = vector[:, None]*pairs[None, :]
    energies, omega, eta, nk = np.array([-.4, .5]), .8, .013, 3
    blocks = [dict(gram=jnp.asarray(gram), projected=jnp.asarray(projected),
                   differences=jnp.asarray(differences), signs=jnp.asarray(signs),
                   q_weight=weight) for weight in (.1, .2)]
    frequency = omega-energies[1]
    chi = np.sum(vector**2*2*signs/(nk*(frequency+differences+1j*eta*(1+signs))))
    expected = .3*pairs[1]**2*chi/(1-chi)
    actual = query._sigma_residue_full(omega, jnp.tile(jnp.asarray(energies), 2),
        blocks, 0., eta, nk, None)
    np.testing.assert_allclose(actual, expected, atol=3e-16, rtol=2e-13)


def test_native_gamma_cell_correlation_matches_uncompressed_numpy_dielectric():
    """Independent full-G inversion checks the shifted-cell CD assembly."""
    from gradscf.pbc import gto, dft
    from gradscf.gw.freq import scaled_legendre_grid
    from gradscf.integrals.periodic.fft import build_band_inputs
    from gradscf.gw.pbc.product_basis import mixed_product_factors
    from gradscf.pbc.momentum import wrap_to_kmesh
    from gradscf.solvers.nonlinear import ScalarRootConfig

    cell = gto.M(atom='H 3 3 2.3; H 3 3 3.7', a=np.eye(3)*6., unit='Bohr',
        basis='gth-szv', pseudo='gth-pbe', mesh=(5, 5, 5), precision=1e-10)
    mf = dft.KRKS(cell, kpts=cell.make_kpts((1, 1, 1)), xc='pbe',
        conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-9, max_cycle=150).run()
    assert mf.converged
    nw, eta = 8, .001
    result = query.g0w0_bands(mf, np.zeros((1, 3)), [0], nw=nw, eta=eta,
        fc=True, q0_order=2, max_memory_bytes=128*1024**2,
        root_config=ScalarRootConfig(method='hybrid', ftol=1e-9, xtol=1e-9, maxiter=100))
    assert bool(result.converged)
    assert result.method.endswith('_qcell2')
    omega = float(result.mo_energy[0, 0])
    points, qweights = q0.gamma_cell_quadrature(mf.inputs.lattice, (1, 1, 1), order=2)
    reciprocal = 2*np.pi*np.linalg.inv(np.asarray(mf.inputs.lattice)).T
    wrap = lambda k: wrap_to_kmesh(k@np.linalg.inv(reciprocal))@reciprocal
    partner_points, internal_points = wrap(-points), wrap(points)
    partner_energy, partner_coeff = mf.get_bands(partner_points)
    internal_energy, internal_coeff = mf.get_bands(internal_points)
    source_energy = np.asarray(mf.mo_energy)[0]
    source_values = np.asarray(mf.inputs.ao[0, 0])@np.asarray(mf.mo_coeff)[0]
    partner_inputs = build_band_inputs(cell, partner_points, reference=mf.inputs)
    internal_inputs = build_band_inputs(cell, internal_points, reference=mf.inputs)
    partner_values = np.einsum('qga,qam->qgm', partner_inputs.ao[:, 0], partner_coeff)
    internal_values = np.einsum('qga,qam->qgm', internal_inputs.ao[:, 0], internal_coeff)
    all_energy = np.concatenate([source_energy[None], partner_energy, internal_energy])
    ef = (np.max(all_energy[:, 0])+np.min(all_energy[:, 1]))/2
    frequencies, quadrature = map(np.asarray, scaled_legendre_grid(nw))
    total = 0j
    for q, weight, response_energy, green_energy, response_values, green_values in zip(
            points, qweights, partner_energy, internal_energy, partner_values, internal_values):
        response = np.asarray(mixed_product_factors(mf.inputs, source_values, response_values,
            q_cart=q, left_k=np.zeros(3), right_k=wrap(-q), pairs=np.array([[0, 1], [1, 0]]),
            mesh=cell.mesh))
        pair = np.asarray(mixed_product_factors(mf.inputs, green_values, source_values,
            q_cart=q, left_k=wrap(q), right_k=np.zeros(3), pairs=np.array([[0, 0], [1, 0]]),
            mesh=cell.mesh))
        differences = source_energy[[0, 1]]-np.asarray(response_energy)[[1, 0]]
        signs = np.array([1., -1.])

        def screened(response_weights):
            dielectric = np.eye(len(response))-response@np.diag(response_weights)@response.conj().T
            action = np.linalg.solve(dielectric, pair)-pair
            return np.sum(pair.conj()*action, axis=0)

        positive = np.array([screened(2*signs/(1j*frequency+differences))
                             for frequency in frequencies])
        negative = np.array([screened(2*signs/(-1j*frequency+differences))
                             for frequency in frequencies])
        delta = omega-np.asarray(green_energy)-1j*eta*np.sign(ef-np.asarray(green_energy))
        value = -np.sum(quadrature[:, None]*(positive/(delta[None]-1j*frequencies[:, None])
                    +negative/(delta[None]+1j*frequencies[:, None])))/(2*np.pi)
        for m, energy in enumerate(np.asarray(green_energy)):
            if not (ef < energy < omega or omega < energy < ef):
                continue
            frequency = omega-energy
            pole_weights = 2*signs/(frequency+differences+1j*eta*(np.sign(frequency)+signs))
            value += np.sign(omega-ef)*screened(pole_weights)[m]
        total += weight*value
    np.testing.assert_allclose(result.sigma_qp[0, 0], total, atol=3e-10, rtol=3e-9)
