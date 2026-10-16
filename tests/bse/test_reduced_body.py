"""Low-D caller metrics, signed finite-head BSE and complex optical measures."""
import importlib
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.bse import pbc as bse
from gradscf.gw.pbc import product_basis as products
from gradscf.integrals.periodic.coulomb import ReducedBoundary, coulomb_kernel


def model(dimension, *, vacuum=None):
    mesh = (3, 5, 5)
    lattice = np.diag([5., 16., 16.] if dimension == 1 else [5., 6., 18.])
    if vacuum is not None:
        lattice[1, 1] = lattice[2, 2] = vacuum
    reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    indices = np.stack(np.meshgrid(*[np.fft.fftfreq(n)*n for n in mesh], indexing='ij'), -1).reshape(-1, 3)
    coords = np.stack(np.meshgrid(*[np.arange(n)/n for n in mesh], indexing='ij'), -1).reshape(-1, 3)@lattice
    volume, nmo = np.linalg.det(lattice), 4
    rng = np.random.default_rng(20261016)
    values = np.stack([np.linalg.qr(rng.normal(size=(len(coords), nmo))
        +1j*rng.normal(size=(len(coords), nmo)))[0] for _ in range(2)])*np.sqrt(len(coords)/volume)
    ao = np.zeros((2, 4, len(coords), nmo), dtype=complex)
    ao[:, 0] = values
    data = SimpleNamespace(ao=ao, overlap=np.broadcast_to(np.eye(nmo), (2, nmo, nmo)),
        lattice=lattice, coords=coords, gvectors=indices@reciprocal,
        kpoints=np.array([[0., 0., 0.], [-.5, 0., 0.]])@reciprocal,
        weights=np.full(len(coords), volume/len(coords)), kweights=np.full(2, .5),
        boundary=ReducedBoundary(jnp.asarray(lattice), dimension, mesh))
    c = np.broadcast_to(np.eye(nmo), (2, nmo, nmo))
    energy = np.array([[-1., -.6, .2, .7], [-1.1, -.5, .3, .8]])
    return data, c, energy, np.array([[0, 1], [1, 0]]), mesh, 2


def full_auxiliary_oracle(case, *, screened=True):
    """Tiny independent signed-v auxiliary Dyson inverse, only for unit tests."""
    data, c, energy, table, mesh, no = case
    nk, nmo = energy.shape
    volume, ngrid = np.linalg.det(data.lattice), len(data.coords)
    psi = np.einsum('kgp,kpm->kgm', data.ao[:, 0], c)
    occupation = np.arange(nmo) < no
    all_pairs = [(m, n) for m in range(nmo) for n in range(nmo) if occupation[m] != occupation[n]]
    rho, interactions, bare = [], [], []
    for q in range(nk):
        blocks = []
        for k, partner in enumerate(table[:, q]):
            pair = psi[k].conj()[:, :, None]*psi[partner][:, None, :]
            pair = pair*np.exp(1j*data.coords@(data.kpoints[partner]-data.kpoints[k]+data.kpoints[q]))[:, None, None]
            blocks.append(np.fft.fftn(pair.reshape(*mesh, nmo, nmo), axes=(0, 1, 2))
                .reshape(ngrid, nmo, nmo)*np.sqrt(volume)/ngrid)
        blocks = np.asarray(blocks)
        raw = np.asarray(coulomb_kernel(jnp.asarray(data.gvectors-data.kpoints[q]), boundary=data.boundary))
        interaction = np.diag(raw)
        chi = np.zeros((ngrid, ngrid), dtype=complex)
        if screened:
            for k, partner in enumerate(table[:, q]):
                for m, n in all_pairs:
                    weight = 2*(int(occupation[m])-int(occupation[n]))/(nk*(energy[k, m]-energy[partner, n]))
                    chi += weight*np.outer(blocks[k, :, m, n], blocks[k, :, m, n].conj())
        rho.append(blocks)
        bare.append(interaction)
        interactions.append(np.linalg.solve(np.eye(ngrid)-interaction@chi, interaction))
    space = [(k, v, a) for k in range(nk) for v in range(no) for a in range(no, nmo)]
    matrix = np.empty((len(space), len(space)), dtype=complex)
    for i, (k, v, a) in enumerate(space):
        for j, (kp, vp, ap) in enumerate(space):
            q = np.flatnonzero(table[k] == kp)[0]
            exchange = np.vdot(rho[0][k, :, v, a], bare[0]@rho[0][kp, :, vp, ap])
            direct = np.vdot(rho[q][k, :, v, vp], interactions[q]@rho[q][k, :, a, ap])
            matrix[i, j] = (2*exchange-direct)/nk+(energy[k, a]-energy[k, v] if i == j else 0.)
    return matrix


@pytest.mark.parametrize('dimension', [1, 2])
def test_lowdim_caller_rejects_nonzero_raw_fft_head_even_with_identity_ao_metric(dimension):
    data, c, _, table, mesh, _ = model(dimension)
    data.ao[:, 0, :, 2] += .1*data.ao[:, 0, :, 0]
    # C†S C is still I, but the actual sampled OV/VO Fourier head is not zero.
    with pytest.raises(ValueError, match='FFT|sampled|grid'):
        products.kpoint_product_factor_builder(data, c, mesh=mesh, momentum_table=table)


def test_lowdim_caller_checks_supplied_ao_metric_too():
    data, c, _, table, mesh, _ = model(1)
    data.overlap = 2*data.overlap
    with pytest.raises(ValueError, match='metric|orthonormal'):
        products.kpoint_product_factor_builder(data, c, mesh=mesh, momentum_table=table)


@pytest.mark.parametrize('dimension', [1, 2])
def test_lowdim_finite_head_requires_an_exact_zero_source_q(dimension):
    data, c, e, table, mesh, no = model(dimension)
    data.kpoints = data.kpoints.copy()
    data.kpoints[0, 0] = 8e-11  # Within the existing BSE near-Gamma tolerance.
    with pytest.raises(ValueError, match='exact.zero'):
        products.kpoint_product_factor_builder(data, c, mesh=mesh, momentum_table=table)
    with pytest.raises(ValueError, match='exact.zero'):
        bse.build_tda_matrix(data, c, e, e, no, mesh=mesh)


@pytest.mark.parametrize('dimension', [1, 2])
def test_mixed_lowdim_checks_each_sampled_frame_without_assuming_cross_identity(dimension):
    data, _, _, _, mesh, _ = model(dimension)
    left = data.ao[0, 0]
    rotation = np.linalg.qr(np.random.default_rng(8).normal(size=(4, 4)))[0]
    right = left@rotation
    assert abs(rotation[0, 2]) > .01
    arguments = dict(q_cart=np.zeros(3), left_k=np.zeros(3), right_k=np.zeros(3),
                     pairs=np.array([[0, 2], [1, 3]]), mesh=mesh)
    actual = products.mixed_product_factors(data, left, right, **arguments)
    assert np.isfinite(actual).all()
    with pytest.raises(ValueError, match='FFT|sampled|grid'):
        products.mixed_product_factors(data, left, right*1.01, **arguments)


@pytest.mark.parametrize('dimension', [1, 2])
def test_vacuum_reciprocal_momenta_are_rejected_before_aliasing(dimension):
    data, c, e, table, mesh, no = model(dimension)
    vacuum_vector = (2*np.pi*np.linalg.inv(data.lattice).T)[dimension]
    bad = SimpleNamespace(**vars(data))
    bad.kpoints = data.kpoints.copy()
    bad.kpoints[0] += vacuum_vector
    with pytest.raises(ValueError, match='nonperiodic'):
        products.kpoint_product_factor_builder(bad, c, mesh=mesh, momentum_table=table)
    with pytest.raises(ValueError, match='nonperiodic'):
        bse.build_tda_matrix(bad, c, e, e, no, mesh=mesh)
    values = data.ao[0, 0]
    for q, k in [(vacuum_vector, np.zeros(3)), (np.zeros(3), vacuum_vector)]:
        with pytest.raises(ValueError, match='nonperiodic'):
            products.mixed_product_factors(data, values, values, q_cart=q, left_k=k, right_k=k,
                                           pairs=np.array([[0, 2]]), mesh=mesh)


@pytest.mark.parametrize('dimension', [1, 2])
@pytest.mark.parametrize('screened', [False, True])
def test_reduced_bse_matches_signed_full_auxiliary_oracle(dimension, screened):
    case = model(dimension)
    data, c, e, _, mesh, no = case
    actual = bse.build_tda_matrix(data, c, e, e, no, mesh=mesh, screened=screened)
    expected = full_auxiliary_oracle(case, screened=screened)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-10)


def test_positive_wire_head_is_not_restored_twice():
    case = model(1, vacuum=.5)
    data, c, e, _, mesh, no = case
    assert float(coulomb_kernel(jnp.asarray(data.gvectors), boundary=data.boundary)[0]) > 0.
    actual = bse.build_tda_matrix(data, c, e, e, no, mesh=mesh)
    np.testing.assert_allclose(actual, full_auxiliary_oracle(case), atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize('dimension', [1, 2])
def test_reduced_bse_retains_phase_and_k_order_covariance(dimension):
    data, c, e, _, mesh, no = model(dimension)
    phase = np.exp(1j*np.random.default_rng(91).normal(size=e.shape))
    original = bse.build_tda_matrix(data, c, e, e, no, mesh=mesh)
    rotated = bse.build_tda_matrix(data, c*phase[:, None, :], e, e, no, mesh=mesh)
    transition = (phase[:, None, no:]/phase[:, :no, None]).reshape(-1)
    np.testing.assert_allclose(rotated, transition.conj()[:, None]*original*transition, atol=2e-11)
    order = np.array([1, 0])
    permuted = SimpleNamespace(**vars(data))
    for name in ('ao', 'overlap', 'kpoints', 'kweights'):
        setattr(permuted, name, getattr(data, name)[order])
    matrix = bse.build_tda_matrix(permuted, c[order], e[order], e[order], no, mesh=mesh)
    rows = np.arange(len(original)).reshape(2, -1)[order].reshape(-1)
    np.testing.assert_allclose(matrix, original[np.ix_(rows, rows)], atol=2e-11)


def test_optional_numpy_reference_uses_signed_transition_space():
    oracle = importlib.import_module('gradscf_tools.periodic_bse_reference')
    case = model(1)
    data, c, e, _, mesh, no = case
    actual = oracle.matrix(data, c, e, e, no, mesh=mesh)
    np.testing.assert_allclose(actual, full_auxiliary_oracle(case), atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize('dimension,measure', [(1, 4.), (2, 20.), (3, 120.)])
def test_complex_polarizability_static_limit_and_periodic_measure(dimension, measure):
    dipoles = jnp.array([[1., 1j, 0.]])
    actual = bse.polarizability(jnp.array([2.]), dipoles, 0., eta=0.,
                               lattice=jnp.diag(jnp.array([4., 5., 6.])), dimension=dimension)
    np.testing.assert_allclose(actual, np.diag([1., 1., 0.])/measure, atol=1e-14)


@pytest.mark.parametrize('dimension', [1, 2])
def test_polarizability_complex_phase_vacuum_invariance_and_retarded_poles(dimension):
    energy = jnp.array([.7, 1.2])
    dipoles = jnp.array([[.2+.1j, .3j, .1], [.1, .2, -.1j]])
    frequency = jnp.array([0., .5, 1.])
    lattice = jnp.diag(jnp.array([4., 5., 6.]))
    original = bse.polarizability(energy, dipoles, frequency, eta=.03, lattice=lattice, dimension=dimension)
    phase = jnp.exp(1j*jnp.array([.4, -.8]))
    padded = lattice.at[dimension:, :].multiply(2.)
    changed = bse.polarizability(energy, dipoles*phase[:, None], frequency, eta=.03,
                                lattice=padded, dimension=dimension)
    np.testing.assert_allclose(changed, original, atol=1e-14)
    reverse = bse.polarizability(energy, dipoles, -frequency, eta=.03, lattice=lattice, dimension=dimension)
    np.testing.assert_allclose(reverse, original.conj(), atol=1e-14)
    assert np.all(np.diagonal(np.asarray(original[1:]).imag, axis1=-2, axis2=-1) > 0.)
